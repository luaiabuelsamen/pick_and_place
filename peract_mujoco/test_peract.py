import sys
import os
import numpy as np
import torch
import time
from PIL import Image

# Add required paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/home/jetson3/luai/peract/peract_colab')
sys.path.append('/home/jetson3/luai/peract')

# Import MuJoCo environment
from mujoco_parser import MuJoCoParserClass, init_env
from util import execute_peract_action
# Import necessary PerAct components
from arm.utils import stack_on_channel, discrete_euler_to_quaternion
from arm.optim.lamb import Lamb
from peract_mujoco.peract_class import PerceiverIO, VoxelGrid, PerceiverActorAgent

# Constants (matching your training settings)
CAMERAS = ['wrist', 'front', 'left_shoulder', 'right_shoulder']
IMAGE_SIZE = 128
LOW_DIM_SIZE = 4
VOXEL_SIZES = [100]
NUM_LATENTS = 512
SCENE_BOUNDS = [-5, -5, -5, 5, 5, 5]
ROTATION_RESOLUTION = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEPTH_SCALE = 2**24 - 1  # Same as in RLBench

def process_rgb_image(rgb_img):
    """Process RGB image for model input"""
    if not isinstance(rgb_img, Image.Image):
        rgb_img = Image.fromarray(rgb_img)
    
    rgb_img = rgb_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_array = np.array(rgb_img)
    
    # Convert to CHW format (channels first)
    if len(rgb_array.shape) == 3 and rgb_array.shape[2] >= 3:
        rgb_array = np.transpose(rgb_array[:, :, :3], (2, 0, 1))
    else:
        if len(rgb_array.shape) == 2:
            rgb_array = np.repeat(rgb_array[:, :, np.newaxis], 3, axis=2)
        rgb_array = np.transpose(rgb_array, (2, 0, 1))
    
    return rgb_array

def process_depth_image(depth_img, near=0.01, far=10.0):
    """
    Process depth image preserving metric values
    
    Args:
        depth_img: Depth image from MuJoCo
        near: Near plane distance in meters
        far: Far plane distance in meters
        
    Returns:
        Depth array in format (1, H, W) with metric values
    """
    # Convert PIL to numpy if needed
    if isinstance(depth_img, Image.Image):
        depth_img = np.array(depth_img)
    
    # Extract single channel depth
    if len(depth_img.shape) == 3:
        depth_img_2d = depth_img[:, :, 0]
    else:
        depth_img_2d = depth_img
    
    # Resize to target size if needed
    if depth_img_2d.shape[0] != IMAGE_SIZE or depth_img_2d.shape[1] != IMAGE_SIZE:
        depth_pil = Image.fromarray(depth_img_2d)
        depth_pil = depth_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        depth_img_2d = np.array(depth_pil)
    
    # No need to normalize - just make sure it's in the right range
    # MuJoCo typically provides depth in 0-1 range
    if np.max(depth_img_2d) > 1.0:
        depth_img_2d = depth_img_2d / np.max(depth_img_2d)
    
    # Convert normalized depth to metric depth using near and far planes
    metric_depth = near + depth_img_2d * (far - near)
    
    # Reshape to required format (1, H, W) for point cloud computation
    depth_array = metric_depth.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
    
    return depth_array

def depth_to_point_cloud(depth_img, camera_intrinsics, camera_extrinsics):
    """
    Convert depth image to point cloud using camera parameters
    
    This manually implements the functionality from VisionSensor.pointcloud_from_depth_and_camera_params
    """
    # Ensure depth image is 2D
    if len(depth_img.shape) == 3:
        depth_img = depth_img[0]  # Take first channel if it's (1, H, W)
    
    height, width = depth_img.shape
    
    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.reshape(-1)
    v = v.reshape(-1)
    z = depth_img.reshape(-1)
    
    # Filter invalid depths
    valid = z > 0.001
    u, v, z = u[valid], v[valid], z[valid]
    
    # Convert to normalized device coordinates
    x = (u - camera_intrinsics[0, 2]) / camera_intrinsics[0, 0] * z
    y = (v - camera_intrinsics[1, 2]) / camera_intrinsics[1, 1] * z
    
    # Create point cloud in camera coordinates
    points_camera = np.vstack([x, y, z, np.ones_like(z)])
    
    # Transform to world coordinates
    points_world = camera_extrinsics @ points_camera
    
    # Return points in the expected format (3, H*W)
    point_cloud = np.zeros((3, height*width))
    valid_indices = np.arange(len(u))
    point_cloud[:, valid_indices] = points_world[:3, :]
    
    # Reshape to match expected format (3, H, W)
    pcd_reshaped = point_cloud.reshape(3, height, width)
    
    return pcd_reshaped

def create_camera_intrinsics(fovy, width, height):
    """Calculate camera intrinsics matrix from field of view"""
    # Convert fovy from degrees to radians
    fovy_rad = np.deg2rad(fovy)
    
    # Calculate focal length
    f = height / (2 * np.tan(fovy_rad / 2))
    
    # Create intrinsics matrix
    K = np.array([
        [f, 0, width/2],
        [0, f, height/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K

def get_camera_extrinsics(env, camera_name):
    """Get camera extrinsics matrix from MuJoCo environment"""
    # Get camera position and orientation from MuJoCo
    p_cam, R_cam = env.get_pR_body(body_name=camera_name)
    
    # Create extrinsics matrix (4x4 transformation matrix)
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = R_cam
    extrinsics[:3, 3] = p_cam
    
    # Camera axis correction (as needed for MuJoCo)
    camera_axis_correction = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    return extrinsics @ camera_axis_correction

def load_peract_model(model_path):
    """Load the saved PerAct model components"""
    print(f"Loading PerAct model from {model_path}")
    
    try:
        # Load the saved components
        saved_data = torch.load(model_path, map_location=DEVICE)
        
        # Extract configuration
        voxel_size = saved_data['voxel_size']
        voxel_feature_size = saved_data['voxel_feature_size']
        num_rotation_classes = saved_data['num_rotation_classes']
        rotation_resolution = saved_data['rotation_resolution']
        coordinate_bounds = saved_data['coordinate_bounds']
        
        # Create the perceiver encoder
        perceiver_encoder = PerceiverIO(
            depth=6,
            iterations=1,
            voxel_size=voxel_size,
            initial_dim=10,
            low_dim_size=LOW_DIM_SIZE,
            num_latents=NUM_LATENTS,
            latent_dim=512,
            num_rotation_classes=num_rotation_classes,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            activation='relu',
        )
        
        # Create the agent
        agent = PerceiverActorAgent(
            coordinate_bounds=SCENE_BOUNDS,
            perceiver_encoder=perceiver_encoder,
            camera_names=CAMERAS,
            batch_size=1,
            voxel_size=voxel_size,
            voxel_feature_size=voxel_feature_size,
            num_rotation_classes=num_rotation_classes,
            rotation_resolution=rotation_resolution,
        )
        
        # Build the agent
        agent.build(training=False, device=DEVICE)
        
        # Load the state dictionaries
        agent._perceiver_encoder.load_state_dict(saved_data['perceiver_encoder'])
        agent._q.load_state_dict(saved_data['q'])
        
        print("PerAct model loaded successfully!")
        return agent
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_live_observations(env, rgb_images, depth_images, robot_pos, lang_goal):
    """Prepare observations from live MuJoCo simulation for PerAct model"""
    # Create batch dictionary
    batch = {}
    batch_size = 1
    timesteps = 1
    
    # Initialize low_dim_state with robot state
    # Format: [left_finger_joint, right_finger_joint, gripper_open, timestep]
    low_dim_state = np.zeros((batch_size, timesteps, LOW_DIM_SIZE), dtype=np.float32)
    low_dim_state[0, 0, 0] = robot_pos[5]  # Left finger joint
    low_dim_state[0, 0, 1] = robot_pos[6]  # Right finger joint
    low_dim_state[0, 0, 2] = 1.0 if abs(robot_pos[5] - robot_pos[6]) > 0.01 else 0.0  # Gripper open state
    low_dim_state[0, 0, 3] = 0.0  # Timestep
    
    batch['low_dim_state'] = torch.from_numpy(low_dim_state).float()
    
    # Dictionary for camera parameters
    camera_parameters = {}
    
    # Camera body names mapping - for getting extrinsics
    camera_bodies = {
        'wrist': 'camera_center',
        'front': 'front_camera',
        'left_shoulder': 'left_shoulder_camera',
        'right_shoulder': 'right_shoulder_camera'
    }
    
    # Camera FOVs - for getting intrinsics
    camera_fovs = {
        'wrist': 80,
        'front': 60,
        'left_shoulder': 80,
        'right_shoulder': 60
    }
    
    # Process each camera
    for i, camera in enumerate(CAMERAS):
        # Process RGB image
        rgb_array = process_rgb_image(rgb_images[i])
        rgb_tensor = torch.from_numpy(rgb_array).float().unsqueeze(0).unsqueeze(0)
        batch[f'{camera}_rgb'] = rgb_tensor
        
        # Process depth image with proper metric depth values
        depth_array = process_depth_image(depth_images[i], near=0.01, far=10.0)
        depth_tensor = torch.from_numpy(depth_array).float().unsqueeze(0).unsqueeze(0)
        batch[f'{camera}_depth'] = depth_tensor
        
        # Get camera intrinsics based on FOV
        fovy = camera_fovs[camera]
        intrinsics = create_camera_intrinsics(fovy, IMAGE_SIZE, IMAGE_SIZE)
        intrinsics_tensor = torch.from_numpy(intrinsics).float().unsqueeze(0).unsqueeze(0)
        batch[f'{camera}_camera_intrinsics'] = intrinsics_tensor
        
        # Get camera extrinsics from the environment
        camera_body = camera_bodies[camera]
        extrinsics = get_camera_extrinsics(env, camera_body)
        extrinsics_tensor = torch.from_numpy(extrinsics).float().unsqueeze(0).unsqueeze(0)
        batch[f'{camera}_camera_extrinsics'] = extrinsics_tensor
        
        # Generate point cloud from depth and camera parameters
        point_cloud = depth_to_point_cloud(depth_array, intrinsics, extrinsics)
        pcd_tensor = torch.from_numpy(point_cloud).float().unsqueeze(0).unsqueeze(0)
        batch[f'{camera}_point_cloud'] = pcd_tensor
        
        # Store params for debugging
        camera_parameters[camera] = {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics
        }
    
    # Add language embedding placeholder (this would be calculated in the model)
    # In real scenario, this would use CLIP to encode the language
    batch['lang_goal_embs'] = torch.ones((batch_size, timesteps, 77, 512), dtype=torch.float32)
    batch['lang_goal'] = np.array([[lang_goal]], dtype=object)
    
    # Add scene bounds
    batch['scene_bounds'] = torch.tensor(SCENE_BOUNDS, dtype=torch.float32).unsqueeze(0)
    
    # Add placeholders for action indices (these are only needed for training)
    batch['trans_action_indicies'] = torch.zeros((batch_size, timesteps, 3), dtype=torch.int32)
    batch['rot_grip_action_indicies'] = torch.zeros((batch_size, timesteps, 4), dtype=torch.int32)
    batch['ignore_collisions'] = torch.zeros((batch_size, timesteps, 1), dtype=torch.int32)
    batch['gripper_pose'] = torch.zeros((batch_size, timesteps, 7), dtype=torch.float32)
    batch['demo'] = torch.zeros((batch_size, timesteps), dtype=torch.bool)
    
    # Move all tensors to device
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Debug info about point clouds
    print("\nPoint Cloud Statistics:")
    for camera in CAMERAS:
        pcd_vals = batch[f'{camera}_point_cloud']
        non_zero = torch.count_nonzero(pcd_vals).item()
        total = pcd_vals.numel()
        print(f"  {camera}: Shape {pcd_vals.shape}, Non-zero values: {non_zero}/{total} ({non_zero/total*100:.2f}%)")
        print(f"  {camera}: Min {pcd_vals.min().item():.4f}, Max {pcd_vals.max().item():.4f}")
    
    return batch, camera_parameters

def run_live_peract(model_path):
    """Run live PerAct model inference with MuJoCo environment"""
    # Load the model
    peract_agent = load_peract_model(model_path)
    if peract_agent is None:
        return
    
    # Initialize MuJoCo environment
    env, obj_names, q_init_upright, platform_xyz = init_env()
    print("MuJoCo environment initialized")
    
    # Set language goal
    lang_goal = "Put the green block on the red square"  # Default goal
    
    try:
        iteration = 0
        while True:
            # Get user input for language goal (optional)
            # if iteration == 0 or input("Update language goal? (y/n): ").lower() == 'y':
            #     new_goal = input("Enter new language goal (or press Enter to keep current): ")
            #     if new_goal:
            #         lang_goal = new_goal
            #     print(f"Current language goal: '{lang_goal}'")
            
            # Get camera positions and orientations
            cameras_pr = {
                'wrist': env.get_pR_body(body_name='camera_center'),
                'front': env.get_pR_body(body_name='front_camera'),
                'left_shoulder': env.get_pR_body(body_name='left_shoulder_camera'),
                'right_shoulder': env.get_pR_body(body_name='right_shoulder_camera')
            }
            
            # Calculate targets
            camera_targets = {camera: p + R[:,2] for camera, (p, R) in cameras_pr.items()}
            
            # Capture images from all cameras
            rgb_images = []
            depth_images = []
            
            for camera in CAMERAS:
                p_ego, R_ego = cameras_pr[camera]
                p_trgt = camera_targets[camera]
                
                # Use the correct FOV for each camera
                fovy = 80 if camera == 'wrist' else 60
                
                rgb_img, depth_img, pcd, xyz_img = env.get_egocentric_rgb_depth_pcd(
                    p_ego=p_ego, p_trgt=p_trgt, rsz_rate=None, fovy=fovy, BACKUP_AND_RESTORE_VIEW=True)
                
                rgb_images.append(rgb_img)
                depth_images.append(depth_img)
            
            # Get robot position
            robot_pos = env.get_q([0, 1, 2, 3, 4, 5, 6])
            
            # Prepare batch for inference with correct camera parameters
            batch, camera_params = prepare_live_observations(env, rgb_images, depth_images, robot_pos, lang_goal)
            
            # Run inference
            print("\nRunning PerAct inference...")
            with torch.no_grad():
                update_dict = peract_agent.update(iteration, batch, backprop=False)
            
            # Extract predictions
            trans_coords = update_dict['pred_action']['trans'][0].cpu().numpy()
            continuous_trans = update_dict['pred_action']['continuous_trans'][0].cpu().numpy()
            rot_and_grip = update_dict['pred_action']['rot_and_grip'][0].cpu().numpy()
            continuous_quat = discrete_euler_to_quaternion(
                rot_and_grip[:3], 
                resolution=peract_agent._rotation_resolution
            )
            gripper_open = bool(rot_and_grip[3])
            ignore_collision = bool(update_dict['pred_action']['collision'][0][0].cpu().numpy())
            
            # Print predictions
            print("\nPerAct Predictions:")
            print(f"Language goal: '{lang_goal}'")
            print(f"Translation (discrete): {trans_coords}")
            print(f"Translation (continuous): {continuous_trans}")
            print(f"Rotation (discrete euler angles): {rot_and_grip[:3]}")
            print(f"Rotation (continuous quaternion): {continuous_quat}")
            print(f"Gripper open: {gripper_open}")
            print(f"Ignore collision: {ignore_collision}")
            
            # Update iteration counter
            iteration += 1
            
            # Execute the predicted action
            execute_peract_action(
                env, 
                continuous_trans, 
                continuous_quat, 
                gripper_open
            )
            env.render()
            
            # # Ask to continue
            # if input("Continue (y/n)? ").lower() != 'y':
            #     break
    
    except KeyboardInterrupt:
        print("\nInference interrupted. Closing environment...")
    
    finally:
        env.close_viewer()
        print("Environment closed")


if __name__ == "__main__":
    # Path to saved model
    model_path = '/home/jetson3/luai/peract/peract_model_components.pth'
    
    # Run live inference
    run_live_peract(model_path)