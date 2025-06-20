import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/home/jetson3/luai/peract/peract_colab')
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import queue
from PIL import Image

import json
import pickle
from mujoco_parser import MuJoCoParserClass, init_env
from util import generate_trajectories, r2quat

# Define camera parameters at the top of your script
CAMERAS = ['wrist', 'front', 'left_shoulder', 'right_shoulder']
IMAGE_SIZE = 128

# Camera body names mapping
CAMERA_BODIES = {
    'wrist': 'camera_center',
    'front': 'front_camera',
    'left_shoulder': 'left_shoulder_camera',
    'right_shoulder': 'right_shoulder_camera'
}

# Near and far planes from MuJoCo model
NEAR_PLANE = 0.01  # From XML
FAR_PLANE = 10.0   # From XML
NUM_DEMOS = 2
DATA_PATH = "/home/jetson3/luai/peract/peract_colab/data/colab_dataset/open_drawer/all_variations/episodes"

demo_data = {}

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_pointcloud_data(pcd, filepath):
    """Save point cloud data in efficient format"""
    np.save(filepath, pcd.astype(np.float32))

def load_pointcloud_data(filepath):
    """Load point cloud data"""
    return np.load(filepath)

def process_rgb_image(rgb_img):
    """Process RGB image to match demo quality/size"""
    # Convert numpy array to PIL if needed
    if not isinstance(rgb_img, Image.Image):
        rgb_img = Image.fromarray(rgb_img)
    
    # Resize to target size
    rgb_img = rgb_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    return rgb_img

def process_depth_image(depth_img, near=0.01, far=10.0):
    if isinstance(depth_img, Image.Image):
        depth_img = np.array(depth_img)
    
    # Extract single channel depth
    if len(depth_img.shape) == 3:
        depth_img_2d = depth_img[:, :, 0]
    else:
        depth_img_2d = depth_img
    
    depth_img_2d = np.clip(depth_img_2d, near, far)
    
    # Resize to target size if needed
    if depth_img_2d.shape[0] != IMAGE_SIZE or depth_img_2d.shape[1] != IMAGE_SIZE:
        depth_pil = Image.fromarray(depth_img_2d)
        depth_pil = depth_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        depth_img_2d = np.array(depth_pil)
    
    # Reshape to RLBench format (1, H, W)
    depth_array = depth_img_2d.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
    
    return depth_array

def image_to_float_array(image, scale_factor=None):
    if scale_factor is None:
        scale_factor = 2**24 - 1  # Default used by RLBench
        
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
        
    # Extract RGB channels
    if len(img.shape) == 3 and img.shape[2] == 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        
        # Combine channels in the same way they'll be decoded
        # This converts the 3-channel RGB image back to a normalized float value
        scaled_array = (r.astype(np.int32) << 16 | g.astype(np.int32) << 8 | b.astype(np.int32))
        float_array = scaled_array.astype(np.float32) / scale_factor
    else:
        # If not RGB, assume it's already a depth image and just normalize
        float_array = img.astype(np.float32) / 255.0
        
    return float_array

def float_array_to_rgb(float_array, scale_factor=None):
    if scale_factor is None:
        scale_factor = 2**24 - 1  # Default used by RLBench
        
    # Scale values to the full range
    scaled_values = (float_array * scale_factor).astype(np.uint32)
    
    # Split into RGB channels (R = MSB, B = LSB)
    r = (scaled_values >> 16) & 255
    g = (scaled_values >> 8) & 255
    b = scaled_values & 255
    
    # Combine channels into an RGB image
    rgb_image = np.stack([r, g, b], axis=2).astype(np.uint8)
    
    return rgb_image

def get_camera_extrinsics(env, camera_body_name):
    """Get camera extrinsics in the format expected by RLBench"""
    p_cam, R_cam = env.get_pR_body(camera_body_name)
    
    # Create transformation matrix (camera pose in world frame)
    T_world_cam = np.eye(4, dtype=np.float32)
    T_world_cam[:3, :3] = R_cam
    T_world_cam[:3, 3] = p_cam
    
    # RLBench expects extrinsics as world-to-camera transform
    T_cam_world = np.linalg.inv(T_world_cam)
    
    return T_cam_world

def save_image_thread(running_event):
    """Thread function to save images AND point clouds from queue"""
    while running_event.is_set() or not image_queue.empty():
        if image_queue.empty():
            time.sleep(0.1)
            continue
        
        # Now expects pcds in the queue
        run_number, rgb_images, depth_images, pcds, joint_angles, pose, vels, gripper_open, tick, camera_params = image_queue.get(timeout=0.1)
        
        # Initialize run data if not exists
        if run_number not in demo_data:
            demo_data[run_number] = {}
        
        # Initialize timestep data
        demo_data[run_number][tick] = {
            'gripper_pose': list(pose),
            'gripper_open': gripper_open,
            'joint_velocities': vels,
            'joint_positions': list(joint_angles),
            'camera_params': camera_params  # Store camera parameters for later use
        }
        
        # Define episode directory for this run
        episode_dir = f"{DATA_PATH}/episode{run_number}"
        
        # Define depth scale factor (same as RLBench)
        DEPTH_SCALE = 2**24 - 1
        
        # Process and save for each camera
        for camera_idx, camera in enumerate(CAMERAS):
            # Create directories if they don't exist - Added pointcloud directory
            for data_type in ['rgb', 'depth', 'mask', 'pointcloud']:
                ensure_dir(os.path.join(episode_dir, f"{camera}_{data_type}"))
            
            # Process and save RGB image
            rgb_img = process_rgb_image(rgb_images[camera_idx])
            rgb_path = os.path.join(episode_dir, f"{camera}_rgb/{tick}.png")
            rgb_img.save(rgb_path, quality=85)
            demo_data[run_number][tick][f'{camera}_rgb'] = rgb_path
                        
            # Get original depth size for proper intrinsics
            original_depth = depth_images[camera_idx]
            if len(original_depth.shape) == 3:
                orig_h, orig_w = original_depth.shape[:2]
            else:
                orig_h, orig_w = original_depth.shape

            # Process depth image to get properly formatted metric depth
            depth_array = process_depth_image(original_depth, NEAR_PLANE, FAR_PLANE)

            # Store the metric depth for later use in observations
            demo_data[run_number][tick][f'{camera}_depth_metric'] = depth_array

            # Calculate CORRECTED intrinsics for the RESIZED depth image
            fovy = 45
            resized_intrinsics = calculate_intrinsics(fovy, IMAGE_SIZE, IMAGE_SIZE)
            demo_data[run_number][tick][f'{camera}_intrinsics_resized'] = resized_intrinsics
            
            # Now convert the metric depth back to normalized depth for storage
            normalized_depth = (depth_array[0] - NEAR_PLANE) / (FAR_PLANE - NEAR_PLANE)
            
            # Convert normalized depth to RGB using the 24-bit encoding
            depth_rgb = float_array_to_rgb(normalized_depth, DEPTH_SCALE)
            
            # Save the depth as PNG
            depth_png_path = os.path.join(episode_dir, f"{camera}_depth/{tick}.png")
            depth_png = Image.fromarray(depth_rgb)
            depth_png.save(depth_png_path)
            
            # Store the path to the depth PNG
            demo_data[run_number][tick][f'{camera}_depth'] = depth_png_path
            
            # NEW: Save the correct point cloud directly
            pcd_path = os.path.join(episode_dir, f"{camera}_pointcloud/{tick}.npy")
            save_pointcloud_data(pcds[camera_idx], pcd_path)
            demo_data[run_number][tick][f'{camera}_pointcloud'] = pcd_path
            
            # Create and save empty mask
            mask_array = create_empty_mask()
            mask_path = os.path.join(episode_dir, f"{camera}_mask/{tick}.png")
            mask_png = Image.fromarray(mask_array[0].astype(np.uint8))
            mask_png.save(mask_path)
            demo_data[run_number][tick][f'{camera}_mask'] = mask_path
        
        print(f'Saved images and point clouds for timestep {tick} to episode{run_number} ({IMAGE_SIZE}x{IMAGE_SIZE} resolution)')

# FUNCTION 1: ADD this function to properly reshape your MuJoCo point clouds
def reshape_mujoco_pcd_for_peract(pcd_mujoco, target_shape=(128, 128)):
    """
    Reshape MuJoCo point cloud (N, 3) to PerAct format (3, H, W)
    Uses your actual MuJoCo point clouds, just reshapes them properly
    """
    h, w = target_shape
    total_pixels = h * w
    
    if len(pcd_mujoco) >= total_pixels:
        # Use first H*W points from your MuJoCo point cloud
        pcd_reshaped = pcd_mujoco[:total_pixels].reshape(h, w, 3)
    else:
        # If not enough points, pad with the last point or zeros
        pcd_full = np.zeros((total_pixels, 3), dtype=np.float32)
        pcd_full[:len(pcd_mujoco)] = pcd_mujoco
        # Fill remaining with the last valid point if available
        if len(pcd_mujoco) > 0:
            pcd_full[len(pcd_mujoco):] = pcd_mujoco[-1]
        pcd_reshaped = pcd_full.reshape(h, w, 3)
    
    # Convert to PerAct format (3, H, W)
    pcd_array = np.transpose(pcd_reshaped, (2, 0, 1)).astype(np.float32)
    
    return pcd_array

# FUNCTION 2: REPLACE your save_demonstration_data function with this updated version:
def save_demonstration_data(run_number, env):
    """Save demonstration data using your MuJoCo point clouds in PerAct format"""
    # Create episode directory
    episode_dir = f"{DATA_PATH}/episode{run_number}"
    ensure_dir(episode_dir)
    
    # Get timesteps
    timesteps = sorted(demo_data[run_number].keys())
    
    # Import necessary RLBench classes
    from rlbench.backend.observation import Observation
    from rlbench.demo import Demo
    
    # Create observation list
    observations = []
    
    # Define depth scale factor (same as RLBench)
    DEPTH_SCALE = 2**24 - 1
    
    # For each timestep, create an Observation
    for t in timesteps:
        # Extract data for this timestep
        data = demo_data[run_number][t]
        
        # Create observation parameters
        obs_params = {}
        
        # Add camera images (RGB, depth, mask, point cloud)
        for camera in CAMERAS:
            # Process RGB - ensure format (3, H, W)
            if f'{camera}_rgb' in data:
                rgb_img = Image.open(data[f'{camera}_rgb'])
                rgb_array = np.array(rgb_img)
                # Convert to CHW format
                if len(rgb_array.shape) == 3 and rgb_array.shape[2] >= 3:
                    rgb_array = np.transpose(rgb_array[:, :, :3], (2, 0, 1))
                else:
                    # Handle grayscale
                    if len(rgb_array.shape) == 2:
                        rgb_array = np.repeat(rgb_array[:, :, np.newaxis], 3, axis=2)
                    rgb_array = np.transpose(rgb_array, (2, 0, 1))
            else:
                rgb_array = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            
            # Use the pre-calculated metric depth if available
            if f'{camera}_depth_metric' in data:
                depth_array = data[f'{camera}_depth_metric']
                # FIX: Convert from (1, H, W) to (H, W) for RLBench
                if len(depth_array.shape) == 3 and depth_array.shape[0] == 1:
                    depth_array = depth_array[0]  # Remove the first dimension
            # Otherwise, load and convert the depth PNG
            elif f'{camera}_depth' in data:
                depth_img = Image.open(data[f'{camera}_depth'])
                # Convert the PNG back to normalized depth
                normalized_depth = image_to_float_array(depth_img, DEPTH_SCALE)
                # Convert normalized to metric
                metric_depth = NEAR_PLANE + normalized_depth * (FAR_PLANE - NEAR_PLANE)
                depth_array = metric_depth  # Keep as (H, W) for RLBench
            else:
                depth_array = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)  # (H, W) format
            
            # Create mask array
            mask_array = create_empty_mask()
            
            # FIXED: Use your MuJoCo point clouds, just reshape them properly
            if f'{camera}_pointcloud' in data:
                pcd_file_path = data[f'{camera}_pointcloud']
                pcd_mujoco = load_pointcloud_data(pcd_file_path)  # Your original MuJoCo PCD
                
                # Reshape your MuJoCo point cloud to PerAct format (3, H, W)
                pcd_array = reshape_mujoco_pcd_for_peract(pcd_mujoco, target_shape=(IMAGE_SIZE, IMAGE_SIZE))
            else:
                pcd_array = create_empty_pcd()  # Fallback to empty
            
            # Add to observation parameters
            obs_params[f'{camera}_rgb'] = rgb_array
            obs_params[f'{camera}_depth'] = depth_array
            obs_params[f'{camera}_mask'] = mask_array
            obs_params[f'{camera}_point_cloud'] = pcd_array  # Use your MuJoCo point cloud
        
        # Add robot state
        obs_params['joint_velocities'] = np.array(data.get('joint_velocities', np.zeros(6)), dtype=np.float32)
        obs_params['gripper_open'] = bool(data.get('gripper_open', True))
        obs_params['gripper_pose'] = np.array(data.get('gripper_pose', np.zeros(7)), dtype=np.float32)
        
        # Task state
        obs_params['task_low_dim_state'] = np.zeros(1, dtype=np.float32)
        obs_params['ignore_collisions'] = np.zeros(1, dtype=np.bool_)
        
        # Camera parameters (misc)
        obs_params['misc'] = {}
        
        # Add camera parameters for each camera
        for camera in CAMERAS:
            # Use resized intrinsics if available, otherwise use original
            if f'{camera}_intrinsics_resized' in data:
                intrinsics = data[f'{camera}_intrinsics_resized']
            else:
                intrinsics = data['camera_params']['intrinsics'][camera]
            
            extrinsics = data['camera_params']['extrinsics'][camera]
            
            # Add to observation parameters
            obs_params['misc'][f'{camera}_camera_intrinsics'] = intrinsics
            obs_params['misc'][f'{camera}_camera_extrinsics'] = extrinsics
            obs_params['misc'][f'{camera}_camera_near'] = NEAR_PLANE
            obs_params['misc'][f'{camera}_camera_far'] = FAR_PLANE
        
        # Add additional required parameters
        obs_params['misc']['overhead_camera_intrinsics'] = calculate_intrinsics(60, IMAGE_SIZE, IMAGE_SIZE)
        obs_params['misc']['overhead_camera_extrinsics'] = np.identity(4, dtype=np.float32)
        obs_params['misc']['overhead_camera_near'] = NEAR_PLANE
        obs_params['misc']['overhead_camera_far'] = FAR_PLANE
        
        # Required overhead camera placeholders
        obs_params['overhead_rgb'] = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        obs_params['overhead_depth'] = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        obs_params['overhead_mask'] = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
        obs_params['overhead_point_cloud'] = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        # Additional robot state parameters
        obs_params['joint_positions'] = np.array(data.get('joint_positions', np.zeros(6)), dtype=np.float32)
        obs_params['joint_forces'] = np.zeros(6, dtype=np.float32)
        obs_params['gripper_matrix'] = np.identity(4, dtype=np.float32)
        obs_params['gripper_joint_positions'] = np.zeros(2, dtype=np.float32)
        obs_params['gripper_touch_forces'] = np.zeros(2, dtype=np.float32)

        # Create observation
        obs = Observation(**obs_params)

        # Set ignore_collisions (must be a bool_ type)
        obs.ignore_collisions = np.bool_(False)
        
        # Clean up unused attributes
        objects_to_delete = [
            'front_mask', 
            'left_shoulder_mask',
            'right_shoulder_mask',
            'wrist_mask',
            'overhead_depth', 
            'overhead_mask', 
            'overhead_point_cloud', 
            'overhead_rgb',
        ]
        
        # Delete each object if it exists in the observation
        for obj_name in objects_to_delete:
            if hasattr(obs, obj_name):
                delattr(obs, obj_name)
                
        observations.append(obs)

    # Create Demo object
    demo = Demo(observations=observations)
    demo.variation_number = 0
    
    # Save Demo object
    with open(os.path.join(episode_dir, 'low_dim_obs.pkl'), 'wb') as f:
        pickle.dump(demo, f)
    
    # Save variation number
    with open(os.path.join(episode_dir, 'variation_number.pkl'), 'wb') as f:
        pickle.dump(0, f)
    
    # Sample variation descriptions for open_drawer task
    variation_descriptions = [
        "Put the green block on the red square"
    ]

    # Save variation descriptions
    with open(os.path.join(episode_dir, 'variation_descriptions.pkl'), 'wb') as f:
        pickle.dump(variation_descriptions, f)
        
    print(f"Saved demonstration {run_number} using your MuJoCo point clouds in PerAct format ({len(observations)} timesteps)")

def create_empty_mask(shape=(IMAGE_SIZE, IMAGE_SIZE)):
    """Create an empty mask image with the right format"""
    return np.zeros((1, shape[0], shape[1]), dtype=np.int32)

def create_empty_pcd(shape=(IMAGE_SIZE, IMAGE_SIZE)):
    """Create an empty point cloud with the right format"""
    return np.zeros((3, shape[0], shape[1]), dtype=np.float32)

def calculate_intrinsics(fovy, width, height):
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

# Main execution
running_event = threading.Event()
running_event.set()
run_number = 0
image_queue = queue.Queue()

# Start image processing thread
image_thread = threading.Thread(target=save_image_thread, args=[running_event], daemon=True)
image_thread.start()

try:
    while True:
        env, obj_names, q_init_upright, platform_xyz = init_env()
        tick = 0
        t = 0
        q_traj_combined = generate_trajectories(env, obj_names, q_init_upright, platform_xyz)
        
        while tick < q_traj_combined.shape[0]:
            # Get camera positions and orientations
            cameras_pr = {
                'wrist': env.get_pR_body(body_name='camera_center'),
                'front': env.get_pR_body(body_name='front_camera'),
                'left_shoulder': env.get_pR_body(body_name='left_shoulder_camera'),
                'right_shoulder': env.get_pR_body(body_name='right_shoulder_camera')
            }
                        
            # Target for all cameras
            if obj_names:
                common_target = env.get_p_body(obj_names[0])  # Look at the object
            else:
                common_target = np.array([1.0, 0.0, 0.8])  # Fixed point in space

            p_trgt = common_target  # SAME target for ALL cameras
            
            # Capture images every N steps
            if tick % 5 == 0:
                # Initialize camera parameter storage
                camera_params = {
                    'intrinsics': {},
                    'extrinsics': {}
                }
                
                # Capture RGB, depth, and point clouds for each camera
                rgb_images = []
                depth_images = []
                pcds = []  # NEW: Store point clouds

                import open3d as o3d

                for camera_idx, camera in enumerate(CAMERAS):
                    p_ego, R_ego = cameras_pr[camera]
                    
                    fovy = 45
                    rgb_img, depth_img, pcd, xyz_img = env.get_egocentric_rgb_depth_pcd(
                        p_ego=p_ego, p_trgt=p_trgt, rsz_rate=None, fovy=fovy, BACKUP_AND_RESTORE_VIEW=True)
                    print("MuJoCo depth shape:", depth_img.shape)
                    print("MuJoCo depth sample values:", depth_img[100:105, 100:105])
                    print("MuJoCo depth min/max:", depth_img.min(), depth_img.max())
                    rgb_images.append(rgb_img)
                    depth_images.append(depth_img)
                    pcds.append(pcd)  # NEW: Save the working point cloud
                    
                    original_height, original_width = depth_img.shape[:2] if len(depth_img.shape) == 3 else depth_img.shape
                    intrinsics = calculate_intrinsics(fovy, original_width, original_height)
                    camera_params['intrinsics'][camera] = intrinsics

                    extrinsics = get_camera_extrinsics(env, CAMERA_BODIES[camera])
                    camera_params['extrinsics'][camera] = extrinsics

                # Get robot position
                joint_angles = env.get_q([0, 1, 2, 3, 4, 5])
                gripper_open = env.get_q([6])[0] + 0.7 < 0
                p, R = env.get_pR_body(body_name='tcp_link')
                q = r2quat(R)
                pose = np.concatenate([p, q])
                vels = env.get_qvel_joints(['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'])
                
                # Add to queue for processing - NOW includes pcds
                image_queue.put((run_number, rgb_images, depth_images, pcds, joint_angles, pose, vels, gripper_open, t, camera_params))
                t += 1
            
            # Step simulation
            q = q_traj_combined[tick, :]
            env.step(ctrl=q, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
            env.render()
            tick += 1
        
        # Process collected data
        save_demonstration_data(run_number, env)
        
        run_number += 1
        env.close_viewer()
        if run_number >= NUM_DEMOS:
            break

except KeyboardInterrupt:
    print("\nMain thread interrupted. Stopping worker thread...")
    running_event.clear()
    image_thread.join()

finally:
    # Save any remaining demonstrations
    for run_num in demo_data:
        save_demonstration_data(run_num, None)