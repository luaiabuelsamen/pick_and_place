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

# Camera FOVs from MuJoCo model - update these values based on your XML
CAMERA_FOVS = {
    'wrist': 80,         # Assuming 80 from the advice
    'front': 60,         # Assuming 60 from the advice
    'left_shoulder': 80, # Assuming 80 from the advice
    'right_shoulder': 60 # Assuming 60 from the advice
}

# Near and far planes from MuJoCo model
NEAR_PLANE = 0.01  # From XML
FAR_PLANE = 10.0   # From XML
DATA_PATH = "/home/jetson3/luai/peract/peract_colab/data/colab_dataset/open_drawer/all_variations/episodes"

demo_data = {}

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_rgb_image(rgb_img):
    """Process RGB image to match demo quality/size"""
    # Convert numpy array to PIL if needed
    if not isinstance(rgb_img, Image.Image):
        rgb_img = Image.fromarray(rgb_img)
    
    # Resize to target size
    rgb_img = rgb_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    return rgb_img

def process_depth_image(depth_img, near=0.01, far=10.0):
    """
    Process depth image to match RLBench format
    
    Args:
        depth_img: Input depth image from MuJoCo (either as numpy array or PIL Image)
        near: Near plane distance in meters
        far: Far plane distance in meters
    
    Returns:
        depth_array: Depth array in format expected by RLBench (1, H, W)
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
    
    # Normalize MuJoCo depth to 0-1 range if it isn't already
    if np.max(depth_img_2d) > 1.0:
        depth_img_2d = depth_img_2d / np.max(depth_img_2d)
    
    # Convert from normalized 0-1 to actual metric depth
    metric_depth = near + depth_img_2d * (far - near)
    
    # Reshape to RLBench format (1, H, W)
    depth_array = metric_depth.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
    
    return depth_array

def image_to_float_array(image, scale_factor=None):
    """
    Convert a depth image to a float array, scaling by the provided factor.
    This mimics RLBench's function for encoding/decoding depth images.
    
    Args:
        image: PIL Image or numpy array
        scale_factor: Value to scale by (default is 2**24 - 1)
        
    Returns:
        Float array of values between 0 and 1
    """
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
    """
    Convert a float array to an RGB image by scaling and splitting into channels.
    This is the inverse of image_to_float_array.
    
    Args:
        float_array: Numpy array of float values between 0 and 1
        scale_factor: Value to scale by (default is 2**24 - 1)
        
    Returns:
        RGB image as a numpy array
    """
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

# Update the save_image_thread function to correctly save depth images
def save_image_thread(running_event):
    """Thread function to save images from queue"""
    while running_event.is_set() or not image_queue.empty():
        if image_queue.empty():
            time.sleep(0.1)
            continue
        
        run_number, rgb_images, depth_images, joint_angles, pose, vels, gripper_open, tick, camera_params = image_queue.get(timeout=0.1)
        
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
            # Create directories if they don't exist
            for data_type in ['rgb', 'depth', 'mask']:
                ensure_dir(os.path.join(episode_dir, f"{camera}_{data_type}"))
            
            # Process and save RGB image
            rgb_img = process_rgb_image(rgb_images[camera_idx])
            rgb_path = os.path.join(episode_dir, f"{camera}_rgb/{tick}.png")
            rgb_img.save(rgb_path, quality=85)  # Reduced quality to match demo
            demo_data[run_number][tick][f'{camera}_rgb'] = rgb_path
            
            # Process depth image to get properly formatted metric depth
            depth_array = process_depth_image(depth_images[camera_idx], NEAR_PLANE, FAR_PLANE)
            
            # Store the metric depth for later use in observations
            demo_data[run_number][tick][f'{camera}_depth_metric'] = depth_array
            
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
            
            # Create and save empty mask
            mask_array = create_empty_mask()
            mask_path = os.path.join(episode_dir, f"{camera}_mask/{tick}.png")
            mask_png = Image.fromarray(mask_array[0].astype(np.uint8))
            mask_png.save(mask_path)
            demo_data[run_number][tick][f'{camera}_mask'] = mask_path
            
            # No need to save PCD, but record for later use in observation
            demo_data[run_number][tick][f'{camera}_pcd'] = None
        
        print(f'Saved images for timestep {tick} to episode{run_number} ({IMAGE_SIZE}x{IMAGE_SIZE} resolution)')

# Update the save_demonstration_data function to correctly handle depth conversion
def save_demonstration_data(run_number, env):
    """Save demonstration data in RLBench format with correct camera parameters"""
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
            # Otherwise, load and convert the depth PNG
            elif f'{camera}_depth' in data:
                depth_img = Image.open(data[f'{camera}_depth'])
                # Convert the PNG back to normalized depth
                normalized_depth = image_to_float_array(depth_img, DEPTH_SCALE)
                # Convert normalized to metric
                metric_depth = NEAR_PLANE + normalized_depth * (FAR_PLANE - NEAR_PLANE)
                depth_array = metric_depth.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
            else:
                depth_array = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            
            # Create mask and point cloud arrays
            mask_array = create_empty_mask()
            pcd_array = create_empty_pcd()
            
            # Add to observation parameters
            obs_params[f'{camera}_rgb'] = rgb_array
            obs_params[f'{camera}_depth'] = depth_array
            obs_params[f'{camera}_mask'] = mask_array
            obs_params[f'{camera}_point_cloud'] = pcd_array
        
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
            # Get parameters from stored data if available
            if 'camera_params' in data and 'intrinsics' in data['camera_params'] and camera in data['camera_params']['intrinsics']:
                intrinsics = data['camera_params']['intrinsics'][camera]
            else:
                # Fallback to calculated intrinsics
                fovy = CAMERA_FOVS[camera]
                intrinsics = calculate_intrinsics(fovy, IMAGE_SIZE, IMAGE_SIZE)
            
            if 'camera_params' in data and 'extrinsics' in data['camera_params'] and camera in data['camera_params']['extrinsics']:
                extrinsics = data['camera_params']['extrinsics'][camera]
            else:
                # Fallback to identity if not available
                extrinsics = np.identity(4, dtype=np.float32)
            
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
        
    print(f"Saved demonstration {run_number} in RLBench format with {len(observations)} timesteps")

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
            
            # Calculate targets
            camera_targets = {camera: p + R[:,2] for camera, (p, R) in cameras_pr.items()}
            
            # Capture images every N steps
            if tick % 5 == 0:
                # Initialize camera parameter storage
                camera_params = {
                    'intrinsics': {},
                    'extrinsics': {}
                }
                
                # Capture RGB and depth for each camera
                rgb_images = []
                depth_images = []
                
                for camera_idx, camera in enumerate(CAMERAS):
                    p_ego, R_ego = cameras_pr[camera]
                    p_trgt = camera_targets[camera]
                    
                    # Get correct FOV for this camera
                    fovy = CAMERA_FOVS[camera]
                    
                    rgb_img, depth_img, pcd, xyz_img = env.get_egocentric_rgb_depth_pcd(
                        p_ego=p_ego, p_trgt=p_trgt, rsz_rate=None, fovy=fovy, BACKUP_AND_RESTORE_VIEW=True)
                    
                    rgb_images.append(rgb_img)
                    depth_images.append(depth_img)
                    
                    # Calculate intrinsics and store
                    intrinsics = calculate_intrinsics(fovy, IMAGE_SIZE, IMAGE_SIZE)
                    camera_params['intrinsics'][camera] = intrinsics
                    
                    # Get extrinsics and store
                    extrinsics = get_camera_extrinsics(env, CAMERA_BODIES[camera])
                    camera_params['extrinsics'][camera] = extrinsics
                
                # Get robot position
                joint_angles = env.get_q([0, 1, 2, 3, 4, 5])
                gripper_open = env.get_q([6])[0] + 0.7 < 0
                p, R = env.get_pR_body(body_name='tcp_link')
                q = r2quat(R)
                pose = np.concatenate([p, q])
                vels = env.get_qvel_joints(['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'])
                
                # Add to queue for processing with camera parameters
                image_queue.put((run_number, rgb_images, depth_images, joint_angles, pose, vels, gripper_open, t, camera_params))
                t += 1
            
            # Step simulation
            q = q_traj_combined[tick, :]
            env.step(ctrl=q, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
            env.render()
            tick += 1
        
        # Process collected data - now pass env to save correct camera parameters
        save_demonstration_data(run_number, env)
        
        run_number += 1
        env.close_viewer()
        if run_number >= 20:
            break

except KeyboardInterrupt:
    print("\nMain thread interrupted. Stopping worker thread...")
    running_event.clear()
    image_thread.join()

finally:
    # Save any remaining demonstrations - note we don't have env here, so passing None
    for run_number in demo_data:
        save_demonstration_data(run_number, None)