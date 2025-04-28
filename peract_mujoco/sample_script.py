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
CAMERAS = ['wrist', 'front', 'left_shoulder', 'right_shoulder']
IMAGE_SIZE = 128  # Target image size for all cameras
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

def process_depth_image(depth_img):
    """Process depth image to match demo quality/size"""
    # Convert PIL to numpy if needed
    if isinstance(depth_img, Image.Image):
        depth_img = np.array(depth_img)
    
    # Extract single channel depth
    if len(depth_img.shape) == 3:
        depth_img_2d = depth_img[:, :, 0]
    else:
        depth_img_2d = depth_img
    
    # Resize to target size
    if depth_img_2d.shape[0] != IMAGE_SIZE or depth_img_2d.shape[1] != IMAGE_SIZE:
        depth_pil = Image.fromarray(depth_img_2d)
        depth_pil = depth_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
        depth_img_2d = np.array(depth_pil)
    
    # Normalize depth values to 0-1 range
    if np.max(depth_img_2d) > 0:
        normalized_depth = depth_img_2d / np.max(depth_img_2d)
    else:
        normalized_depth = np.zeros_like(depth_img_2d, dtype=np.float32)
    
    # Reshape to (1, H, W) as needed by RLBench
    depth_array = normalized_depth.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
    
    return depth_array

def create_empty_mask(shape=(IMAGE_SIZE, IMAGE_SIZE)):
    """Create an empty mask image with the right format"""
    return np.zeros((1, shape[0], shape[1]), dtype=np.int32)

def create_empty_pcd(shape=(IMAGE_SIZE, IMAGE_SIZE)):
    """Create an empty point cloud with the right format"""
    return np.zeros((3, shape[0], shape[1]), dtype=np.float32)

def save_image_thread(running_event):
    """Thread function to save images from queue"""
    while running_event.is_set() or not image_queue.empty():
        if image_queue.empty():
            time.sleep(0.1)
            continue
        
        run_number, rgb_images, depth_images, joint_angles, pose, vels, gripper_open, tick = image_queue.get(timeout=0.1)
        
        # Initialize run data if not exists
        if run_number not in demo_data:
            demo_data[run_number] = {}
        
        # Initialize timestep data
        demo_data[run_number][tick] = {
            'gripper_pose': list(pose),
            'gripper_open': gripper_open,
            'joint_velocities': vels,

            #?
            'joint_positions': list(joint_angles)
        }
        
        # Define episode directory for this run
        episode_dir = f"{DATA_PATH}/episode{run_number}"
        
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
            
            # Process and save depth image
            depth_array = process_depth_image(depth_images[camera_idx])
            
            # Save as NPY file (for accurate data)
            depth_npy_path = os.path.join(episode_dir, f"{camera}_depth/{tick}.npy")
            np.save(depth_npy_path, depth_array)
            demo_data[run_number][tick][f'{camera}_depth'] = depth_npy_path
            
            # Also save as PNG for visualization
            depth_png_path = os.path.join(episode_dir, f"{camera}_depth/{tick}.png")
            depth_png = Image.fromarray((depth_array[0] * 255).astype(np.uint8))
            depth_png.save(depth_png_path)
            
            # Create and save empty mask
            mask_array = create_empty_mask()
            mask_path = os.path.join(episode_dir, f"{camera}_mask/{tick}.png")
            mask_png = Image.fromarray(mask_array[0].astype(np.uint8))
            mask_png.save(mask_path)
            demo_data[run_number][tick][f'{camera}_mask'] = mask_path
            
            # No need to save PCD, but record for later use in observation
            demo_data[run_number][tick][f'{camera}_pcd'] = None
        
        print(f'Saved images for timestep {tick} to episode{run_number} ({IMAGE_SIZE}x{IMAGE_SIZE} resolution)')

def save_demonstration_data(run_number):
    """Save demonstration data in RLBench format"""
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
            
            # Process depth - ensure format (1, H, W)
            if f'{camera}_depth' in data:
                if data[f'{camera}_depth'].endswith('.npy'):
                    depth_array = np.load(data[f'{camera}_depth'])
                else:
                    depth_img = Image.open(data[f'{camera}_depth'])
                    depth_array = np.array(depth_img, dtype=np.float32) / 255.0
                    depth_array = depth_array[np.newaxis, :, :]
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
        obs_params['ignore_collisions'] = np.zeros(1, dtype=np.bool_)  # Creates a 1D array with shape (1,)
        
        # Camera parameters (misc)
        obs_params['misc'] = {}
        
        # Add camera parameters for each camera
        for camera in CAMERAS + ['overhead']:
            obs_params['misc'][f'{camera}_camera_extrinsics'] = np.identity(4, dtype=np.float32)
            obs_params['misc'][f'{camera}_camera_intrinsics'] = np.array([
                [IMAGE_SIZE, 0.0, IMAGE_SIZE/2],
                [0.0, IMAGE_SIZE, IMAGE_SIZE/2],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            obs_params['misc'][f'{camera}_camera_near'] = 0.01
            obs_params['misc'][f'{camera}_camera_far'] = 10.0
        



        # (UNUSED, but not deleted)
        # Masks are deleted
        obs_params['overhead_rgb'] = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        obs_params['overhead_depth'] = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        obs_params['overhead_mask'] = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
        obs_params['overhead_point_cloud'] = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        obs_params['joint_positions'] = np.array(data.get('joint_positions', np.zeros(6)), dtype=np.float32)
        obs_params['joint_forces'] = np.zeros(6, dtype=np.float32)
        obs_params['gripper_matrix'] = np.identity(4, dtype=np.float32)
        obs_params['gripper_joint_positions'] = np.zeros(2, dtype=np.float32)
        obs_params['gripper_touch_forces'] = np.zeros(2, dtype=np.float32)

        # Create observation
        obs = Observation(**obs_params)

        obs.ignore_collisions = np.bool_(False)
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
    

    variation_descriptions = [
        "go to the block",
        "grip the block and pick it up",
        "move the block to the red platform",
        "drop the block on the red platform"
    ]

    # Save to the expected location
    with open(os.path.join(episode_dir, 'variation_descriptions.pkl'), 'wb') as f:
        pickle.dump(variation_descriptions, f)
        
    print(f"Saved demonstration {run_number} in RLBench format with {len(observations)} timesteps")



running_event = threading.Event()
running_event.set()
run_number = 1
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
                # Capture RGB and depth for each camera
                rgb_images = []
                depth_images = []
                
                for camera in CAMERAS:
                    p_ego, R_ego = cameras_pr[camera]
                    p_trgt = camera_targets[camera]
                    
                    rgb_img, depth_img, pcd, xyz_img = env.get_egocentric_rgb_depth_pcd(
                        p_ego=p_ego, p_trgt=p_trgt, rsz_rate=None, fovy=45, BACKUP_AND_RESTORE_VIEW=True)
                    
                    rgb_images.append(rgb_img)
                    depth_images.append(depth_img)
                
                # Get robot position
                joint_angles = env.get_q([0, 1, 2, 3, 4, 5])
                gripper_open = env.get_q([6])[0] + 0.7 < 0
                p,R = env.get_pR_body(body_name='tcp_link')
                q = r2quat(R)
                pose = np.concatenate([p, q])
                vels = env.get_qvel_joints(['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'])
                # Add to queue for processing
                image_queue.put((run_number, rgb_images, depth_images, joint_angles, pose, vels, gripper_open, t))
                t += 1
            
            # Step simulation
            q = q_traj_combined[tick, :]
            env.step(ctrl=q, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
            env.render()
            tick += 1
        
        # Process collected data
        save_demonstration_data(run_number)
        
        run_number += 1
        env.close_viewer()
        if run_number == 5:
            break

except KeyboardInterrupt:
    print("\nMain thread interrupted. Stopping worker thread...")
    running_event.clear()
    image_thread.join()

finally:
    # Save any remaining demonstrations
    for run_number in demo_data:
        save_demonstration_data(run_number)
