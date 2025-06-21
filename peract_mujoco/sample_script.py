import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('../peract/peract_colab')
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import queue
from PIL import Image
import cv2
import json
import pickle
from mujoco_parser import MuJoCoParserClass, init_env
from util import generate_trajectories, r2quat

# Define camera parameters
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
NEAR_PLANE = 0.01
FAR_PLANE = 10.0
NUM_DEMOS = 2
DATA_PATH = "../peract/peract_colab/data/colab_dataset/open_drawer/all_variations/episodes"

demo_data = {}

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def capture_synchronized_multiview_data(env, target_pos, target_size=128, fovy=45):
    """
    FIXED: Use MuJoCo's pcd (not xyz_img) and reshape to RLBench format
    """
    print(f"Capturing synchronized multi-view data at timestep {env.tick}")
    
    # Store original viewer state
    backup_azimuth, backup_distance, backup_elevation, backup_lookat = env.get_viewer_cam_info()
    
    rgb_images = []
    depth_images = []
    point_clouds = []
    camera_params = {'intrinsics': {}, 'extrinsics': {}}
    
    # Process each camera in sequence
    for camera in CAMERAS:
        camera_body = CAMERA_BODIES[camera]
        
        # Get camera pose AT THIS EXACT TIMESTEP
        p_cam, R_cam = env.get_pR_body(camera_body)
        
        # Get MuJoCo's point cloud data
        rgb_img, depth_img, pcd, xyz_img = env.get_egocentric_rgb_depth_pcd(
            p_ego=p_cam, 
            p_trgt=target_pos, 
            rsz_rate=None, 
            fovy=fovy, 
            BACKUP_AND_RESTORE_VIEW=True
        )
        
        # Process RGB and depth
        rgb_resized = cv2.resize(rgb_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        depth_processed = improved_depth_processing(depth_img, target_size, NEAR_PLANE, FAR_PLANE)
        
        # FIXED: Use MuJoCo's pcd (multi-view consistent) and reshape to RLBench format
        h, w = xyz_img.shape[:2]  # Get original image dimensions
        pcd_img = pcd.reshape(h, w, 3)  # Reshape pcd (N, 3) to image format (H, W, 3)
        pcd_resized = cv2.resize(pcd_img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
        # Calculate camera parameters
        intrinsics = calculate_mujoco_intrinsics(fovy, depth_img.shape, target_size)
        extrinsics = create_camera_extrinsics_matrix(p_cam, R_cam)
        
        # Store everything
        rgb_images.append(rgb_resized)
        depth_images.append(depth_processed)
        point_clouds.append(pcd_resized)
        camera_params['intrinsics'][camera] = intrinsics
        camera_params['extrinsics'][camera] = extrinsics
        
        print(f"  {camera}: pcd {pcd.shape} -> img {pcd_img.shape} -> resized {pcd_resized.shape}")
    
    # Restore viewer state
    env.update_viewer(azimuth=backup_azimuth, distance=backup_distance,
                     elevation=backup_elevation, lookat=backup_lookat)
    
    return rgb_images, depth_images, point_clouds, camera_params

def improved_depth_processing(depth_img, target_size=128, near=0.01, far=10.0):
    """Process depth images maintaining precision"""
    # Ensure we have a 2D depth array
    if len(depth_img.shape) == 3:
        depth_2d = depth_img[:, :, 0] if depth_img.shape[2] == 1 else depth_img[:, :, 0]
    else:
        depth_2d = depth_img.copy()
    
    # Clip to valid range
    depth_2d = np.clip(depth_2d, near, far)
    
    # Resize using nearest neighbor
    if depth_2d.shape[0] != target_size or depth_2d.shape[1] != target_size:
        depth_resized = cv2.resize(depth_2d, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    else:
        depth_resized = depth_2d
    
    return depth_resized

def calculate_mujoco_intrinsics(fovy_deg, original_shape, target_size):
    """Calculate intrinsics that match MuJoCo's camera model"""
    if len(original_shape) == 3:
        orig_height, orig_width = original_shape[:2]
    else:
        orig_height, orig_width = original_shape
    
    # MuJoCo's focal length calculation
    fovy_rad = np.deg2rad(fovy_deg)
    focal_scaling = 0.5 * orig_height / np.tan(fovy_rad / 2)
    
    # Scale to target size
    scale_x = target_size / orig_width
    scale_y = target_size / orig_height
    
    # Build intrinsics matrix
    K = np.array([
        [focal_scaling * scale_x, 0, target_size / 2],
        [0, focal_scaling * scale_y, target_size / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K

def create_camera_extrinsics_matrix(p_cam, R_cam):
    """Create camera extrinsics matrix in RLBench format"""
    T_world_cam = np.eye(4, dtype=np.float32)
    T_world_cam[:3, :3] = R_cam
    T_world_cam[:3, 3] = p_cam
    return T_world_cam

def save_pointcloud_data(pcd, filepath):
    """Save point cloud data in HWC format for extract_obs compatibility"""
    if len(pcd.shape) != 3 or pcd.shape[2] != 3:
        raise ValueError(f"Point cloud must have shape (H, W, 3), got {pcd.shape}")
    
    print(f"Saving point cloud with shape (HWC format): {pcd.shape}")
    np.save(filepath, pcd.astype(np.float32))

def load_pointcloud_data(filepath):
    """Load point cloud data"""
    pcd = np.load(filepath)
    
    if len(pcd.shape) != 3 or pcd.shape[2] != 3:
        print(f"WARNING: Loaded point cloud has incorrect shape: {pcd.shape}")
        print("Expected (H, W, 3) format for extract_obs compatibility")
    
    print(f"Loaded point cloud with shape (HWC format): {pcd.shape}")
    return pcd

def process_rgb_image(rgb_img):
    """Process RGB image"""
    if not isinstance(rgb_img, Image.Image):
        rgb_img = Image.fromarray(rgb_img)
    return rgb_img

def process_depth_image_for_storage(depth_img, near=0.01, far=10.0):
    """Process depth for storage"""
    if len(depth_img.shape) == 2:
        depth_array = depth_img.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
    else:
        depth_array = depth_img
    return depth_array

def image_to_float_array(image, scale_factor=None):
    if scale_factor is None:
        scale_factor = 2**24 - 1
        
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
        
    if len(img.shape) == 3 and img.shape[2] == 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        scaled_array = (r.astype(np.int32) << 16 | g.astype(np.int32) << 8 | b.astype(np.int32))
        float_array = scaled_array.astype(np.float32) / scale_factor
    else:
        float_array = img.astype(np.float32) / 255.0
        
    return float_array

def float_array_to_rgb(float_array, scale_factor=None):
    if scale_factor is None:
        scale_factor = 2**24 - 1
        
    scaled_values = (float_array * scale_factor).astype(np.uint32)
    
    r = (scaled_values >> 16) & 255
    g = (scaled_values >> 8) & 255
    b = scaled_values & 255
    
    rgb_image = np.stack([r, g, b], axis=2).astype(np.uint8)
    return rgb_image

def save_image_thread(running_event):
    """Thread function to save images and point clouds"""
    while running_event.is_set() or not image_queue.empty():
        if image_queue.empty():
            time.sleep(0.1)
            continue
        
        try:
            run_number, rgb_images, depth_images, rlbench_pcds, joint_angles, pose, vels, gripper_open, tick, camera_params = image_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # Initialize run data
        if run_number not in demo_data:
            demo_data[run_number] = {}
        
        demo_data[run_number][tick] = {
            'gripper_pose': list(pose),
            'gripper_open': gripper_open,
            'joint_velocities': vels,
            'joint_positions': list(joint_angles),
            'camera_params': camera_params
        }
        
        episode_dir = f"{DATA_PATH}/episode{run_number}"
        DEPTH_SCALE = 2**24 - 1
        
        # Process and save for each camera
        for camera_idx, camera in enumerate(CAMERAS):
            # Create directories
            for data_type in ['rgb', 'depth', 'mask', 'pointcloud']:
                ensure_dir(os.path.join(episode_dir, f"{camera}_{data_type}"))
            
            # Save RGB
            rgb_img = process_rgb_image(rgb_images[camera_idx])
            rgb_path = os.path.join(episode_dir, f"{camera}_rgb/{tick}.png")
            rgb_img.save(rgb_path, quality=85)
            demo_data[run_number][tick][f'{camera}_rgb'] = rgb_path
            
            # Save depth
            depth_array = process_depth_image_for_storage(depth_images[camera_idx], NEAR_PLANE, FAR_PLANE)
            demo_data[run_number][tick][f'{camera}_depth_metric'] = depth_array
            
            normalized_depth = (depth_array[0] - NEAR_PLANE) / (FAR_PLANE - NEAR_PLANE)
            depth_rgb = float_array_to_rgb(normalized_depth, DEPTH_SCALE)
            
            depth_png_path = os.path.join(episode_dir, f"{camera}_depth/{tick}.png")
            depth_png = Image.fromarray(depth_rgb)
            depth_png.save(depth_png_path)
            demo_data[run_number][tick][f'{camera}_depth'] = depth_png_path
            
            # Save FIXED point cloud
            pcd_path = os.path.join(episode_dir, f"{camera}_pointcloud/{tick}.npy")
            save_pointcloud_data(rlbench_pcds[camera_idx], pcd_path)
            demo_data[run_number][tick][f'{camera}_pointcloud'] = pcd_path
            
            # Save empty mask
            mask_array = create_empty_mask()
            mask_path = os.path.join(episode_dir, f"{camera}_mask/{tick}.png")
            mask_png = Image.fromarray(mask_array[0].astype(np.uint8))
            mask_png.save(mask_path)
            demo_data[run_number][tick][f'{camera}_mask'] = mask_path
        
        print(f'Saved FIXED multi-view consistent data for timestep {tick} to episode{run_number}')

def save_demonstration_data(run_number, env):
    """Save demonstration data using FIXED point clouds"""
    episode_dir = f"{DATA_PATH}/episode{run_number}"
    ensure_dir(episode_dir)
    
    timesteps = sorted(demo_data[run_number].keys())
    
    from rlbench.backend.observation import Observation
    from rlbench.demo import Demo
    
    observations = []
    DEPTH_SCALE = 2**24 - 1
    
    for t in timesteps:
        data = demo_data[run_number][t]
        obs_params = {}
        
        # Add camera data
        for camera in CAMERAS:
            # RGB
            if f'{camera}_rgb' in data:
                rgb_img = Image.open(data[f'{camera}_rgb'])
                rgb_array = np.array(rgb_img)
                if len(rgb_array.shape) == 3 and rgb_array.shape[2] >= 3:
                    rgb_array = np.transpose(rgb_array[:, :, :3], (2, 0, 1))
                else:
                    if len(rgb_array.shape) == 2:
                        rgb_array = np.repeat(rgb_array[:, :, np.newaxis], 3, axis=2)
                    rgb_array = np.transpose(rgb_array, (2, 0, 1))
            else:
                rgb_array = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            
            # Depth
            if f'{camera}_depth_metric' in data:
                depth_array = data[f'{camera}_depth_metric']
                if len(depth_array.shape) == 3 and depth_array.shape[0] == 1:
                    depth_array = depth_array[0]
            else:
                depth_array = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            
            # Mask
            mask_array = create_empty_mask()
            
            # FIXED Point cloud
            if f'{camera}_pointcloud' in data:
                pcd_file_path = data[f'{camera}_pointcloud']
                pcd_hwc = load_pointcloud_data(pcd_file_path)
                
                if pcd_hwc.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                    print(f"WARNING: {camera} point cloud has shape {pcd_hwc.shape}, expected ({IMAGE_SIZE}, {IMAGE_SIZE}, 3)")
                
                pcd_array = pcd_hwc
            else:
                pcd_array = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
            
            # Add to observation
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
        
        # Camera parameters
        obs_params['misc'] = {}
        
        for camera in CAMERAS:
            intrinsics = data['camera_params']['intrinsics'][camera]
            extrinsics = data['camera_params']['extrinsics'][camera]
            
            obs_params['misc'][f'{camera}_camera_intrinsics'] = intrinsics
            obs_params['misc'][f'{camera}_camera_extrinsics'] = extrinsics
            obs_params['misc'][f'{camera}_camera_near'] = NEAR_PLANE
            obs_params['misc'][f'{camera}_camera_far'] = FAR_PLANE
        
        # Required overhead camera placeholders
        obs_params['misc']['overhead_camera_intrinsics'] = calculate_mujoco_intrinsics(60, (IMAGE_SIZE, IMAGE_SIZE), IMAGE_SIZE)
        obs_params['misc']['overhead_camera_extrinsics'] = np.identity(4, dtype=np.float32)
        obs_params['misc']['overhead_camera_near'] = NEAR_PLANE
        obs_params['misc']['overhead_camera_far'] = FAR_PLANE
        
        obs_params['overhead_rgb'] = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        obs_params['overhead_depth'] = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        obs_params['overhead_mask'] = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
        obs_params['overhead_point_cloud'] = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        # Additional robot state
        obs_params['joint_positions'] = np.array(data.get('joint_positions', np.zeros(6)), dtype=np.float32)
        obs_params['joint_forces'] = np.zeros(6, dtype=np.float32)
        obs_params['gripper_matrix'] = np.identity(4, dtype=np.float32)
        obs_params['gripper_joint_positions'] = np.zeros(2, dtype=np.float32)
        obs_params['gripper_touch_forces'] = np.zeros(2, dtype=np.float32)

        # Create observation
        obs = Observation(**obs_params)
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

    # Save demo
    demo = Demo(observations=observations)
    demo.variation_number = 0
    
    with open(os.path.join(episode_dir, 'low_dim_obs.pkl'), 'wb') as f:
        pickle.dump(demo, f)
    
    with open(os.path.join(episode_dir, 'variation_number.pkl'), 'wb') as f:
        pickle.dump(0, f)
    
    variation_descriptions = ["Put the green block on the red square"]
    with open(os.path.join(episode_dir, 'variation_descriptions.pkl'), 'wb') as f:
        pickle.dump(variation_descriptions, f)
        
    print(f"Saved demonstration {run_number} with FIXED multi-view consistent point clouds ({len(observations)} timesteps)")

def create_empty_mask(shape=(IMAGE_SIZE, IMAGE_SIZE)):
    """Create empty mask"""
    return np.zeros((1, shape[0], shape[1]), dtype=np.int32)

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
            # Target for all cameras
            if obj_names:
                common_target = env.get_p_body(obj_names[0])
            else:
                common_target = np.array([1.0, 0.0, 0.8])

            # Capture images every N steps
            if tick % 5 == 0:
                # FIXED: Use synchronized capture with MuJoCo's pcd
                rgb_images, depth_images, rlbench_pcds, camera_params = capture_synchronized_multiview_data(
                    env, common_target, target_size=IMAGE_SIZE, fovy=45
                )

                # Get robot state
                joint_angles = env.get_q([0, 1, 2, 3, 4, 5])
                gripper_open = env.get_q([6])[0] + 0.7 < 0
                p, R = env.get_pR_body(body_name='tcp_link')
                q = r2quat(R)
                pose = np.concatenate([p, q])
                vels = env.get_qvel_joints(['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'])
                
                # Add to queue with FIXED point clouds
                image_queue.put((run_number, rgb_images, depth_images, rlbench_pcds, joint_angles, pose, vels, gripper_open, t, camera_params))
                t += 1
            
            # Step simulation
            q = q_traj_combined[tick, :]
            env.step(ctrl=q, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
            env.render()
            tick += 1
        
        # Save collected data
        save_demonstration_data(run_number, env)
        
        run_number += 1
        env.close_viewer()
        if run_number >= NUM_DEMOS:
            break

except KeyboardInterrupt:
    print("\nInterrupted. Stopping...")
    running_event.clear()
    image_thread.join()

finally:
    # Save any remaining demonstrations
    for run_num in demo_data:
        save_demonstration_data(run_num, None)
    
    running_event.clear()
    if image_thread.is_alive():
        image_thread.join()
    
    print("All demonstrations saved with FIXED multi-view consistent point clouds!")