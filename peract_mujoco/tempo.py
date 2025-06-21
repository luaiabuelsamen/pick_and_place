import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/Users/labuelsamen/Desktop/berkeley/peract/peract_colab')
import numpy as np
import cv2
import open3d as o3d
from mujoco_parser import MuJoCoParserClass, init_env

def visualize_multiview_before_after():
    """Show MuJoCo's original pcd vs your resized xyz_img"""
    
    print("🔧 Getting multiview data...")
    env, obj_names, q_init_upright, platform_xyz = init_env()
    
    # Target for all cameras
    target = env.get_p_body(obj_names[0]) if obj_names else np.array([1.0, 0.0, 0.8])
    
    # Camera names
    cameras = ['wrist', 'front', 'left_shoulder', 'right_shoulder']
    camera_bodies = {
        'wrist': 'camera_center',
        'front': 'front_camera',
        'left_shoulder': 'left_shoulder_camera',
        'right_shoulder': 'right_shoulder_camera'
    }
    
    # Colors for each camera
    colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0]]  # Red, Green, Blue, Yellow
    
    # Store data
    original_pcds = []
    xyz_imgs = []
    
    print("\n📷 Capturing all cameras...")
    for i, camera in enumerate(cameras):
        camera_body = camera_bodies[camera]
        p_cam, R_cam = env.get_pR_body(camera_body)
        
        # Get MuJoCo point cloud - pcd is the original one you want to see
        rgb_img, depth_img, pcd, xyz_img = env.get_egocentric_rgb_depth_pcd(
            p_ego=p_cam, p_trgt=target, rsz_rate=None, fovy=45, BACKUP_AND_RESTORE_VIEW=True
        )
        
        original_pcds.append(pcd)
        xyz_imgs.append(xyz_img)
        print(f"  {camera}: pcd shape {pcd.shape}, xyz_img shape {xyz_img.shape}")
    
    # Show BEFORE (MuJoCo's original pcd)
    print("\n🔍 Showing MuJoCo's ORIGINAL pcd...")
    geometries_before = []
    
    for i, (camera, pcd) in enumerate(zip(cameras, original_pcds)):
        # pcd is already (N, 3) format from MuJoCo
        valid_mask = np.isfinite(pcd).all(axis=1) & (np.linalg.norm(pcd, axis=1) > 0.01)
        valid_points = pcd[valid_mask]
        
        if len(valid_points) > 0:
            center = np.mean(valid_points, axis=0)
            print(f"  {camera}: {len(valid_points)} points, center [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
            
            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(valid_points)
            pcd_vis.paint_uniform_color(colors[i])
            geometries_before.append(pcd_vis)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    geometries_before.append(coord_frame)
    
    print(f"\n🎨 BEFORE: MuJoCo's original pcd output")
    print("🔴 wrist, 🟢 front, 🔵 left_shoulder, 🟡 right_shoulder")
    print("Close when ready for AFTER...")
    
    # Show BEFORE
    o3d.visualization.draw_geometries(
        geometries_before,
        window_name="BEFORE: MuJoCo's Original pcd",
        width=1200, height=800
    )
    
    # Show AFTER (FIXED: Use MuJoCo's pcd reshaped to RLBench format)
    print("\n🔍 Showing FIXED: MuJoCo pcd reshaped to RLBench format...")
    geometries_after = []
    
    for i, (camera, pcd, xyz_img) in enumerate(zip(cameras, original_pcds, xyz_imgs)):
        # FIXED METHOD: Use MuJoCo's good pcd, reshape to RLBench format
        h, w = xyz_img.shape[:2]  # Get original image dimensions
        
        # Reshape pcd (N, 3) back to image format (H, W, 3)
        pcd_img = pcd.reshape(h, w, 3)
        
        # Now resize like you would for RLBench
        pcd_resized = cv2.resize(pcd_img, (128, 128), interpolation=cv2.INTER_NEAREST)
        
        # Convert back to flat points for visualization
        points = pcd_resized.reshape(-1, 3)
        valid_mask = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 0.01)
        valid_points = points[valid_mask]
        
        if len(valid_points) > 0:
            center = np.mean(valid_points, axis=0)
            print(f"  {camera}: {len(valid_points)} points, center [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
            print(f"    Original pcd shape: {pcd.shape} -> Image: {pcd_img.shape} -> Resized: {pcd_resized.shape}")
            
            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(valid_points)
            pcd_vis.paint_uniform_color(colors[i])
            geometries_after.append(pcd_vis)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    geometries_after.append(coord_frame)
    
    print(f"\n🎨 AFTER: FIXED - MuJoCo pcd reshaped to RLBench format")
    print("🔴 wrist, 🟢 front, 🔵 left_shoulder, 🟡 right_shoulder")
    print("This should be multi-view consistent like the original!")
    
    # Show AFTER
    o3d.visualization.draw_geometries(
        geometries_after,
        window_name="AFTER: FIXED - MuJoCo pcd in RLBench format",
        width=1200, height=800
    )
    
    env.close_viewer()

if __name__ == "__main__":
    visualize_multiview_before_after()