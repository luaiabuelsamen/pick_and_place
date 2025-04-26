import open3d as o3d
import numpy as np
import json
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def load_reconstruction_data(run_number):
    """Load data from the new JSON schema structure"""
    with open("traj/data.json", "r") as f:
        log_data = json.load(f)
    
    run_data = log_data[str(run_number)]
    
    ticks = sorted([int(tick) for tick in run_data.keys()])
    
    rgb_images = []
    depth_images = []
    camera_poses = []
    
    base_path = "/home/horowitzlab/dataset/"
    
    # Assuming the first frame has camera intrinsics that we can use for all frames
    # If camera intrinsics aren't available in the JSON, we'll need to set default values
    camera_intrinsics = np.array([
        [615.0, 0.0, 320.0],
        [0.0, 615.0, 240.0], 
        [0.0, 0.0, 1.0]
    ])  # Default intrinsics, replace with actual values if available
    
    for tick in ticks:
        tick_str = str(tick)
        frame_data = run_data[tick_str]
        
        # Get the RGB image path
        rgb_path = os.path.join(base_path, frame_data["camera_id"]["realsensecameracolorimage_raw"])
        rgb = np.array(Image.open(rgb_path))
        rgb_images.append(rgb)
        
        # Get the depth image path
        depth_path = os.path.join(base_path, frame_data["depth_id"]["realsensecameradepthimage_rect_raw"])
        depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0  # Convert to meters
        depth_images.append(depth)
        
        # Get camera pose from robot data
        # Since the new JSON doesn't have explicit camera poses, we'll derive them from robot position
        # This is an approximation and may need to be adjusted based on the robot's configuration
        robot_pos = np.array(frame_data["robot_data"]["observations"]["obs_pos"])
        
        # Create a simple transformation matrix from robot position and orientation
        # This is a placeholder - you'll need to adjust this based on your actual robot setup
        pose = np.eye(4)
        pose[:3, 3] = robot_pos[:3] / 1000.0  # Convert from mm to m (assuming units are in mm)
        
        # For rotation, we need to convert from robot's representation to a 3x3 rotation matrix
        # This is a simplified example, adjust based on your robot's coordinate system
        # Here assuming the last 3 values in obs_pos are Euler angles in degrees
        if len(robot_pos) >= 6:
            rx, ry, rz = np.radians(robot_pos[3:6])
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]
            ])
            Ry = np.array([
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]
            ])
            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]
            ])
            R = Rz @ Ry @ Rx
            pose[:3, :3] = R
        
        camera_poses.append(pose)
    
    return rgb_images, depth_images, camera_poses, camera_intrinsics

def integrate_rgbd_frames(rgb_images, depth_images, camera_poses, intrinsics):
    """Integrate RGB-D frames to create a 3D mesh"""
    # Check if we have at least one depth image
    if not depth_images:
        raise ValueError("No depth images found")
    
    height, width = depth_images[0].shape
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=intrinsics[0, 0],
        fy=intrinsics[1, 1],
        cx=intrinsics[0, 2],
        cy=intrinsics[1, 2]
    )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.005,
        sdf_trunc=0.02,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    for i, (color, depth, pose) in enumerate(zip(rgb_images, depth_images, camera_poses)):
        print(f"Integrating frame {i+1}/{len(rgb_images)}")
        
        # Convert images to Open3D format
        color_o3d = o3d.geometry.Image(color)
        depth_o3d = o3d.geometry.Image(depth)
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )
        
        # Integrate the RGBD frame into the volume
        volume.integrate(rgbd, o3d_intrinsics, np.linalg.inv(pose))
        
        # Optional: limit number of frames for testing
        if i == 5:
            break
    
    print("Extracting triangle mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    return mesh

def visualize_camera_trajectory(camera_poses, mesh=None):
    """Visualize camera trajectory and optionally the reconstructed mesh"""
    camera_frames = []
    for pose in camera_poses:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.transform(pose)
        camera_frames.append(frame)
    
    if mesh is not None:
        o3d.visualization.draw_geometries([mesh, *camera_frames])
    else:
        o3d.visualization.draw_geometries(camera_frames)

def main():
    run_number = 1
    
    print(f"Loading data from run {run_number}...")
    rgb_images, depth_images, camera_poses, camera_intrinsics = load_reconstruction_data(run_number)
    
    print(f"Found {len(rgb_images)} frames")
    
    # Display first frame for verification
    if rgb_images and depth_images:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(rgb_images[0])
        axs[0].set_title("RGB Image")
        axs[1].imshow(depth_images[0])
        axs[1].set_title("Depth Image")
        plt.show()

    print("Starting 3D reconstruction...")
    mesh = integrate_rgbd_frames(rgb_images, depth_images, camera_poses, camera_intrinsics)
    
    # Create the output directory if it doesn't exist
    os.makedirs("traj/output", exist_ok=True)
    
    # Save the mesh
    output_path = "traj/output/reconstructed_mesh.ply"
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Mesh saved to {output_path}")
    
    print("Visualizing results...")
    visualize_camera_trajectory(camera_poses, mesh)

if __name__ == "__main__":
    main()