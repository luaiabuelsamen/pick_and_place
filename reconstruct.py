import open3d as o3d
import numpy as np
import json
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def load_reconstruction_data(run_number):
    with open("data/reconstruction_data.json", "r") as f:
        log_data = json.load(f)
    
    run_data = log_data[str(run_number)]
    
    ticks = sorted([int(tick) for tick in run_data.keys()])
    
    rgb_images = []
    depth_images = []
    camera_poses = []
    
    for tick in ticks:
        tick_str = str(tick)
        frame_data = run_data[tick_str]
        rgb_path = frame_data["ego"]
        rgb = np.array(Image.open(rgb_path))
        rgb_images.append(rgb)
        depth_path = frame_data["depth"]
        depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0
        depth_images.append(depth)
        pose = np.array(frame_data["camera_pose"])
        camera_poses.append(pose)
    camera_intrinsics = np.array(run_data[str(ticks[0])]["camera_intrinsics"])
    
    return rgb_images, depth_images, camera_poses, camera_intrinsics

def integrate_rgbd_frames(rgb_images, depth_images, camera_poses, intrinsics):
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
        
        color_o3d = o3d.geometry.Image(color)
        depth_o3d = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )
        
        volume.integrate(rgbd, o3d_intrinsics, np.linalg.inv(pose))
        if i == 150:
            break
    print("Extracting triangle mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    return mesh

def visualize_camera_trajectory(camera_poses, mesh=None):
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
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(rgb_images[0])
    axs[0].set_title("RGB Image")
    axs[1].imshow(depth_images[0])
    axs[1].set_title("Depth Image")
    plt.show()

    print("Starting 3D reconstruction...")
    mesh = integrate_rgbd_frames(rgb_images, depth_images, camera_poses, camera_intrinsics)
    o3d.io.write_triangle_mesh("data/reconstructed_mesh.ply", mesh)
    print("Mesh saved to data/reconstructed_mesh.ply")
    print("Visualizing results...")
    visualize_camera_trajectory(camera_poses, mesh)

if __name__ == "__main__":
    main()