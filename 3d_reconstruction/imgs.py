import os
import numpy as np
import cv2
import open3d as o3d
import json
from tqdm import tqdm

json_file = "/home/jetson3/luai/pick_and_place/traj/data.json"
base_path = "/home/jetson3/luai/pick_and_place/traj"
calibration_file = "/home/jetson3/luai/pick_and_place/hand_eye_calibration_0109_65deg_checker.npz"

print("Loading JSON data...")
with open(json_file, 'r') as f:
    data = json.load(f)

print("Loading camera calibration...")
calibration = np.load(calibration_file)
rotation_matrix = calibration['rotation']
translation_vector = calibration['translation']

camera_to_ee = np.eye(4)
camera_to_ee[:3, :3] = rotation_matrix
camera_to_ee[:3, 3] = translation_vector.flatten()

print("Loaded camera extrinsics:")
print(camera_to_ee)

intrinsics = np.load('realsense.npy')
fx = intrinsics[0, 0]
fy = intrinsics[1, 1]
cx = intrinsics[0, 2]
cy = intrinsics[1, 2]

image_path = '/home/jetson3/luai/pick_and_place/traj/visual_observations/realsensecameracolorimage_raw/0_1743722294248812.png'
image = cv2.imread(image_path)
height, width = image.shape[:2]

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=width, height=height,
    fx=fx, fy=fy,
    cx=cx, cy=cy
)

combined_pcd = o3d.geometry.PointCloud()
visualizer_points = []

print("Processing images and creating point clouds...")
timestamps = list(data.keys())

for ts in timestamps:
    for inner_ts in  tqdm(data[ts]):
        try:
            rgb_path = os.path.join(base_path, data[ts][inner_ts]["camera_id"]["realsensecameracolorimage_raw"])
            depth_path = os.path.join(base_path, data[ts][inner_ts]["depth_id"]["realsensecameradepthimage_rect_raw"])
            rgb_img = cv2.imread(rgb_path)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e)
            continue
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_img),
            o3d.geometry.Image(depth_img),
            depth_scale=1000.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        
        robot_pos = data[ts][inner_ts]["robot_data"]["observations"]["obs_pos"]
        x, y, z = robot_pos[0], robot_pos[1], robot_pos[2]
        rx, ry, rz = robot_pos[3], robot_pos[4], robot_pos[5]
        
        rx_rad, ry_rad, rz_rad = np.radians([rx, ry, rz])
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx_rad), -np.sin(rx_rad)],
            [0, np.sin(rx_rad), np.cos(rx_rad)]
        ])
        
        Ry = np.array([
            [np.cos(ry_rad), 0, np.sin(ry_rad)],
            [0, 1, 0],
            [-np.sin(ry_rad), 0, np.cos(ry_rad)]
        ])
        
        Rz = np.array([
            [np.cos(rz_rad), -np.sin(rz_rad), 0],
            [np.sin(rz_rad), np.cos(rz_rad), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx
        
        ee_to_world = np.eye(4)
        ee_to_world[:3, :3] = R
        ee_to_world[:3, 3] = [x/1000.0, y/1000.0, z/1000.0]
        
        camera_to_world = ee_to_world @ camera_to_ee
        
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        coord_frame.transform(camera_to_world)
        visualizer_points.append(coord_frame)
        
        pcd.transform(camera_to_world)
        
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        distances = np.linalg.norm(points, axis=1)
        valid_indices = distances < 2.0
        
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(points[valid_indices])
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[valid_indices])
        
        filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.005)
        
        combined_pcd += filtered_pcd
    break

print(f"Created point cloud with {len(combined_pcd.points)} points")
print("Removing outliers...")
combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"After outlier removal: {len(combined_pcd.points)} points")
combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.01)
print(f"After voxel downsampling: {len(combined_pcd.points)} points")
print("Estimating normals...")
combined_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
combined_pcd.orient_normals_consistent_tangent_plane(k=15)

o3d.io.write_point_cloud("robot_environment_pointcloud.ply", combined_pcd)
print("Point cloud saved as robot_environment_pointcloud.ply")
# print("Creating mesh using Poisson reconstruction...")
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#     combined_pcd, depth=9, width=0, scale=1.1, linear_fit=False
# )
# density_threshold = np.quantile(densities, 0.1)
# vertices_to_remove = densities < density_threshold
# mesh.remove_vertices_by_mask(vertices_to_remove)
# o3d.io.write_triangle_mesh("robot_environment_mesh.ply", mesh)
# print("Mesh saved as robot_environment_mesh.ply")
# print("Visualizing point cloud with camera positions...")
# visualizer_objects = [combined_pcd] + visualizer_points[:10]
# o3d.visualization.draw_geometries(visualizer_objects)
# print("Visualizing final mesh...")
# o3d.visualization.draw_geometries([mesh])
