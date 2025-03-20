import open3d as o3d
import numpy as np
import json
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
import pandas as pd

def load_reconstruction_data(run_number):
    """Load all the data needed for reconstruction from a specific run."""
    # Load the JSON data
    with open("data/reconstruction_data.json", "r") as f:
        log_data = json.load(f)
    
    if str(run_number) not in log_data:
        raise ValueError(f"Run {run_number} not found in log data")
    
    run_data = log_data[str(run_number)]
    
    # Sort frames by tick number
    ticks = sorted([int(tick) for tick in run_data.keys()])
    
    rgb_images = []
    depth_images = []
    camera_poses = []
    image_paths = []
    
    for tick in ticks:
        tick_str = str(tick)
        frame_data = run_data[tick_str]
        
        # Load RGB image (using the ego view for better results)
        rgb_path = frame_data["ego"]
        rgb = np.array(Image.open(rgb_path))
        rgb_images.append(rgb)
        image_paths.append(rgb_path)
        
        # Load depth image and convert back from millimeters
        depth_path = frame_data["depth"]
        depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0  # Convert back to meters
        depth_images.append(depth)
        
        # Load camera pose
        pose = np.array(frame_data["camera_pose"])
        camera_poses.append(pose)
    
    # Load camera intrinsics (same for all frames)
    camera_intrinsics = np.array(run_data[str(ticks[0])]["camera_intrinsics"])
    
    return rgb_images, depth_images, camera_poses, camera_intrinsics, image_paths

def manual_point_selection(mesh):
    """
    Alternative point selection method using bounding boxes
    """
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    
    # Convert to point cloud for easier visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # First show the whole point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Full Point Cloud")
    
    # Ask user for desired number of points
    num_points = int(input("How many points would you like to select? "))
    selected_points = []
    
    for i in range(num_points):
        print(f"\nSelecting point {i+1}/{num_points}")
        print("1. Provide manual XYZ coordinates for the point")
        print("2. Select from a min/max bounding box")
        choice = input("Enter your choice (1 or 2): ")
        
        if choice == '1':
            try:
                x = float(input("Enter X coordinate: "))
                y = float(input("Enter Y coordinate: "))
                z = float(input("Enter Z coordinate: "))
                selected_points.append(np.array([x, y, z]))
            except ValueError:
                print("Invalid input. Using default point.")
                # Use centroid as default point
                selected_points.append(np.mean(points, axis=0))
        else:
            # Get bounding box dimensions
            min_bound = pcd.get_min_bound()
            max_bound = pcd.get_max_bound()
            
            print(f"Point cloud bounds: Min {min_bound}, Max {max_bound}")
            print("Enter constraints for a bounding box to narrow down selection:")
            
            try:
                x_min = float(input(f"X min [{min_bound[0]:.3f}]: ") or min_bound[0])
                x_max = float(input(f"X max [{max_bound[0]:.3f}]: ") or max_bound[0])
                y_min = float(input(f"Y min [{min_bound[1]:.3f}]: ") or min_bound[1])
                y_max = float(input(f"Y max [{max_bound[1]:.3f}]: ") or max_bound[1])
                z_min = float(input(f"Z min [{min_bound[2]:.3f}]: ") or min_bound[2])
                z_max = float(input(f"Z max [{max_bound[2]:.3f}]: ") or max_bound[2])
            except ValueError:
                print("Invalid input. Using full bounds.")
                x_min, y_min, z_min = min_bound
                x_max, y_max, z_max = max_bound
            
            # Filter points within bounding box
            mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
                   (points[:, 1] >= y_min) & (points[:, 1] <= y_max) & \
                   (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
            
            filtered_points = points[mask]
            
            if len(filtered_points) == 0:
                print("No points in the specified region. Using centroid.")
                selected_points.append(np.mean(points, axis=0))
            else:
                # Visualize the filtered points
                filtered_pcd = o3d.geometry.PointCloud()
                filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
                if colors is not None:
                    filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
                
                o3d.visualization.draw_geometries([filtered_pcd], window_name=f"Filtered Points for Selection {i+1}")
                
                # Use centroid of filtered points for now
                centroid = np.mean(filtered_points, axis=0)
                selected_points.append(centroid)
                print(f"Selected point at {centroid}")
    
    return np.array(selected_points)

def select_points_from_file(mesh):
    """
    Alternative point selection method from an input file or direct coordinates
    """
    print("\nPoint Selection Options:")
    print("1. Enter coordinates interactively")
    print("2. Load points from a CSV file (x,y,z format)")
    print("3. Load points from a JSON file (list of [x,y,z] coordinates)")
    choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice == '1':
        points = []
        num_points = int(input("How many points would you like to enter? "))
        
        for i in range(num_points):
            print(f"\nEnter coordinates for point {i+1}:")
            try:
                x = float(input("X: "))
                y = float(input("Y: "))
                z = float(input("Z: "))
                points.append([x, y, z])
            except ValueError:
                print("Invalid input. Skipping this point.")
        
        return np.array(points)
    
    elif choice == '2':
        file_path = input("Enter path to CSV file: ")
        try:
            df = pd.read_csv(file_path)
            if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                points = df[['x', 'y', 'z']].values
            else:
                # Assume first three columns are x,y,z
                points = df.iloc[:, :3].values
            print(f"Loaded {len(points)} points from CSV")
            return points
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return np.array([[0, 0, 0]])  # Default point
    
    elif choice == '3':
        file_path = input("Enter path to JSON file: ")
        try:
            with open(file_path, 'r') as f:
                point_data = json.load(f)
            
            if isinstance(point_data, list) and len(point_data) > 0:
                # Check if it's a list of lists or a list of dicts
                if isinstance(point_data[0], list):
                    points = np.array(point_data)
                elif isinstance(point_data[0], dict):
                    # Try to extract x,y,z from dicts
                    points = []
                    for p in point_data:
                        if all(k in p for k in ['x', 'y', 'z']):
                            points.append([p['x'], p['y'], p['z']])
                    points = np.array(points)
                else:
                    raise ValueError("JSON format not recognized")
                
                print(f"Loaded {len(points)} points from JSON")
                return points
            else:
                raise ValueError("JSON must contain a list of points")
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return np.array([[0, 0, 0]])  # Default point
    
    # Default fallback
    return np.array([[0, 0, 0]])

def visualize_selected_points(mesh, points):
    """Create a visualization with red spheres marking the picked points."""
    # Convert to Open3D format
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Create a point cloud for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    # Create a separate point cloud for the selected points
    selected_pcd = o3d.geometry.PointCloud()
    selected_pcd.points = o3d.utility.Vector3dVector(points)
    selected_pcd.paint_uniform_color([1, 0, 0])  # Red color
    
    # Increase point size for selected points
    selected_pcd.points = o3d.utility.Vector3dVector(points)
    
    # Show mesh and selected points
    o3d.visualization.draw_geometries([mesh, selected_pcd], 
                                     window_name="Mesh with Selected Points",
                                     point_show_normal=False)
    
    return

def project_3d_to_2d(point_3d, camera_pose, intrinsics):
    """Project a 3D point to 2D image coordinates."""
    # Convert point to camera coordinates
    # The camera pose is world-to-camera, so we apply it directly
    point_homogeneous = np.append(point_3d, 1)
    point_camera = camera_pose @ point_homogeneous
    
    # Check if point is in front of camera
    if point_camera[2] <= 0:
        return None  # Point is behind the camera
    
    # Project to image plane
    x_proj = point_camera[0] / point_camera[2]
    y_proj = point_camera[1] / point_camera[2]
    
    # Apply camera intrinsics
    u = intrinsics[0, 0] * x_proj + intrinsics[0, 2]
    v = intrinsics[1, 1] * y_proj + intrinsics[1, 2]
    
    # Return pixel coordinates if within image bounds
    return (int(round(u)), int(round(v)))

def create_projection_visualization(rgb_images, camera_poses, intrinsics, points_3d, image_paths, output_dir):
    """Project 3D points onto all images and save the results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create result dictionary
    result = {
        "points_3d": points_3d.tolist(),
        "projections": {}
    }
    
    # Process each image
    for i, (image, pose, img_path) in enumerate(zip(rgb_images, camera_poses, image_paths)):
        height, width = image.shape[:2]
        vis_image = image.copy()
        
        # Project each 3D point to this image
        projections = []
        for j, point in enumerate(points_3d):
            proj = project_3d_to_2d(point, pose, intrinsics)
            if proj is not None:
                u, v = proj
                # Check if projection is within image bounds
                if 0 <= u < width and 0 <= v < height:
                    # Draw a marker on the image
                    cv2.drawMarker(vis_image, (u, v), (0, 0, 255), 
                                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                    cv2.putText(vis_image, f"#{j}", (u+5, v+5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    projections.append({"point_id": j, "u": u, "v": v})
        
        # Save the visualization
        output_path = os.path.join(output_dir, f"projection_{i}.png")
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Add to result dictionary
        result["projections"][img_path] = projections
    
    # Save the projection data as JSON
    with open(os.path.join(output_dir, "projections.json"), "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Projections saved to {output_dir}/projections.json")
    return result

def main():
    # Configuration
    run_number = 1
    output_dir = "data/projections"
    
    # Load data
    print("Loading reconstruction data...")
    rgb_images, depth_images, camera_poses, camera_intrinsics, image_paths = load_reconstruction_data(run_number)
    
    # Either load or create the mesh
    mesh_path = "data/reconstructed_mesh.ply"
    if os.path.exists(mesh_path):
        print(f"Loading existing mesh from {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
    else:
        print("Reconstructing mesh from RGBD data...")
        # You would need to implement/call your reconstruction function here
        print("No existing mesh found. Please run reconstruction first.")
        return
    
    # Show the mesh first
    print("Showing the reconstructed mesh. Close the window to continue.")
    o3d.visualization.draw_geometries([mesh], window_name="Reconstructed Mesh")
    
    # Select points on the mesh (using alternative methods)
    print("\nPoint selection methods:")
    print("1. Manual point selection with bounding boxes")
    print("2. Enter coordinates or load from file")
    method = input("Select method (1 or 2): ")
    
    if method == '1':
        points_3d = manual_point_selection(mesh)
    else:
        points_3d = select_points_from_file(mesh)
    
    if len(points_3d) == 0:
        print("No points selected. Exiting.")
        return
    
    # Visualize selected points
    print(f"Selected {len(points_3d)} points:")
    for i, point in enumerate(points_3d):
        print(f"Point #{i}: [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]")
    
    visualize_selected_points(mesh, points_3d)
    
    # Project points to images and save results
    print("Projecting points to images...")
    projection_data = create_projection_visualization(
        rgb_images, camera_poses, camera_intrinsics, points_3d, image_paths, output_dir)

if __name__ == "__main__":
    main()