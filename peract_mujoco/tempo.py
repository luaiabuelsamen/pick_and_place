import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/home/jetson3/luai/peract/peract_colab')
from mujoco_parser import MuJoCoParserClass, init_env


def visualize_point_cloud_o3d(pcd_points, camera_pos=None, camera_rot=None):
    """Visualize point cloud using Open3D"""
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    
    # Calculate colors based on distance from camera 
    distances = np.sqrt(np.sum((pcd_points - camera_pos)**2, axis=1))
    
    # Map distances to colors using a max distance of 3.0m
    max_display_distance = 3.0
    normalized_distances = np.minimum(distances / max_display_distance, 1.0)
    
    # Create color map (blue to red)
    colors = np.zeros((len(normalized_distances), 3))
    colors[:, 0] = normalized_distances  # Red channel
    colors[:, 2] = 1 - normalized_distances  # Blue channel
    
    # Assign colors to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # Add geometry for visualization
    geometries = [pcd, coord_frame]
    
    # Create a small sphere to represent camera position
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    camera_sphere.translate(camera_pos)
    camera_sphere.paint_uniform_color([1, 0, 0])  # Red for camera
    geometries.append(camera_sphere)
    
    # Add a frustum to represent camera view
    if camera_rot is not None:
        # Create a simple camera frustum
        camera_frustum_points = np.array([
            [0, 0, 0],  # Camera center
            [0.2, 0.15, 0.3],  # Top-right
            [0.2, -0.15, 0.3],  # Bottom-right
            [-0.2, -0.15, 0.3],  # Bottom-left
            [-0.2, 0.15, 0.3],  # Top-left
        ])
        
        # Transform to camera position and orientation
        camera_frustum_points = (camera_rot @ camera_frustum_points.T).T + camera_pos
        
        # Create lines to represent frustum
        frustum_lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # Lines connecting corners
        ]
        
        frustum_line_set = o3d.geometry.LineSet()
        frustum_line_set.points = o3d.utility.Vector3dVector(camera_frustum_points)
        frustum_line_set.lines = o3d.utility.Vector2iVector(frustum_lines)
        frustum_line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(frustum_lines))])
        geometries.append(frustum_line_set)
    
    # Create distance reference spheres
    for dist in [1.0, 2.0, 3.0]:
        dist_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        dist_sphere.translate([dist, 0, 0])
        dist_sphere.paint_uniform_color([0.7, 0.7, 0])
        geometries.append(dist_sphere)
    
    # Add a scale reference box (1m x 1m x 1m)
    reference_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    reference_box.translate([-0.5, -0.5, -0.5])  # Center at origin
    reference_box.compute_vertex_normals()
    reference_box.paint_uniform_color([0.5, 0.5, 0.5])
    # Make wireframe by reducing opacity (note: Open3D doesn't directly support alpha for meshes)
    wire_box = o3d.geometry.LineSet.create_from_triangle_mesh(reference_box)
    wire_box.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(wire_box)
    
    # Open visualization window
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud Visualization (3m limit)",
        width=1024,
        height=768,
        point_show_normal=False,
        mesh_show_wireframe=True,
        mesh_show_back_face=False,
    )


def main():
    # Initialize the environment
    print("Initializing environment...")
    env, obj_names, q_init_upright, platform_xyz = init_env()
    
    # Get the camera position and orientation
    camera = 'front'  # Try 'wrist' or 'front'
    
    # Get camera parameters
    if camera == 'wrist':
        body_name = 'camera_center'
    else:
        body_name = f'{camera}_camera'
    
    p_ego, R_ego = env.get_pR_body(body_name=body_name)
    p_trgt = p_ego + R_ego[:,2]  # Look along z-axis
    
    # Camera parameters
    fovy = 80 if camera == 'wrist' else 60
    
    # Capture images and point cloud using the existing method
    print(f"Capturing images and point cloud from {camera} camera...")
    rgb_img, depth_img, pcd, xyz_img = env.get_egocentric_rgb_depth_pcd(
        p_ego=p_ego, p_trgt=p_trgt, rsz_rate=None, fovy=fovy, BACKUP_AND_RESTORE_VIEW=True)
    
    # Print debug information
    print("\n=== DEBUG INFORMATION ===")
    print(f"RGB Image shape: {rgb_img.shape}")
    print(f"Depth Image shape: {depth_img.shape}")
    print(f"Depth range: [{np.min(depth_img):.4f}, {np.max(depth_img):.4f}]")
    print(f"Point cloud shape: {pcd.shape}")
    
    if pcd.shape[0] > 0:
        print(f"Point cloud X range: [{np.min(pcd[:,0]):.4f}, {np.max(pcd[:,0]):.4f}]")
        print(f"Point cloud Y range: [{np.min(pcd[:,1]):.4f}, {np.max(pcd[:,1]):.4f}]")
        print(f"Point cloud Z range: [{np.min(pcd[:,2]):.4f}, {np.max(pcd[:,2]):.4f}]")
    
    # Create visualization with matplotlib for RGB and depth images
    fig = plt.figure(figsize=(12, 5))
    
    # RGB Image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(rgb_img)
    ax1.set_title(f'{camera} RGB')
    ax1.axis('off')
    
    # Depth Image
    ax2 = fig.add_subplot(1, 2, 2)
    depth_plot = ax2.imshow(depth_img, cmap='viridis')
    plt.colorbar(depth_plot, ax=ax2, label='Depth (m)')
    ax2.set_title(f'{camera} Depth')
    ax2.axis('off')
    
    plt.suptitle(f'Camera RGB and Depth: {camera} camera', fontsize=16)
    plt.tight_layout()
    
    print("\nDisplaying RGB and depth images. Close the window to proceed to point cloud visualization.")
    plt.show()
    
    # Now display point cloud with Open3D if we have points
    if pcd.shape[0] > 0:
        print("\nLaunching Open3D point cloud visualizer...")
        print("Controls: Left click + drag to rotate, Right click + drag to pan, Scroll to zoom")
        print("          Hold Shift for finer control, Hold Ctrl to select points")
        
        # Filter points if there are too many (for performance)
        if pcd.shape[0] > 100000:
            sample_rate = max(1, pcd.shape[0] // 100000)
            sampled_pcd = pcd[::sample_rate]
            print(f"Sampling point cloud for visualization: {pcd.shape[0]} -> {sampled_pcd.shape[0]} points")
        else:
            sampled_pcd = pcd
        
        # Remove NaN or inf values that might cause issues in Open3D
        valid_indices = np.all(np.isfinite(sampled_pcd), axis=1)
        
        # Filter out points farther than 3 meters from the camera
        distances = np.sqrt(np.sum((sampled_pcd - p_ego)**2, axis=1))
        distance_filter = distances <= 3.0
        
        # Combine filters
        combined_filter = np.logical_and(valid_indices, distance_filter)
        cleaned_pcd = sampled_pcd[combined_filter]
        
        print(f"Filtered out {np.sum(~distance_filter)} points beyond 3 meters distance")
        print(f"Points remaining: {np.sum(combined_filter)} of {len(sampled_pcd)}")
        
        if len(cleaned_pcd) > 0:
            visualize_point_cloud_o3d(cleaned_pcd, camera_pos=p_ego, camera_rot=R_ego)
        else:
            print("Error: No valid points in point cloud after cleaning")
    else:
        print("No point cloud data available for visualization")
    
    # Display the scene in MuJoCo
    print("\nRendering MuJoCo scene...")
    env.render()
    input("Press Enter to exit...")
    
    # Clean up
    env.close_viewer()
    print("Visualization closed.")

if __name__ == "__main__":
    main()