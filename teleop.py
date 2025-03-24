import numpy as np
import matplotlib.pyplot as plt
from mujoco_parser import MuJoCoParserClass
from util import sample_xyzs, rpy2r, get_interp_const_vel_traj
import threading
import time
import queue
from PIL import Image
import os
import json
import pygame
from pygame.locals import *

image_queue = queue.Queue()
log = {}

def save_image_thread(running_event):
    while running_event.is_set() or not image_queue.empty():
        if image_queue.empty():
            continue
        run_number, img, ego, depth_img, pose_matrix, camera_intrinsics, tick = image_queue.get(timeout=0.1)
        run_folder = f"data/run_{run_number}"
        if run_number not in log:
            log[run_number] = {}
        log[run_number][tick] = {}
        log[run_number][tick]['camera_pose'] = pose_matrix.tolist()
        log[run_number][tick]['camera_intrinsics'] = camera_intrinsics.tolist()
        
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
        # img = Image.fromarray(img)
        # img_path = os.path.join(run_folder, f"image_{tick}.png")
        # img.save(img_path)
        # log[run_number][tick]['rgb'] = img_path
        
        img = Image.fromarray(ego)
        img_path = os.path.join(run_folder, f"ego_{tick}.png")
        img.save(img_path)
        log[run_number][tick]['ego'] = img_path
        
        # Save depth as 16-bit PNG for better precision
        depth_img_scaled = (depth_img * 1000).astype(np.uint16)  # Scale to millimeters
        depth_path = os.path.join(run_folder, f"depth_{tick}.png")
        cv2_depth = Image.fromarray(depth_img_scaled)
        cv2_depth.save(depth_path)
        log[run_number][tick]['depth'] = depth_path

running_event = threading.Event()
running_event.set()
run_number = 1
image_thread = threading.Thread(target=save_image_thread, args=[running_event], daemon=True)
image_thread.start()

try:
    while True:
        xml_path = 'assets/ur5e/scene_ur5e_rg2_d435i_obj.xml'
        env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=xml_path, VERBOSE=True)
        
        obj_names = [body_name for body_name in env.body_names if body_name is not None and (body_name.startswith("obj_"))]
        n_obj = len(obj_names)
        xyzs = sample_xyzs(n_sample=n_obj,
                    x_range=[1, 1], y_range=[0, 0], z_range=[0.81, 0.81], min_dist=0.2)
        colors = np.array([plt.cm.gist_rainbow(x) for x in np.linspace(0, 1, n_obj)])
        
        env.model.body('base_table').pos = np.array([0, 0, 0])
        env.model.body('front_object_table').pos = np.array([1.05, 0, 0])
        env.model.body('side_object_table').pos = np.array([0, -0.85, 0])
        env.model.body('base').pos = np.array([0.18, 0, 0.8])
        platform_xyz = np.array([0.8, 0.0, 0.81])
        # env.model.body('red_platform').pos = platform_xyz
        
        for obj_idx, obj_name in enumerate(obj_names):
            jntadr = env.model.body(obj_name).jntadr[0]
            env.model.joint(jntadr).qpos0[:3] = xyzs[obj_idx, :]
            geomadr = env.model.body(obj_name).geomadr[0]

        q_init_upright = np.array([0, -np.pi/2, 0, 0, np.pi/2, 0])
        env.reset()
        env.forward(q=q_init_upright, joint_idxs=env.idxs_forward)
        target_obj_name = obj_names[0]
        target_position = env.get_p_body(target_obj_name)

        pygame.init()
        display = pygame.display.set_mode((320, 240))
        pygame.display.set_caption('Robot Teleop Controller')
        font = pygame.font.Font(None, 30)

        env.init_viewer(viewer_title='UR5e Teleop Control', viewer_width=1200, viewer_height=800, viewer_hide_menus=True)
        env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71], 
                        VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False, contactwidth=0.05, contactheight=0.05, 
                        contactrgba=np.array([1, 0, 0, 1]), VIS_JOINT=True, jointlength=0.25, jointwidth=0.05, 
                        jointrgba=[0.2, 0.6, 0.8, 0.6])

        # Current robot state
        current_q = q_init_upright.copy()
        gripper_state = 1.0  # 1.0 = open, 0.0 = closed
        step_size = 0.01  # radians for joint movement
        tcp_step_size = 0.02  # meters for TCP movement

        # Define camera intrinsics (these values should be adjusted based on your camera)
        # For a simulated camera, you might need to compute these from the field of view
        width, height = 640, 480  # Adjust based on your image size
        fovy = 45  # vertical field of view in degrees
        aspect_ratio = width / height
        f = height / (2 * np.tan(np.radians(fovy) / 2))
        
        camera_intrinsics = np.array([
            [f, 0, width/2],
            [0, f, height/2],
            [0, 0, 1]
        ])

        record = False
        tick = 0
        R_trgt = rpy2r(np.radians([0,80,0]))@rpy2r(np.radians([-180,0,90]))

        pick_position = env.get_p_body(obj_names[0])
        pick_position[2] += 0.01
        pre_grasp_position = pick_position + np.array([0.0, 0.0, 0.1])
        q_ik_pregrasp = env.solve_ik(body_name='tcp_link', p_trgt=pre_grasp_position, R_trgt=R_trgt, IK_P=True, IK_R=False, q_init=np.array(q_init_upright), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-2, w_weight=0.5)
        current_q = q_ik_pregrasp
        while True:
            keys = pygame.key.get_pressed()
            
            if keys[K_LEFTBRACKET]:
                current_q[0] += step_size
            if keys[K_RIGHTBRACKET]:
                current_q[0] -= step_size
            if keys[K_SEMICOLON]:
                current_q[1] += step_size
            if keys[K_QUOTE]:
                current_q[1] -= step_size
            if keys[K_PERIOD]:
                current_q[2] += step_size
            if keys[K_SLASH]:
                current_q[2] -= step_size
            if keys[K_EQUALS]:
                current_q[3] += step_size
            if keys[K_MINUS]:
                current_q[3] -= step_size
            if keys[K_z]:
                current_q[4] += step_size
            if keys[K_x]:
                current_q[4] -= step_size
            if keys[K_c]:
                current_q[5] += step_size
            if keys[K_v]:
                current_q[5] -= step_size
            if keys[K_b]:
                print('Keypress')
                current_q = q_init_upright
            if keys[K_COMMA]:
                record = not record
            if keys[K_SPACE]:
                gripper_state = not gripper_state
                
            # Handle event queue for window close
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    run_number += 1
                    env.close_viewer()
                    raise KeyboardInterrupt
                    
            traj_gen = False
            ctrl = np.append(current_q, gripper_state)
            env.step(ctrl=ctrl, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
            env.render()
        
            p_cam, R_cam = env.get_pR_body(body_name='camera_center')
            p_ego = p_cam
            p_trgt = p_cam + R_cam[:, 2]

            if record:
                if tick % 2 == 0:
                    # img = env.grab_image()
                    rgb_img, depth_img, pcd, xyz_img = env.get_egocentric_rgb_depth_pcd(
                        p_ego=p_ego, p_trgt=p_trgt, rsz_rate=None, fovy=fovy, BACKUP_AND_RESTORE_VIEW=True)
                    
                    # Create the camera pose matrix (4x4 transformation matrix)
                    # This matrix represents the camera's position and orientation in world coordinates
                    pose_matrix = np.eye(4)
                    pose_matrix[:3, :3] = R_cam  # Rotation matrix
                    pose_matrix[:3, 3] = p_cam   # Translation vector
                    
                    image_queue.put((run_number, [], rgb_img, depth_img, pose_matrix, camera_intrinsics, tick))
                tick += 1

        pygame.quit()
        run_number += 1
        env.close_viewer()
        break

except KeyboardInterrupt:
    print("\nMain thread interrupted. Stopping worker thread...")
    running_event.clear()
    image_thread.join()
finally:
    with open('data/reconstruction_data.json', 'w') as json_file:
        json.dump(log, json_file, indent=4)