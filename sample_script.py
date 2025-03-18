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

image_queue = queue.Queue()
log = {}

def save_image_thread(running_event):
    while running_event.is_set() or not image_queue.empty():
        if image_queue.empty():
            continue
        run_number, img, ego, pos, tick = image_queue.get(timeout=0.1)
        run_folder = f"data/run_{run_number}"
        if run_number not in log:
            log[run_number] = {}
        log[run_number][tick] = {}
        log[run_number][tick]['pos'] = list(pos)
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
        img = Image.fromarray(img)
        img_path = os.path.join(run_folder, f"image_{tick}.png")
        img.save(img_path)
        log[run_number][tick]['rgb'] = img_path
        img = Image.fromarray(ego)
        img_path = os.path.join(run_folder, f"ego_{tick}.png")
        img.save(img_path)
        log[run_number][tick]['ego'] = img_path

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
                    x_range=[0.75, 1.25],y_range=[-0.38,0.38],z_range=[0.81,0.81],min_dist=0.2)
        colors = np.array([plt.cm.gist_rainbow(x) for x in np.linspace(0, 1, n_obj)])

        # Move tables and robot base
        env.model.body('base_table').pos = np.array([0,0,0])
        env.model.body('front_object_table').pos = np.array([1.05,0,0])
        env.model.body('side_object_table').pos = np.array([0,-0.85,0])
        env.model.body('base').pos = np.array([0.18,0,0.8])

        for obj_idx, obj_name in enumerate(obj_names):
            jntadr = env.model.body(obj_name).jntadr[0]
            env.model.joint(jntadr).qpos0[:3] = xyzs[obj_idx, :]
            geomadr = env.model.body(obj_name).geomadr[0]

        platform_xyz = np.random.uniform([0.6, -0.3, 0.81], [1.0, 0.3, 0.81])
        env.model.body('red_platform').pos = platform_xyz

        q_init_upright = np.array([0, -np.pi/2, 0, 0, np.pi/2, 0])
        env.reset()
        env.forward(q=q_init_upright, joint_idxs=env.idxs_forward)
        R_trgt = rpy2r(np.radians([-180, 0, 90]))

        pick_position = env.get_p_body(obj_names[0])
        pick_position[2] += 0.005
        pre_grasp_position = pick_position + np.array([0.0, 0.0, 0.1])
        q_ik_pregrasp = env.solve_ik(body_name='tcp_link', p_trgt=pre_grasp_position, R_trgt=R_trgt, IK_P=True, IK_R=True, q_init=np.array(q_init_upright), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-3, w_weight=0.5)
        q_ik = env.solve_ik(body_name='tcp_link', p_trgt=pick_position, R_trgt=R_trgt, IK_P=True, IK_R=True, q_init=np.array(q_ik_pregrasp), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-3, w_weight=0.5)
        post_pick_position = pick_position + np.array([0.0, 0.0, 0.1])
        q_ik_postpick = env.solve_ik(body_name='tcp_link', p_trgt=post_pick_position, R_trgt=R_trgt, IK_P=True, IK_R=True, q_init=np.array(q_ik), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-3, w_weight=0.5)

        q_traj_combined = np.vstack([q_init_upright, q_ik_pregrasp, q_ik])
        times, q_traj = get_interp_const_vel_traj(q_traj_combined, vel=np.radians(90), HZ=env.HZ)

        place_position = platform_xyz + np.array([0.0, 0.0, 0.1])
        q_ik_place = env.solve_ik(body_name='tcp_link', p_trgt=place_position, R_trgt=R_trgt, IK_P=True, IK_R=True, q_init=np.array(q_traj[-1, :]), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-3, w_weight=0.5)
        pre_place_position = place_position + np.array([0.0, 0.0, 0.1])
        q_ik_preplace = env.solve_ik(body_name='tcp_link', p_trgt=pre_place_position, R_trgt=R_trgt, IK_P=True, IK_R=True, q_init=np.array(q_ik_place), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-3, w_weight=0.5)

        q_traj_place_combined = np.vstack([q_ik_postpick, q_ik_preplace, q_ik_place])
        times_place, q_traj_place = get_interp_const_vel_traj(q_traj_place_combined, vel=np.radians(90), HZ=env.HZ)

        env.init_viewer(viewer_title='UR5e with RG2 gripper', viewer_width=1200, viewer_height=800, viewer_hide_menus=True)
        env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71], VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False, contactwidth=0.05, contactheight=0.05, contactrgba=np.array([1, 0, 0, 1]), VIS_JOINT=True, jointlength=0.25, jointwidth=0.05, jointrgba=[0.2, 0.6, 0.8, 0.6])

        tick = 0
        gripper_closed_steps_1 = np.tile(np.append(q_traj[-1], 0.5), (150, 1))
        gripper_closed_steps_2 = np.tile(np.append(q_traj_place[-1], 1), (150, 1))

        q_traj_combined = np.vstack([
            np.hstack([q_traj, np.ones((q_traj.shape[0], 1))]),
            gripper_closed_steps_1,
            np.hstack([q_traj_place, np.ones((q_traj_place.shape[0], 1))*0.5]),
            gripper_closed_steps_2,
        ])

        tick = 0
        while tick < q_traj_combined.shape[0]:
            p_tcp,R_tcp = env.get_pR_body(body_name='tcp_link')
            p_cam,R_cam = env.get_pR_body(body_name='camera_center')
            p_base,R_base = env.get_pR_body(body_name='base')
            # Get PCD from a specific view
            p_ego  = p_cam
            p_trgt = p_cam + R_cam[:,2]
            if tick % 10 == 0:
                img = env.grab_image()
                rgb_img,depth_img,pcd,xyz_img = env.get_egocentric_rgb_depth_pcd(
                    p_ego=p_ego,p_trgt=p_trgt,rsz_rate=None,fovy=45,BACKUP_AND_RESTORE_VIEW=True)
                pos = env.get_q([0, 1, 2, 3, 4, 5, 6])
                image_queue.put((run_number, img, rgb_img, pos, tick))
            q = q_traj_combined[tick, :]
            env.step(ctrl=q, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
            env.render()
            tick += 1

        run_number += 1
        env.close_viewer()

except KeyboardInterrupt:
    print("\nMain thread interrupted. Stopping worker thread...")
    running_event.clear()
    image_thread.join()
finally:
    with open('data.json', 'w') as json_file:
        json.dump(log, json_file, indent=4)
    