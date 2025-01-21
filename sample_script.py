import yaml
import os
import mujoco
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from mujoco_parser import MuJoCoParserClass
from util import sample_xyzs, rpy2r, get_interp_const_vel_traj

xml_path = 'assets/ur5e/scene_ur5e_rg2_d435i_obj.xml'
env = MuJoCoParserClass(name='UR5e with RG2 gripper',rel_xml_path=xml_path,VERBOSE=True)
obj_names = [body_name for body_name in env.body_names
             if body_name is not None and (body_name.startswith("obj_"))]
n_obj = len(obj_names)

xyzs = sample_xyzs(n_sample=n_obj,
                   x_range=[0.75, 1.25],y_range=[-0.38,0.38],z_range=[0.81,0.81],min_dist=0.2)
colors = np.array([plt.cm.gist_rainbow(x) for x in np.linspace(0,1,n_obj)])
for obj_idx,obj_name in enumerate(obj_names):
    jntadr = env.model.body(obj_name).jntadr[0]
    env.model.joint(jntadr).qpos0[:3] = xyzs[obj_idx,:]
    geomadr = env.model.body(obj_name).geomadr[0]

env.model.body('base_table').pos = np.array([0,0,0])
env.model.body('front_object_table').pos = np.array([1.05,0,0])
env.model.body('side_object_table').pos = np.array([0,-0.85,0])
env.model.body('base').pos = np.array([0.18,0,0.8])

q_init_upright = np.array([0,-np.pi/2,0,0,np.pi/2,0])
env.reset()
env.forward(q=q_init_upright, joint_idxs=env.idxs_forward)

tick = 0

R_trgt = rpy2r(np.radians([-180,0,90]))
cylinder_height = 0.3
cylinder_radius = 0.025
pick_position = env.get_p_body(obj_names[0])
place_position = env.get_p_body("side_object_table") 
place_position =  [0,-0.85, cylinder_height / 2 + 0.81]

pre_grasp_position = pick_position + np.array([-0.1, -0.1, 0.1])
q_ik_pregrasp = env.solve_ik(
    body_name='tcp_link',p_trgt=pre_grasp_position,R_trgt=R_trgt,
    IK_P=True,IK_R=True, q_init=np.array(q_init_upright),idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian,
    RESET=False,DO_RENDER=False,render_every=1,th=1*np.pi/180.0,err_th=1e-3, w_weight=0.5)
q_ik = env.solve_ik(
    body_name='tcp_link',p_trgt=pick_position,R_trgt=R_trgt,
    IK_P=True,IK_R=True, q_init=np.array(q_ik_pregrasp),idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian,
    RESET=False,DO_RENDER=False,render_every=1,th=1*np.pi/180.0,err_th=1e-3, w_weight=0.5)

q_ik_drop = env.solve_ik(
    body_name='tcp_link',
    p_trgt=place_position,
    R_trgt=R_trgt,
    IK_P=True, IK_R=True,
    q_init=q_ik,
    idxs_forward=env.idxs_forward,
    idxs_jacobian=env.idxs_jacobian,
    RESET=False, DO_RENDER=False,
    th=1 * np.pi / 180.0, err_th=1e-3, w_weight=0.5
)

# Generate full trajectory
q_traj_combined = np.vstack([q_init_upright, q_ik_pregrasp, q_ik, q_ik_drop])
times, q_traj = get_interp_const_vel_traj(q_traj_combined, vel=np.radians(30), HZ=env.HZ)
print ("Joint trajectory ready. duration:[%.2f]sec"%(times[-1]))


env.init_viewer(viewer_title='UR5e with RG2 gripper',viewer_width=1200,viewer_height=800,
                viewer_hide_menus=True)
env.update_viewer(azimuth=66.08,distance=3.0,elevation=-50,lookat=[0.4,0.18,0.71],
                  VIS_TRANSPARENT=False,VIS_CONTACTPOINT=False,
                  contactwidth=0.05,contactheight=0.05,contactrgba=np.array([1,0,0,1]),
                  VIS_JOINT=True,jointlength=0.25,jointwidth=0.05,jointrgba=[0.2,0.6,0.8,0.6])

tick,max_sec = 0,1000
while env.get_sim_time() <= max_sec:
    # q = q_traj[tick, :]
    # Step
    if tick < q_traj.shape[0]:
        if np.linalg.norm(env.get_p_body('tcp_link')-pick_position) < 0.01:
            q = np.append(q_traj[tick, :],0.0) # close gripper
        else:
            q = np.append(q_traj[tick, :],1.0) # open gripper
        env.step(ctrl=q,ctrl_idxs=[0,1,2,3,4,5,6])
    else:
        q = np.append(q_traj[-1, :],0.0) # close gripper
    env.render()

    # print(f'Joint values: {env.data.ctrl[env.idxs_forward]}')
    # for obj_name in obj_names:
    #     print(f'Object locations: {env.get_p_body(obj_name)}')
    # print(f'Gripper value: {env.data.ctrl[6]}')
    tick += 1
    

env.close_viewer()