import yaml
import os
import mujoco
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from mujoco_parser import MuJoCoParserClass
from util import sample_xyzs, rpy2r, r2quat

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

q_rev_des = np.array([[0,-90,90,-90,0,90]])*np.pi/180.0

env.init_viewer(viewer_title='UR5e with RG2 gripper',viewer_width=1200,viewer_height=800,
                viewer_hide_menus=True)
env.update_viewer(azimuth=66.08,distance=3.0,elevation=-50,lookat=[0.4,0.18,0.71],
                  VIS_TRANSPARENT=False,VIS_CONTACTPOINT=False,
                  contactwidth=0.05,contactheight=0.05,contactrgba=np.array([1,0,0,1]),
                  VIS_JOINT=True,jointlength=0.25,jointwidth=0.05,jointrgba=[0.2,0.6,0.8,0.6])
env.reset()
env.forward(q=q_rev_des, joint_idxs=env.idxs_forward)

tick = 0

while (env.get_sim_time() < 100.0) and env.is_viewer_alive(): 
    env.step(ctrl=q_rev_des[-1,:], ctrl_idxs=env.idxs_forward)
    env.step(ctrl=1, ctrl_idxs=6) #open it
    env.render()

    print(f'Joint values: {env.data.ctrl[env.idxs_forward]}')
    for obj_name in obj_names:
        print(f'Object locations: {env.get_p_body(obj_name)}')
    print(f'Gripper value: {env.data.ctrl[6]}')
    

env.close_viewer()

print("Motion planning complete and executed.")