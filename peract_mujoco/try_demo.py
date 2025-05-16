import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/home/jetson3/luai/peract/peract_colab')

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from mujoco_parser import MuJoCoParserClass, init_env
from tqdm import tqdm
from util import quat2r  # Import quaternion to rotation matrix conversion

# Path to your dataset
DATA_PATH = "/home/jetson3/luai/peract/peract_colab/data/colab_dataset/open_drawer/all_variations/episodes"

def load_and_execute_demo(episode_number=0, use_cartesian=False):
    env, obj_names, q_init_upright, platform_xyz = init_env()
    
    print(f"==== Loading and Executing Demo {episode_number} ====")
    print(f"Execution mode: {'Cartesian (IK)' if use_cartesian else 'Joint space'}")
    
    episode_dir = f"{DATA_PATH}/episode{episode_number}"
    demo_file = os.path.join(episode_dir, 'low_dim_obs.pkl')
    
    if not os.path.exists(demo_file):
        print(f"Demo file not found: {demo_file}")
        return
    
    print(f"Loading demo from {demo_file}")
    with open(demo_file, 'rb') as f:
        demo = pickle.load(f)
    
    # Check if we're using observations or _observations attribute
    if hasattr(demo, 'observations'):
        observations = demo.observations
    elif hasattr(demo, '_observations'):
        observations = demo._observations
    else:
        print("Error: Could not find observations in demo file")
        return
    
    num_steps = len(observations)
    print(f"Loaded {num_steps} steps from demo")
    
    joint_positions = []
    gripper_poses = []
    gripper_states = []
    
    for obs in observations:
        if hasattr(obs, 'joint_positions'):
            joint_positions.append(obs.joint_positions)
        if hasattr(obs, 'gripper_pose'):
            gripper_poses.append(obs.gripper_pose)
        if hasattr(obs, 'gripper_open'):
            gripper_states.append(1.0 if obs.gripper_open else 0.0)
    
    joint_positions = np.array(joint_positions)
    gripper_poses = np.array(gripper_poses)
    gripper_states = np.array(gripper_states)
    
    if len(joint_positions) > 0:
        print("\nJoint position ranges:")
        for i in range(joint_positions.shape[1]):
            min_val = np.min(joint_positions[:, i])
            max_val = np.max(joint_positions[:, i])
            print(f"  Joint {i}: {np.degrees(min_val):.2f}° to {np.degrees(max_val):.2f}°")
    
    # Check if we have the required data
    if use_cartesian and len(gripper_poses) == 0:
        print("Error: Cartesian execution requested but no gripper poses found in demo.")
        print("Falling back to joint space execution.")
        use_cartesian = False
    
    if not use_cartesian and len(joint_positions) == 0:
        print("Error: No joint positions found in demo.")
        return
    
    # Initialize robot
    print("\nInitializing robot...")
    zero_position = np.zeros(len(env.idxs_forward))
    env.step(zero_position, env.idxs_forward)
    
    for _ in range(10):
        env.render()
    
    initial_position = joint_positions[0]
    
    print(f"Initial position: {initial_position}")
    env.step(initial_position, env.idxs_forward)
    
    for _ in range(20):
        env.render()

    if use_cartesian:
        print("Executing in Cartesian space using IK...")
        current_q = initial_position.copy()
        
        for i in tqdm(range(len(gripper_poses))):
            pose = gripper_poses[i]
            p_target = pose[:3]
            quat = pose[3:]
            R_target = quat2r(quat)
            gripper_state = gripper_states[i] if i < len(gripper_states) else 1.0
            q_ik = env.solve_ik(body_name='tcp_link', p_trgt=p_target, R_trgt=R_target, IK_P=True, IK_R=False, q_init=current_q, idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-2, w_weight=0.5)
            current_q = q_ik.copy()
            full_control = np.append(current_q, gripper_state)
            env.step(full_control, env.idxs_forward + [6])
            for _ in range(10):
                env.render()
            
    else:
        print("Executing in joint space...")
        for i in tqdm(range(len(joint_positions))):
            q = joint_positions[i]
            gripper_state = gripper_states[i] if i < len(gripper_states) else 1.0
            full_control = np.append(q, gripper_state)
            
            env.step(full_control, env.idxs_forward + [6])
            for _ in range(10):
                env.render()
    
    print("Trajectory execution completed.")

    # Close environment
    env.close_viewer()
    
    return {
        'episode': episode_number,
        'num_steps': num_steps,
        'joint_positions': joint_positions,
        'gripper_poses': gripper_poses,
        'gripper_states': gripper_states
    }

if __name__ == "__main__":
    episode_number = 1
    use_cartesian = False
    
    load_and_execute_demo(episode_number=episode_number,
                          use_cartesian=use_cartesian)