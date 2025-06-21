import numpy as np
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import cv2

@dataclass
class Demonstration:
    observations: List[Dict[str, np.ndarray]]  # List of observations (images and joint positions)
    actions: List[np.ndarray]  # List of actions (joint configurations + gripper actions)
    rewards: List[float]  # List of rewards
    done: List[bool]  # List of done flags
    info: Dict  # Metadata or additional information (e.g., success)

class DemonstrationCollector:
    def __init__(self, env, save_dir="demonstrations", image_size=(64, 64)):
        self.env = env
        self.save_dir = save_dir
        self.image_size = image_size
        os.makedirs(save_dir, exist_ok=True)

    def _capture_observation(self):
        """Capture and process the image and proprioception (joint positions) from the environment"""
        rgb_img = self.env.grab_rgb_depth_img()[0]  # Get RGB image
        rgb_img_resized = cv2.resize(rgb_img, self.image_size, interpolation=cv2.INTER_AREA)
        joint_positions = self.env.data.ctrl[self.env.idxs_forward]  # Get joint positions
        return {'image': rgb_img_resized, 'proprioception': joint_positions}
    
    def collect_demonstration(self, q_traj, q_traj_place):
        observations = []
        actions = []
        rewards = []
        dones = []
        
        tick = 0
        gripper_opened = False
        pre_place_position = self.env.get_p_body('tcp_link')
        
        while tick < (q_traj.shape[0] + 150 + q_traj_place.shape[0]):
            # Part 1: Executing q_traj movements
            if tick < q_traj.shape[0]:
                q = np.append(q_traj[tick, :], 1.0)  # Open gripper
                self.env.step(ctrl=q, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
                self.env.render()

                # Collect data
                observation = self._capture_observation()
                observations.append(observation)
                actions.append(q)

            # Part 2: Holding position
            elif tick < q_traj.shape[0] + 150:
                q = np.append(q_traj[-1, :], 0.0)  # Gripper closed
                self.env.step(ctrl=q, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
                self.env.render()

                # Collect data
                observation = self._capture_observation()
                observations.append(observation)
                actions.append(q)

            # Part 3: Placement movements
            else:
                place_tick = tick - (q_traj.shape[0] + 150)
                if place_tick < q_traj_place.shape[0]:
                    q_place = q_traj_place[place_tick, :]
                    tcp_position = self.env.get_p_body('tcp_link')

                    if not gripper_opened and np.linalg.norm(tcp_position - pre_place_position) < 0.01:
                        gripper_opened = True
                        q_place = np.append(q_place, 1.0)  # Open gripper for release
                    else:
                        q_place = np.append(q_place, 0.0 if not gripper_opened else 1.0)

                    self.env.step(ctrl=q_place, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
                    self.env.render()

                    # Collect data
                    observation = self._capture_observation()
                    observations.append(observation)
                    actions.append(q_place)

            # Calculate reward based on distances
            tcp_pos = self.env.get_p_body('tcp_link')
            obj_pos = self.env.get_p_body(self.env.obj_names[0])
            target_pos = self.env.get_p_body('red_platform')

            dist_tcp_obj = np.linalg.norm(tcp_pos - obj_pos)
            dist_obj_target = np.linalg.norm(obj_pos - target_pos)

            reward = -0.1 * dist_tcp_obj + (10.0 if dist_tcp_obj < 0.05 else 0.0) + \
                     -0.1 * dist_obj_target + (50.0 if dist_obj_target < 0.05 else 0.0)
            rewards.append(reward)

            # Check if done
            done = dist_obj_target < 0.05
            dones.append(done)

            tick += 1

        # Final demonstration success info (metadata)
        demo_info = {"success": any(dones)}

        # Create and return the demonstration
        return Demonstration(
            observations=observations,
            actions=actions,
            rewards=rewards,
            done=dones,
            info=demo_info
        )

    def save_demonstration(self, demonstration, filename):
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(demonstration, f)

    def load_demonstration(self, filename):
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'rb') as f:
            return pickle.load(f)
