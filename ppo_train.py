import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class PickPlaceEnv(gym.Env):
    def __init__(self):
        super(PickPlaceEnv, self).__init__()
        
        # Load the MuJoCo environment
        from mujoco_parser import MuJoCoParserClass
        from util import sample_xyzs
        import matplotlib.pyplot as plt

        xml_path = 'assets/ur5e/scene_ur5e_rg2_d435i_obj.xml'
        self.env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=xml_path, VERBOSE=True)
        
        # Object setup
        self.obj_names = [body_name for body_name in self.env.body_names
                          if body_name is not None and (body_name.startswith("obj_"))]
        self.n_obj = len(self.obj_names)
        xyzs = sample_xyzs(n_sample=self.n_obj,
                           x_range=[0.75, 1.25], y_range=[-0.38, 0.38], z_range=[0.81, 0.81], min_dist=0.2)
        for obj_idx, obj_name in enumerate(self.obj_names):
            jntadr = self.env.model.body(obj_name).jntadr[0]
            self.env.model.joint(jntadr).qpos0[:3] = xyzs[obj_idx, :]
        
        # Action space: 6 joints + 1 gripper
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation space: joint states + object positions
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_obj * 3 + 7,), dtype=np.float32
        )

        self.env.reset()

    def reset(self):
        self.env.reset()
        self.env.forward()
        state = self._get_obs()
        return state

    def step(self, action):
        # Apply action to the environment
        joint_action = action[:6]  # First 6 for joints
        gripper_action = action[6]  # Last for gripper
        self.env.step(ctrl=joint_action, ctrl_idxs=self.env.idxs_forward)
        self.env.step(ctrl=gripper_action, ctrl_idxs=6)

        # Get new observation
        state = self._get_obs()

        # Calculate reward
        reward = self._compute_reward()

        # Check if the task is complete
        done = self._check_done()

        # Optional additional information
        info = {}

        return state, reward, done, info

    def render(self, mode="human"):
        self.env.render()

    def close(self):
        self.env.close_viewer()

    def _get_obs(self):
        # Joint states + gripper + object positions
        joint_states = self.env.data.ctrl[self.env.idxs_forward]
        gripper_state = self.env.data.ctrl[6]
        object_positions = np.array([self.env.get_p_body(obj_name) for obj_name in self.obj_names]).flatten()
        return np.concatenate([joint_states, [gripper_state], object_positions])

    def _compute_reward(self):
        target_position = self.env.model.body("front_object_table").pos
        object_positions = np.array([self.env.get_p_body(obj_name) for obj_name in self.obj_names])
        distances = np.linalg.norm(object_positions - target_position, axis=1)
        reward = -np.min(distances)  # Negative distance as reward
        return reward

    def _check_done(self):
        target_position = self.env.model.body("front_object_table").pos
        object_positions = np.array([self.env.get_p_body(obj_name) for obj_name in self.obj_names])
        distances = np.linalg.norm(object_positions - target_position, axis=1)
        return np.any(distances < 0.1)

if __name__ == '__main__':
    env = PickPlaceEnv()
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    model.save("ppo_pick_and_place")

    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
