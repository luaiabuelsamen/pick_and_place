import gymnasium as gym
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from mujoco_parser import MuJoCoParserClass
from util import sample_xyzs

class PickPlaceEnvImage(gym.Env):
    def __init__(self, add_proprioception=True, use_viewer=False):
        super(PickPlaceEnvImage, self).__init__()
        xml_path = 'assets/ur5e/scene_ur5e_rg2_d435i_obj.xml'
        self.env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=xml_path, VERBOSE=True)
        self.use_viewer = use_viewer

        # Object and scene initialization
        self.obj_names = [body_name for body_name in self.env.body_names if body_name.startswith("obj_")]
        self.env.model.body('base_table').pos = np.array([0, 0, 0])
        self.env.model.body('front_object_table').pos = np.array([1.05, 0, 0])
        self.env.model.body('side_object_table').pos = np.array([0, -0.85, 0])
        self.env.model.body('base').pos = np.array([0.18, 0, 0.8])

        self._randomize_scene()
        self.env.reset()
        self.env.forward(np.array([0, -np.pi / 2, 0, 0, np.pi / 2, 0]), joint_idxs=self.env.idxs_forward)

        # Viewer setup
        self.env.init_viewer(viewer_title='UR5e Simulation', viewer_width=1200, viewer_height=800, viewer_hide_menus=True)
        self.env.update_viewer(azimuth=66, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71])

        # Observation and action spaces
        self.add_proprioception = add_proprioception
        self.image_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.proprio_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.observation_space = (
            gym.spaces.Dict({'image': self.image_space, 'proprioception': self.proprio_space})
            if self.add_proprioception else self.image_space
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-np.pi] * 6 + [0.0]),
            high=np.array([np.pi] * 6 + [1.0]),
            dtype=np.float32
        )

        self.episode_steps = 0
        self.max_episode_steps = 1000

    def reset(self, seed=None, options=None):
        self.env.reset()
        # Randomize initial joint positions
        self.env.forward(np.random.uniform(-np.pi, np.pi, size=6), joint_idxs=self.env.idxs_forward)
        self._randomize_scene()
        self.episode_steps = 0
        obs = self._get_obs()
        print("Environment reset.")
        return obs, {}

    def _randomize_scene(self):
        n_obj = len(self.obj_names)
        xyzs = sample_xyzs(
            n_sample=n_obj, x_range=[0.75, 1.25], y_range=[-0.38, 0.38], z_range=[0.81, 0.81], min_dist=0.2
        )
        for obj_idx, obj_name in enumerate(self.obj_names):
            jntadr = self.env.model.body(obj_name).jntadr[0]
            self.env.model.joint(jntadr).qpos0[:3] = xyzs[obj_idx]

    def step(self, action):
        print(f"Action taken: {action}")
        joint_positions = np.clip(action[:6], self.action_space.low[:6], self.action_space.high[:6])
        gripper_action = np.clip(action[6], 0.0, 1.0)

        # Apply actions
        self.env.step(ctrl=joint_positions, ctrl_idxs=self.env.idxs_forward)
        self.env.step(ctrl=gripper_action, ctrl_idxs=6)

        tcp_pos = self.env.get_p_body('tcp_link')
        obj_pos = self.env.get_p_body(self.obj_names[0])
        target_pos = self.env.get_p_body('side_object_table')

        reward = self._compute_reward(tcp_pos, obj_pos, target_pos, gripper_action)
        self.episode_steps += 1

        # Check for termination
        done = self._check_done(obj_pos, target_pos) or self.episode_steps >= self.max_episode_steps
        obs = self._get_obs()

        self.env.render()

        print(f"Step reward: {reward}, Done: {done}")
        return obs, reward, done, False, {}

    def _get_obs(self):
        rgb_img = cv2.resize(self.env.grab_rgb_depth_img()[0], (64, 64), interpolation=cv2.INTER_AREA).astype(np.uint8)
        if self.add_proprioception:
            joint_positions = self.env.data.ctrl[self.env.idxs_forward]
            return {'image': rgb_img, 'proprioception': joint_positions}
        return rgb_img

    def _compute_reward(self, tcp_pos, obj_pos, target_pos, gripper_action):
        r_approach = -np.linalg.norm(tcp_pos - obj_pos) * 0.1
        r_grasp = 10.0 if np.linalg.norm(tcp_pos - obj_pos) < 0.05 and gripper_action < 0.5 else 0.0
        r_transport = -np.linalg.norm(obj_pos - target_pos) * 0.1 if gripper_action < 0.5 else -0.1
        r_place = 50.0 if np.linalg.norm(obj_pos - target_pos) < 0.05 and gripper_action > 0.5 else 0.0
        return r_approach + r_grasp + r_transport + r_place

    def _check_done(self, obj_pos, target_pos):
        dist = np.linalg.norm(obj_pos - target_pos)
        print(f"Object-Target Distance: {dist}")
        return dist < 0.05

def make_env(rank, add_proprioception=True, use_viewer=False):
    def _init():
        env = PickPlaceEnvImage(add_proprioception=add_proprioception, use_viewer=use_viewer)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    num_envs = 1  # Use one environment for debugging
    env = DummyVecEnv([make_env(i, use_viewer=False) for i in range(num_envs)])
    env = VecTransposeImage(env)

    eval_env = DummyVecEnv([make_env(0, use_viewer=False)])
    eval_env = VecTransposeImage(eval_env)

    eval_callback = EvalCallback(eval_env, best_model_save_path="./best_model", log_path="./logs", eval_freq=1000,
                                  deterministic=True, render=False, callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1))
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./checkpoints/", name_prefix="ppo_pickplace")

    policy_kwargs = {
        "net_arch": dict(pi=[512, 512], vf=[512, 512]),
        "share_features_extractor": False,
        "features_extractor_kwargs": {"cnn_output_dim": 512}
    }

    model = PPO(
        "MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_pickplace_vision/",
        learning_rate=5e-5, ent_coef=0.1, n_steps=512, batch_size=128, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, max_grad_norm=0.5, vf_coef=0.5, policy_kwargs=policy_kwargs
    )

    model.learn(total_timesteps=100000, callback=[eval_callback, checkpoint_callback], progress_bar=True)
    model.save("ppo_pick_and_place_final")

    # Evaluation
    env = PickPlaceEnvImage(add_proprioception=True, use_viewer=True)
    model = PPO.load("./best_model/best_model")
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
