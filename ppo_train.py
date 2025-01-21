import gymnasium as gym
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from mujoco_parser import MuJoCoParserClass
from util import sample_xyzs

class PickPlaceEnvImage(gym.Env):
    def __init__(self, add_proprioception=True):
        super(PickPlaceEnvImage, self).__init__()
        xml_path = 'assets/ur5e/scene_ur5e_rg2_d435i_obj.xml'
        self.env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=xml_path, VERBOSE=True)
        self.obj_names = [body_name for body_name in self.env.body_names if body_name.startswith("obj_")]
        self.env.model.body('base_table').pos = np.array([0,0,0])
        self.env.model.body('front_object_table').pos = np.array([1.05,0,0])
        self.env.model.body('side_object_table').pos = np.array([0,-0.85,0])
        self.env.model.body('base').pos = np.array([0.18,0,0.8])
        self._randomize_scene()

        self.env.reset()
        q_init_upright = np.array([0,-np.pi/2,0,0,np.pi/2,0])
        self.env.forward(q=q_init_upright, joint_idxs=self.env.idxs_forward)
        self.env.init_viewer(viewer_title='UR5e Simulation', viewer_width=1200, viewer_height=800, viewer_hide_menus=True)
        self.env.update_viewer(azimuth=66, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71])
        self.add_proprioception = add_proprioception
        
        self.image_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.proprio_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        if self.add_proprioception:
            self.observation_space = gym.spaces.Dict({
                'image': self.image_space,
                'proprioception': self.proprio_space
            })
        else:
            self.observation_space = self.image_space

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.episode_steps = 0
        self.max_episode_steps = 1000
        self.cumulative_reward = 0
        self.successful_grasps = 0
        self.successful_placements = 0

    def reset(self, seed=None, options=None):
        self.env.reset()
        self.env.forward()
        self.episode_steps = 0
        self.cumulative_reward = 0
        return self._get_obs(), {}

    def _randomize_scene(self):
        n_obj = len(self.obj_names)
        xyzs = sample_xyzs(n_sample=n_obj, x_range=[0.75, 1.25], y_range=[-0.38, 0.38], z_range=[0.81, 0.81], min_dist=0.2)
        for obj_idx, obj_name in enumerate(self.obj_names):
            jntadr = self.env.model.body(obj_name).jntadr[0]
            self.env.model.joint(jntadr).qpos0[:3] = xyzs[obj_idx]
    
    def step(self, action):
        scaled_action = np.clip(action, -1.0, 1.0)
        joint_action = scaled_action[:6] * 0.5
        gripper_action = scaled_action[6]
        self.env.step(ctrl=joint_action, ctrl_idxs=self.env.idxs_forward)
        self.env.step(ctrl=gripper_action, ctrl_idxs=6)
        
        obs = self._get_obs()
        reward = self._compute_reward()
        self.cumulative_reward += reward
        self.episode_steps += 1
        terminated = self._check_done()
        truncated = self.episode_steps >= self.max_episode_steps
        
        info = {
            'episode_steps': self.episode_steps,
            'cumulative_reward': self.cumulative_reward,
            'successful_grasps': self.successful_grasps,
            'successful_placements': self.successful_placements
        }
        self.env.render()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        rgb_img = self._get_image_obs()
        
        if self.add_proprioception:
            joint_positions = self.env.data.ctrl[self.env.idxs_forward]
            gripper_state = np.array([self.env.data.ctrl[6]])
            tcp_pos = self.env.get_p_body("tcp_link")
            proprio_obs = np.concatenate([joint_positions, gripper_state, tcp_pos])
            return {'image': rgb_img, 'proprioception': proprio_obs}
        else:
            return rgb_img

    def _get_image_obs(self):
        rgb_img, _ = self.env.grab_rgb_depth_img()
        rgb_img = cv2.resize(rgb_img, (64, 64), interpolation=cv2.INTER_LINEAR)
        return rgb_img.astype(np.uint8)

    def _compute_reward(self):
        gripper_pos = self.env.get_p_body("tcp_link")
        target_pos = self.env.model.body("front_object_table").pos
        object_positions = np.array([self.env.get_p_body(obj_name) for obj_name in self.obj_names])
        
        gripper_obj_distances = np.linalg.norm(object_positions - gripper_pos, axis=1)
        obj_target_distances = np.linalg.norm(object_positions - target_pos, axis=1)
        
        min_gripper_obj_dist = np.min(gripper_obj_distances)
        min_obj_target_dist = np.min(obj_target_distances)
        
        reach_reward = -min_gripper_obj_dist * 0.1
        lift_reward = 0
        place_reward = -min_obj_target_dist * 0.1
        
        gripper_width = self.env.data.ctrl[6]
        has_grasped = (min_gripper_obj_dist < 0.05) and (gripper_width < 0.03)
        
        if has_grasped:
            self.successful_grasps += 1
            lift_reward = 2.0
            object_height = np.max(object_positions[:, 2])
            if object_height > 0.15:
                lift_reward += (object_height - 0.15) * 5.0
        
        if min_obj_target_dist < 0.1:
            self.successful_placements += 1
            place_reward = 10.0
        
        reward = reach_reward + lift_reward + place_reward
        action_penalty = -0.01 * np.sum(np.abs(self.env.data.qvel[self.env.idxs_jacobian]))
        reward += action_penalty
        
        return reward

    def _check_done(self):
        target_pos = self.env.model.body("front_object_table").pos
        object_positions = np.array([self.env.get_p_body(obj_name) for obj_name in self.obj_names])
        distances = np.linalg.norm(object_positions - target_pos, axis=1)
        
        success = np.any(distances < 0.1)
        timeout = self.episode_steps >= self.max_episode_steps
        
        return success or timeout

def make_env(rank, add_proprioception=True):
    def _init():
        env = PickPlaceEnvImage(add_proprioception=add_proprioception)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    num_envs = 4
    env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    env = VecTransposeImage(env)
    
    eval_env = DummyVecEnv([make_env(0)])
    eval_env = VecTransposeImage(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="ppo_pickplace"
    )
    
    policy_kwargs = {
        "net_arch": dict(
            pi=[256, 128],
            vf=[256, 128]
        ),
        "share_features_extractor": False,
        "features_extractor_kwargs": {
            "cnn_output_dim": 256
        }
    }
    
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_pickplace_vision/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        max_grad_norm=0.5,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs
    )
    
    model.learn(
        total_timesteps=1_000_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    model.save("ppo_pick_and_place_vision_final")
