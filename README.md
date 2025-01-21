This repository implements a PPO RL agent for a simulated UR5e robotic arm with a gripper to perform pick-and-place tasks. The simulation leverages [MuJoCo](https://mujoco.org/) for physics and rendering, with training designed to process image and proprioceptive inputs.

---

#### Features
- **Custom Pick-and-Place Environment:** 
  - Designed for UR5e with an RG2 gripper.
  - Combines visual and proprioceptive inputs for decision-making.
- **Stable-Baselines3 Integration:**
  - Uses `PPO` from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
- **Callbacks:**
  - Evaluation, checkpoint saving, and stopping based on reward threshold.
- **Observation Spaces:**
  - RGB images (64x64x3).
  - Proprioceptive state: joint positions, velocities, gripper state, and TCP position.

---

#### Credits
This project was inspired and partially adapted from [joonhyung-lee/mujoco-robotics-usage](https://github.com/joonhyung-lee/mujoco-robotics-usage).

--- 

#### Future Work
- Add domain randomization for robustness.
- Use SAC with labeled pick and place tasks