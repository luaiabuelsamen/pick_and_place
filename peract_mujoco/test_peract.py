import sys
import os
import numpy as np
import torch
import time
from PIL import Image
import clip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/home/jetson3/luai/peract/peract_colab')
sys.path.append('/home/jetson3/luai/peract')

from mujoco_parser import MuJoCoParserClass, init_env
from util import execute_peract_action
from arm.utils import stack_on_channel, discrete_euler_to_quaternion
from arm.optim.lamb import Lamb

CAMERAS = ['wrist', 'front', 'left_shoulder', 'right_shoulder']
IMAGE_SIZE = 128
LOW_DIM_SIZE = 4
VOXEL_SIZES = [100]
NUM_LATENTS = 512
SCENE_BOUNDS = [-5, -5, -5, 5, 5, 5]
ROTATION_RESOLUTION = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0

def process_rgb_image(rgb_img):
    if not isinstance(rgb_img, Image.Image):
        rgb_img = Image.fromarray(rgb_img)
    
    rgb_img = rgb_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_array = np.array(rgb_img)
    
    if len(rgb_array.shape) == 3 and rgb_array.shape[2] >= 3:
        rgb_array = np.transpose(rgb_array[:, :, :3], (2, 0, 1))
    else:
        if len(rgb_array.shape) == 2:
            rgb_array = np.repeat(rgb_array[:, :, np.newaxis], 3, axis=2)
        rgb_array = np.transpose(rgb_array, (2, 0, 1))
    
    return rgb_array

def process_depth_image(depth_img, near=0.01, far=10.0):
    if isinstance(depth_img, Image.Image):
        depth_img = np.array(depth_img)
    
    if len(depth_img.shape) == 3:
        depth_img_2d = depth_img[:, :, 0]
    else:
        depth_img_2d = depth_img
    
    if depth_img_2d.shape[0] != IMAGE_SIZE or depth_img_2d.shape[1] != IMAGE_SIZE:
        depth_pil = Image.fromarray(depth_img_2d)
        depth_pil = depth_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        depth_img_2d = np.array(depth_pil)
    
    if np.max(depth_img_2d) > 1.0:
        depth_img_2d = depth_img_2d / np.max(depth_img_2d)
    
    metric_depth = near + depth_img_2d * (far - near)
    depth_array = metric_depth.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
    
    return depth_array

def create_point_cloud_from_depth(depth_array, camera_name):
    if len(depth_array.shape) == 3:
        depth_2d = depth_array[0]
    else:
        depth_2d = depth_array
    
    height, width = depth_2d.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    cx, cy = width / 2, height / 2
    fx, fy = width, height
    
    x = (u - cx) * depth_2d / fx
    y = (v - cy) * depth_2d / fy
    z = depth_2d
    
    point_cloud = np.stack([x, y, z], axis=0).astype(np.float32)
    return point_cloud

def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).type(clip_model.dtype)
    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
    return x, emb

def load_peract_model(checkpoint_path):
    print(f"Loading PerAct model from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model_config = checkpoint['model_config']
        training_constants = checkpoint['training_constants']
        perceiver_config = checkpoint['perceiver_config']
        
        from peract_mujoco.peract_class import PerceiverIO
        from peract_mujoco.peract_class import PerceiverActorAgent
        
        perceiver_encoder = PerceiverIO(
            depth=perceiver_config['depth'],
            iterations=perceiver_config['iterations'],
            voxel_size=perceiver_config['voxel_size'],
            initial_dim=perceiver_config['initial_dim'],
            low_dim_size=perceiver_config['low_dim_size'],
            num_latents=perceiver_config['num_latents'],
            latent_dim=perceiver_config['latent_dim'],
            num_rotation_classes=perceiver_config['num_rotation_classes'],
            cross_heads=perceiver_config['cross_heads'],
            latent_heads=perceiver_config['latent_heads'],
            cross_dim_head=perceiver_config['cross_dim_head'],
            latent_dim_head=perceiver_config['latent_dim_head'],
            activation=perceiver_config['activation'],
        )
        
        agent = PerceiverActorAgent(
            coordinate_bounds=model_config['coordinate_bounds'],
            perceiver_encoder=perceiver_encoder,
            camera_names=model_config['camera_names'],
            batch_size=1,
            voxel_size=model_config['voxel_size'],
            voxel_feature_size=model_config['voxel_feature_size'],
            num_rotation_classes=model_config['num_rotation_classes'],
            rotation_resolution=model_config['rotation_resolution'],
            lr=model_config['lr'],
            image_resolution=model_config['image_resolution'],
            lambda_weight_l2=model_config['lambda_weight_l2'],
            transform_augmentation=model_config['transform_augmentation'],
            optimizer_type=model_config['optimizer_type'],
        )
        
        agent.build(training=False, device=DEVICE)
        agent._q.load_state_dict(checkpoint['q_function_state_dict'])
        agent._q.eval()
        
        print("PerAct model loaded successfully!")
        return agent, training_constants
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def _preprocess_inputs(rgb_images, depth_images):
    obs, pcds = [], []
    for i, camera in enumerate(CAMERAS):
        rgb = process_rgb_image(rgb_images[i])
        rgb = torch.from_numpy(rgb).float()
        rgb = _norm_rgb(rgb)
        
        pcd = create_point_cloud_from_depth(process_depth_image(depth_images[i]), camera)
        pcd = torch.from_numpy(pcd).float()
        
        obs.append([rgb, pcd])
        pcds.append(pcd)
    return obs, pcds

def prepare_live_observations(env, rgb_images, depth_images, robot_pos, lang_goal, clip_model):
    low_dim_state = np.zeros((LOW_DIM_SIZE,), dtype=np.float32)
    low_dim_state[0] = robot_pos[6] if len(robot_pos) > 6 else 0.0
    low_dim_state[1] = robot_pos[6] if len(robot_pos) > 6 else 0.0
    low_dim_state[2] = 1.0 if len(robot_pos) > 6 and robot_pos[6] > 0.5 else 0.0
    low_dim_state[3] = 0.0
    
    proprio = torch.from_numpy(low_dim_state).float().unsqueeze(0).to(DEVICE)
    
    obs, pcd = _preprocess_inputs(rgb_images, depth_images)
    
    obs = [[rgb.unsqueeze(0).to(DEVICE), pc.unsqueeze(0).to(DEVICE)] for rgb, pc in obs]
    pcd = [pc.unsqueeze(0).to(DEVICE) for pc in pcd]
    
    tokens = clip.tokenize([lang_goal]).to(DEVICE)
    lang_feats, lang_embs = _clip_encode_text(clip_model, tokens)
    lang_goal_embs = lang_embs[0].float().detach().unsqueeze(0).to(DEVICE)
    
    bounds = torch.tensor(SCENE_BOUNDS, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    return obs, proprio, pcd, lang_goal_embs, bounds

def run_live_peract(model_path):
    peract_agent, constants = load_peract_model(model_path)
    if peract_agent is None:
        return
    
    clip_model, preprocess = clip.load("RN50", device=DEVICE)
    
    env, obj_names, q_init_upright, platform_xyz = init_env()
    print("MuJoCo environment initialized")
    
    lang_goal = "Put the green block on the red square"
    
    try:
        iteration = 0
        while True:
            cameras_pr = {
                'wrist': env.get_pR_body(body_name='camera_center'),
                'front': env.get_pR_body(body_name='front_camera'),
                'left_shoulder': env.get_pR_body(body_name='left_shoulder_camera'),
                'right_shoulder': env.get_pR_body(body_name='right_shoulder_camera')
            }
            
            camera_targets = {camera: p + R[:,2] for camera, (p, R) in cameras_pr.items()}
            
            rgb_images = []
            depth_images = []
            
            for camera in CAMERAS:
                p_ego, R_ego = cameras_pr[camera]
                p_trgt = camera_targets[camera]
                
                fovy = 80 if camera in ['wrist', 'left_shoulder'] else 60
                
                rgb_img, depth_img, pcd, xyz_img = env.get_egocentric_rgb_depth_pcd(
                    p_ego=p_ego, p_trgt=p_trgt, rsz_rate=None, fovy=fovy, BACKUP_AND_RESTORE_VIEW=True)
                
                rgb_images.append(rgb_img)
                depth_images.append(depth_img)
            
            robot_pos = env.get_q([0, 1, 2, 3, 4, 5, 6])
            
            obs, proprio, pcd, lang_goal_embs, bounds = prepare_live_observations(
                env, rgb_images, depth_images, robot_pos, lang_goal, clip_model)
            
            print("\nRunning PerAct inference...")
            with torch.no_grad():
                q_trans, rot_grip_q, collision_q, voxel_grid = peract_agent._q(
                    obs, proprio, pcd, lang_goal_embs, bounds)
                
                coords_indicies, rot_and_grip_indicies, ignore_collision_indicies = peract_agent._q.choose_highest_action(
                    q_trans, rot_grip_q, collision_q)
                
                res = (bounds[:, 3:] - bounds[:, :3]) / peract_agent._voxel_size
                continuous_trans = bounds[:, :3] + res * coords_indicies.int() + res / 2
            
            trans_coords = coords_indicies[0].cpu().numpy()
            continuous_trans = continuous_trans[0].cpu().numpy()
            rot_and_grip = rot_and_grip_indicies[0].cpu().numpy()
            continuous_quat = discrete_euler_to_quaternion(
                rot_and_grip[:3], 
                resolution=peract_agent._rotation_resolution
            )
            gripper_open = bool(rot_and_grip[3])
            ignore_collision = bool(ignore_collision_indicies[0][0].cpu().numpy())
            
            print("\nPerAct Predictions:")
            print(f"Language goal: '{lang_goal}'")
            print(f"Translation (discrete): {trans_coords}")
            print(f"Translation (continuous): {continuous_trans}")
            print(f"Rotation (discrete euler angles): {rot_and_grip[:3]}")
            print(f"Rotation (continuous quaternion): {continuous_quat}")
            print(f"Gripper open: {gripper_open}")
            print(f"Ignore collision: {ignore_collision}")
            
            iteration += 1
            
            execute_peract_action(
                env, 
                continuous_trans, 
                continuous_quat, 
                gripper_open
            )
            env.render()
    
    except KeyboardInterrupt:
        print("\nInference interrupted. Closing environment...")
    
    finally:
        env.close_viewer()
        print("Environment closed")

if __name__ == "__main__":
    model_path = '/home/jetson3/luai/peract/saved_models/peract_model_20250529_100859_checkpoint.pth'
    run_live_peract(model_path)