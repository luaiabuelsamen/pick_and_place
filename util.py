import math,time,os
import numpy as np
import shapely as sp # handle polygon
from shapely import Polygon,LineString,Point # handle polygons
from scipy.spatial.distance import cdist


def execute_peract_action(env, continuous_trans, continuous_quat, gripper_open):
    """Execute the predicted action in MuJoCo using IK"""
    # Get current robot state
    current_q = env.get_q([0, 1, 2, 3, 4, 5])
    
    # Convert quaternion to rotation matrix
    R_trgt = quat2r(continuous_quat)
    
    # Target position
    p_trgt = continuous_trans
    
    print(f"Executing move to position: {p_trgt}, orientation: {continuous_quat}")
    print(f"Gripper state: {'Open' if gripper_open else 'Closed'}")
    
    q_ik_pregrasp = env.solve_ik(body_name='tcp_link', p_trgt=p_trgt, R_trgt=R_trgt, IK_P=True, IK_R=False, q_init=current_q, idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-2, w_weight=0.5)
    
    # Generate trajectory to pre-grasp position
    q_traj_combined = np.vstack([current_q, q_ik_pregrasp])
    _, q_traj_pregrasp = get_interp_const_vel_traj(q_traj_combined, vel=np.radians(45), HZ=env.HZ)
    
    for q in q_traj_pregrasp:
        # Append current gripper state
        gripper_val = gripper_open # Use current gripper state during approach
        q_with_gripper = np.append(q, gripper_val)
        
        # Execute step
        env.step(ctrl=q_with_gripper, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
        env.render()

# Helper function to convert quaternion to rotation matrix
def quat2r(q):
    """Convert quaternion to rotation matrix."""
    # Ensure quaternion has correct format [x, y, z, w]
    # Reorder if necessary based on your convention
    if len(q) == 4:
        x, y, z, w = q
    else:
        # Default if length doesn't match
        x, y, z, w = 0, 0, 0, 1
    
    # Compute rotation matrix
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    
    yy = y * y
    yz = y * z
    yw = y * w
    
    zz = z * z
    zw = z * w
    
    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
        [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
        [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
    ])
    
    return R

# Function to get the RPY from existing trajectory generation code
def rpy2r(rpy):
    """Convert roll-pitch-yaw to rotation matrix"""
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rpy[0]), -np.sin(rpy[0])],
        [0, np.sin(rpy[0]), np.cos(rpy[0])]
    ])
    R_y = np.array([
        [np.cos(rpy[1]), 0, np.sin(rpy[1])],
        [0, 1, 0],
        [-np.sin(rpy[1]), 0, np.cos(rpy[1])]
    ])
    R_z = np.array([
        [np.cos(rpy[2]), -np.sin(rpy[2]), 0],
        [np.sin(rpy[2]), np.cos(rpy[2]), 0],
        [0, 0, 1]
    ])
    R = R_z @ R_y @ R_x
    return R

def generate_trajectories(env, obj_names, q_init_upright, platform_xyz):
    R_trgt = rpy2r(np.radians([0,80,0]))@rpy2r(np.radians([-180,0,90]))
    pick_position = env.get_p_body(obj_names[0])
    pick_position[2] += 0.01
    pre_grasp_position = pick_position + np.array([0.0, 0.0, 0.1])
    q_ik_pregrasp = env.solve_ik(body_name='tcp_link', p_trgt=pre_grasp_position, R_trgt=R_trgt, IK_P=True, IK_R=False, q_init=np.array(q_init_upright), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-2, w_weight=0.5)
    q_ik = env.solve_ik(body_name='tcp_link', p_trgt=pick_position, R_trgt=R_trgt, IK_P=True, IK_R=False, q_init=np.array(q_ik_pregrasp), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-2, w_weight=0.5)
    post_pick_position = pick_position + np.array([0.0, 0.0, 0.1])
    q_ik_postpick = env.solve_ik(body_name='tcp_link', p_trgt=post_pick_position, R_trgt=R_trgt, IK_P=True, IK_R=False, q_init=np.array(q_ik), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-2, w_weight=0.5)

    q_traj_combined = np.vstack([q_init_upright, q_ik_pregrasp, q_ik])
    times, q_traj = get_interp_const_vel_traj(q_traj_combined, vel=np.radians(90), HZ=env.HZ)

    place_position = platform_xyz + np.array([0.0, 0.0, 0.1])
    q_ik_place = env.solve_ik(body_name='tcp_link', p_trgt=place_position, R_trgt=R_trgt, IK_P=True, IK_R=False, q_init=np.array(q_traj[-1, :]), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-2, w_weight=0.5)
    pre_place_position = place_position + np.array([0.0, 0.0, 0.1])
    q_ik_preplace = env.solve_ik(body_name='tcp_link', p_trgt=pre_place_position, R_trgt=R_trgt, IK_P=True, IK_R=False, q_init=np.array(q_ik_place), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-2, w_weight=0.5)

    q_traj_place_combined = np.vstack([q_ik_postpick, q_ik_preplace, q_ik_place])
    times_place, q_traj_place = get_interp_const_vel_traj(q_traj_place_combined, vel=np.radians(90), HZ=env.HZ)

    gripper_closed_steps_1 = np.tile(np.append(q_traj[-1], 0.5), (100, 1))
    gripper_closed_steps_2 = np.tile(np.append(q_traj_place[-1], 1), (100, 1))

    q_traj_combined = np.vstack([
        np.hstack([q_traj, np.ones((q_traj.shape[0], 1))]),
        gripper_closed_steps_1,
        np.hstack([q_traj_place, np.ones((q_traj_place.shape[0], 1))*0.5]),
        gripper_closed_steps_2,
    ])

    return q_traj_combined

def rot_mtx(deg):
    """
        2 x 2 rotation matrix
    """
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def pr2t(p,R):
    """ 
        Convert pose to transformation matrix 
    """
    p0 = p.ravel() # flatten
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def t2pr(T):
    """
        T to p and R
    """   
    p = T[:3,3]
    R = T[:3,:3]
    return p,R

def t2p(T):
    """
        T to p 
    """   
    p = T[:3,3]
    return p

def t2r(T):
    """
        T to R
    """   
    R = T[:3,:3]
    return R    

def rpy2r(rpy_rad):
    """
        roll,pitch,yaw in radian to R
    """
    roll  = rpy_rad[0]
    pitch = rpy_rad[1]
    yaw   = rpy_rad[2]
    Cphi  = np.math.cos(roll)
    Sphi  = np.math.sin(roll)
    Cthe  = np.math.cos(pitch)
    Sthe  = np.math.sin(pitch)
    Cpsi  = np.math.cos(yaw)
    Spsi  = np.math.sin(yaw)
    R     = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert R.shape == (3, 3)
    return R

def r2rpy(R,unit='rad'):
    """
        Rotation matrix to roll,pitch,yaw in radian
    """
    roll  = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], (math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = math.atan2(R[1, 0], R[0, 0])
    if unit == 'rad':
        out = np.array([roll, pitch, yaw])
    elif unit == 'deg':
        out = np.array([roll, pitch, yaw])*180/np.pi
    else:
        out = None
        raise Exception("[r2rpy] Unknown unit:[%s]"%(unit))
    return out    

def r2w(R):
    """
        R to \omega
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

def r2quat(R):
    """ 
        Convert Rotation Matrix to Quaternion.  See rotation.py for notes 
        (https://gist.github.com/machinaut/dab261b78ac19641e91c6490fb9faa96)
    """
    R = np.asarray(R, dtype=np.float64)
    Qxx, Qyx, Qzx = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    Qxy, Qyy, Qzy = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    Qxz, Qyz, Qzz = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q

def skew(x):
    """ 
        Get a skew-symmetric matrix
    """
    x_hat = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    return x_hat

def rodrigues(a=np.array([1,0,0]),q_rad=0.0):
    """
        Compute the rotation matrix from an angular velocity vector
    """
    a_norm = np.linalg.norm(a)
    if abs(a_norm-1) > 1e-6:
        print ("[rodrigues] norm of a should be 1.0 not [%.2e]."%(a_norm))
        return np.eye(3)
    
    a = a / a_norm
    q_rad = q_rad * a_norm
    a_hat = skew(a)
    
    R = np.eye(3) + a_hat*np.sin(q_rad) + a_hat@a_hat*(1-np.cos(q_rad))
    return R
    
def np_uv(vec):
    """
        Get unit vector
    """
    x = np.array(vec)
    return x/np.linalg.norm(x)

def get_rotation_matrix_from_two_points(p_fr,p_to):
    p_a  = np.copy(np.array([0,0,1]))
    if np.linalg.norm(p_to-p_fr) < 1e-8: # if two points are too close
        return np.eye(3)
    p_b  = (p_to-p_fr)/np.linalg.norm(p_to-p_fr)
    v    = np.cross(p_a,p_b)
    S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    if np.linalg.norm(v) == 0:
        R = np.eye(3,3)
    else:
        R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))
    return R
    

def trim_scale(x,th):
    """
        Trim scale
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def soft_squash(x,x_min=-1,x_max=+1,margin=0.1):
    """
        Soft squashing numpy array
    """
    def th(z,m=0.0):
        # thresholding function 
        return (m)*(np.exp(2/m*z)-1)/(np.exp(2/m*z)+1)
    x_in = np.copy(x)
    idxs_upper = np.where(x_in>(x_max-margin))
    x_in[idxs_upper] = th(x_in[idxs_upper]-(x_max-margin),m=margin) + (x_max-margin)
    idxs_lower = np.where(x_in<(x_min+margin))
    x_in[idxs_lower] = th(x_in[idxs_lower]-(x_min+margin),m=margin) + (x_min+margin)
    return x_in    

def soft_squash_multidim(
    x      = np.random.randn(100,5),
    x_min  = -np.ones(5),
    x_max  = np.ones(5),
    margin = 0.1):
    """
        Multi-dim version of 'soft_squash' function
    """
    x_squash = np.copy(x)
    dim      = x.shape[1]
    for d_idx in range(dim):
        x_squash[:,d_idx] = soft_squash(
            x=x[:,d_idx],x_min=x_min[d_idx],x_max=x_max[d_idx],margin=margin)
    return x_squash 

def kernel_se(X1,X2,hyp={'g':1,'l':1}):
    """
        Squared exponential (SE) kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    return K

def kernel_levse(X1,X2,L1,L2,hyp={'g':1,'l':1}):
    """
        Leveraged SE kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    L = np.cos(np.pi/2.0*cdist(L1,L2,'cityblock'))
    return np.multiply(K,L)

def is_point_in_polygon(point,polygon):
    """
        Is the point inside the polygon
    """
    if isinstance(point,np.ndarray):
        point_check = Point(point)
    else:
        point_check = point
    return sp.contains(polygon,point_check)

def is_point_feasible(point,obs_list):
    """
        Is the point feasible w.r.t. obstacle list
    """
    result = is_point_in_polygon(point,obs_list) # is the point inside each obstacle?
    if sum(result) == 0:
        return True
    else:
        return False

def is_point_to_point_connectable(point1,point2,obs_list):
    """
        Is the line connecting two points connectable
    """
    result = sp.intersects(LineString([point1,point2]),obs_list)
    if sum(result) == 0:
        return True
    else:
        return False
    
class TicTocClass(object):
    """
        Tic toc
    """
    def __init__(self,name='tictoc',print_every=1):
        """
            Initialize
        """
        self.name        = name
        self.time_start  = time.time()
        self.time_end    = time.time()
        self.print_every = print_every

    def tic(self):
        """
            Tic
        """
        self.time_start = time.time()

    def toc(self,str=None,cnt=0,VERBOSE=True):
        """
            Toc
        """
        self.time_end = time.time()
        self.time_elapsed = self.time_end - self.time_start
        if VERBOSE:
            if self.time_elapsed <1.0:
                time_show = self.time_elapsed*1000.0
                time_unit = 'ms'
            elif self.time_elapsed <60.0:
                time_show = self.time_elapsed
                time_unit = 's'
            else:
                time_show = self.time_elapsed/60.0
                time_unit = 'min'
            if (cnt % self.print_every) == 0:
                if str is None:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (self.name,time_show,time_unit))
                else:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (str,time_show,time_unit))

def get_interp_const_vel_traj(traj_anchor,vel=1.0,HZ=100,ord=np.inf):
    """
        Get linearly interpolated constant velocity trajectory
    """
    L = traj_anchor.shape[0]
    D = traj_anchor.shape[1]
    dists = np.zeros(L)
    for tick in range(L):
        if tick > 0:
            p_prev,p_curr = traj_anchor[tick-1,:],traj_anchor[tick,:]
            dists[tick] = np.linalg.norm(p_prev-p_curr,ord=ord)
    times_anchor = np.cumsum(dists/vel) # [L]
    L_interp = int(times_anchor[-1]*HZ)
    times_interp = np.linspace(0,times_anchor[-1],L_interp) # [L_interp]
    traj_interp = np.zeros((L_interp,D)) # [L_interp x D]
    for d_idx in range(D):
        traj_interp[:,d_idx] = np.interp(times_interp,times_anchor,traj_anchor[:,d_idx])
    return times_interp,traj_interp

def meters2xyz(depth_img,cam_matrix):
    """
        Scaled depth image to pointcloud
    """
    fx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    fy = cam_matrix[1][1]
    cy = cam_matrix[1][2]
    
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    indices = np.indices((height, width),dtype=np.float32).transpose(1,2,0)
    
    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    
    # Order of y_ e is reversed !
    xyz_img = np.stack([z_e, -x_e, -y_e], axis=-1) # [H x W x 3] 
    return xyz_img # [H x W x 3]

def compute_view_params(camera_pos,target_pos,up_vector=np.array([0,0,1])):
    """Compute azimuth, distance, elevation, and lookat for a viewer given camera pose in 3D space.

    Args:
        camera_pos (np.ndarray): 3D array of camera position.
        target_pos (np.ndarray): 3D array of target position.
        up_vector (np.ndarray): 3D array of up vector.

    Returns:
        tuple: Tuple containing azimuth, distance, elevation, and lookat values.
    """
    # Compute camera-to-target vector and distance
    cam_to_target = target_pos - camera_pos
    distance = np.linalg.norm(cam_to_target)

    # Compute azimuth and elevation
    azimuth = np.arctan2(cam_to_target[1], cam_to_target[0])
    azimuth = np.rad2deg(azimuth) # [deg]
    elevation = np.arcsin(cam_to_target[2] / distance)
    elevation = np.rad2deg(elevation) # [deg]

    # Compute lookat point
    lookat = target_pos

    # Compute camera orientation matrix
    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    yaxis = np.cross(zaxis, xaxis)
    cam_orient = np.array([xaxis, yaxis, zaxis])

    # Return computed values
    return azimuth, distance, elevation, lookat

def sample_xyzs(n_sample,x_range=[0,1],y_range=[0,1],z_range=[0,1],min_dist=0.1):
    """
        Sample a point in three dimensional space with the minimum distance between points
    """
    xyzs = np.zeros((n_sample,3))
    for p_idx in range(n_sample):
        while True:
            x_rand = np.random.uniform(low=x_range[0],high=x_range[1])
            y_rand = np.random.uniform(low=y_range[0],high=y_range[1])
            z_rand = np.random.uniform(low=z_range[0],high=z_range[1])
            xyz = np.array([x_rand,y_rand,z_rand])
            if p_idx == 0: break
            devc = cdist(xyz.reshape((-1,3)),xyzs[:p_idx,:].reshape((-1,3)),'euclidean')
            if devc.min() > min_dist: break # minimum distance between objects
        xyzs[p_idx,:] = xyz
    return xyzs

def create_folder_if_not_exists(file_path):
    """ 
        Create folder if not exist
    """
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print ("[%s] created."%(folder_path))
        
def softmax(x):
    # Subtract max(x) to compute the softmax in a numerically stable way
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_grasp_pose_primitive(env, obj_name, grasp_pose=None, dist_orientation="geodesic"):
    tcp_position = env.get_p_body('tcp_link')
    tcp_orientation = rpy2r(np.radians([0,0,0])) @ rpy2r(np.radians([-180,0,90]))
    grasp_obj_position = env.get_p_body(obj_name)
    grasp_obj_orientation = env.get_R_body(obj_name)
    if grasp_pose == "upright":
        grasp_obj_position[2] += 0.08
        grasp_obj_orientation = grasp_obj_orientation @ rpy2r(np.radians([-90,0,90]))
    elif grasp_pose == "right":
        grasp_obj_position[1] += 0.03
        grasp_obj_position[2] += 0.10
        grasp_obj_orientation = grasp_obj_orientation @ rpy2r(np.radians([-180,0,180]))
    elif grasp_pose == "left":
        grasp_obj_position[1] += 0.03
        grasp_obj_position[2] += 0.07
        grasp_obj_orientation = grasp_obj_orientation @ rpy2r(np.radians([-180,0,0]))
    elif grasp_pose == "forward":
        grasp_obj_position[0] += 0.015
        grasp_obj_position[2] += 0.05
        grasp_obj_orientation = grasp_obj_orientation @ rpy2r(np.radians([-180,0,90]))
    elif grasp_pose == "side":
        rand_position = np.random.uniform(-0.1, 0.1)
        rand_orientation = (rand_position + 0.10) * 180 / 0.2
        grasp_obj_position[1] -= rand_position
        grasp_obj_position[2] += 0.07
        grasp_obj_orientation = grasp_obj_orientation @ rpy2r(np.radians([-180,0,rand_orientation]))
    else:   # Randomly sample grasp pose based on distance [Euclidean + Orientation]
        grasp_pose_primitive = ["upright", "right", "left", "side", "forward"]
        grasp_obj_positions = []
        grasp_obj_orientations = []
        grasp_orientation_dists = []
        for grasp_pose_prim in grasp_pose_primitive:
            grasp_obj_pose = get_grasp_pose_primitive(obj_name, grasp_pose_prim)
            grasp_obj_positions.append(grasp_obj_pose[:3, 3])
            grasp_obj_orientations.append(grasp_obj_pose[:3, :3])
        grasp_dist = grasp_obj_positions - tcp_position
        # Calculate distances between orientations
        for grasp_obj_orientation_ in grasp_obj_orientations:
            if dist_orientation == "geodesic":
                trace_product = np.trace(np.dot(tcp_orientation.T, grasp_obj_orientation_))
                grasp_orientation_dist = np.arccos((trace_product - 1) / 2)
            elif dist_orientation == "frobenius":
                grasp_orientation_dist = np.linalg.norm(tcp_orientation - grasp_obj_orientation_, 'fro')
            grasp_orientation_dists.append(grasp_orientation_dist)
        grasp_orientation_dists = np.array(grasp_orientation_dists)
        grasp_dist = np.linalg.norm(grasp_dist, axis=1)
        grasp_weight = 1 / (grasp_dist + grasp_orientation_dists)
        grasp_pose = np.random.choice(grasp_pose_primitive, p=grasp_weight / np.sum(grasp_weight))
        print(f"grasp_pose: {grasp_pose}, grasp_weight: {grasp_weight}")
        grasp_obj_pose = get_grasp_pose_primitive(obj_name, grasp_pose)
        return grasp_obj_pose

    grasp_obj_pose = pr2t(grasp_obj_position, grasp_obj_orientation)
    print(grasp_obj_orientation, grasp_obj_position)
    return grasp_obj_pose
