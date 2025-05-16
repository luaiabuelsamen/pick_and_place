import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from mujoco_parser import MuJoCoParserClass, init_env

def check_ik_fk_consistency():
    """
    Simple script to check IK/FK consistency using known configurations:
    1. Initialize robot to all zeros
    2. Get Cartesian position at zero
    3. Move to q_init_upright configuration
    4. Get Cartesian position at upright
    5. Move back to zero
    6. Use IK to move to the upright Cartesian position
    7. Verify joint angles match original q_init_upright
    """
    # Initialize environment
    env, obj_names, q_init_upright, platform_xyz = init_env()
    
    print("==== IK/FK Consistency Check ====")
    
    # Step 1: Reset robot to all zeros
    print("\n1. Resetting robot to all zeros...")
    zero_position = np.zeros(len(env.idxs_forward))
    env.step(zero_position, env.idxs_forward)
    
    # Get position after reset
    p_zero, R_zero = env.get_pR_body(body_name='tcp_link')
    print(f"   Position at zero: {p_zero}")
    print(f"   Joint angles at zero: {zero_position}")
    
    # Step 2: Move to q_init_upright
    print("\n2. Moving to q_init_upright configuration...")
    # Make sure q_init_upright has correct length
    q_init_upright_trunc = q_init_upright[:len(env.idxs_forward)]
    env.step(q_init_upright_trunc, env.idxs_forward)
    
    # Get position at q_init_upright
    p_upright, R_upright = env.get_pR_body(body_name='tcp_link')
    print(f"   Position at upright: {p_upright}")
    print(f"   Joint angles at upright: {q_init_upright_trunc}")
    
    # Step 3: Move back to zero
    print("\n3. Moving back to zero position...")
    env.step(zero_position, env.idxs_forward)
    
    # Step 4: Solve IK to move to upright position
    print("\n4. Solving IK to move to the upright position...")
    ik_solution = env.solve_ik(body_name='tcp_link', p_trgt=p_upright, R_trgt=R_upright, IK_P=True, IK_R=True, q_init=zero_position, idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-5, w_weight=0.5)
    
    print(f"   IK solution joint angles: {ik_solution}")
    
    # Step 5: Set the robot to the IK solution
    print("\n5. Setting robot to IK solution...")
    env.step(ik_solution, env.idxs_forward)
    
    # Step 6: Get the actual position using FK
    print("\n6. Checking resulting position using FK...")
    p_actual, R_actual = env.get_pR_body(body_name='tcp_link')
    
    print(f"   Target position: {p_upright}")
    print(f"   Actual position after IK: {p_actual}")
    
    # Step 7: Calculate and display the position error
    position_error = np.linalg.norm(p_actual - p_upright)
    print(f"\n7. Position error: {position_error*1000:.3f} mm")
    
    # Step 8: Calculate and display the joint angle error
    joint_angle_error = np.linalg.norm(ik_solution - q_init_upright_trunc)
    print(f"8. Joint angle error: {joint_angle_error:.6f} rad ({np.degrees(joint_angle_error):.3f}°)")
    
    # Compare joint by joint
    print("\n   Joint-by-joint comparison:")
    for i in range(len(ik_solution)):
        diff = ik_solution[i] - q_init_upright_trunc[i]
        print(f"   Joint {i}: Original = {np.degrees(q_init_upright_trunc[i]):.2f}°, IK = {np.degrees(ik_solution[i]):.2f}°, Diff = {np.degrees(diff):.2f}°")
    
    # Overall check
    if position_error < 0.005 and joint_angle_error < 0.05:  # 5mm and ~3° thresholds
        print("\nConsistency Check: PASSED ✓")
        print("Your IK and FK are consistent within acceptable tolerance.")
    else:
        print("\nConsistency Check: FAILED ✗")
        if position_error >= 0.005:
            print("Position error exceeds threshold.")
        if joint_angle_error >= 0.05:
            print("Joint angle error exceeds threshold.")
        print("Possible causes:")
        print("- Redundancy in the robot kinematics (multiple joint configs possible)")
        print("- IK solver finding different solution than original configuration")
        print("- Joint limits or other constraints affecting the solution")
    
    # # Show visualization
    # print("\nDisplaying the robot in IK solution configuration...")
    # for _ in range(50):
    #     env.render()
        
    # # Move to the original upright configuration for comparison
    # print("\nMoving to original q_init_upright for comparison...")
    # env.step(q_init_upright_trunc, env.idxs_forward)
    
    # # Show visualization of original upright
    # for _ in range(50):
    #     env.render()
    
    env.close_viewer()
    return position_error, joint_angle_error

if __name__ == "__main__":
    check_ik_fk_consistency()