import jax
import jax.numpy as jnp
from jax import jit
import jaxlie

def analytic_ik(O_T_EE, q7):
    """
    Compute inverse kinematics given the 4x4 EE transform O_T_EE (the imaginary grasp point of the gripper
    at urdf link "panda_grasptarget") and fixed q7 joint (centered at 0 rad). Note that O_T_EE's z axis
    should be facing down if you want your gripper to grasp things under it

    With thanks to:
    
    @InProceedings{HeLiu2021,
        author    = {Yanhao He and Steven Liu},
        booktitle = {2021 9th International Conference on Control, Mechatronics and Automation (ICCMA2021)},
        title     = {Analytical Inverse Kinematics for {F}ranka {E}mika {P}anda -- a Geometrical Solver for 7-{DOF} Manipulators with Unconventional Design},
        year      = {2021},
        month     = nov,
        publisher = {{IEEE}},
        doi       = {10.1109/ICCMA54375.2021.9646185},
    }
    """
    invalid_solution = jnp.full((4,), False)
    q_all = jnp.full((4, 7), jnp.nan)
    
    assert O_T_EE.shape == (4, 4)

    d1 = 0.3330
    d3 = 0.3160
    d5 = 0.3840
    d7e = 0.2104
    a4 = 0.0825
    a7 = 0.0880

    LL24 = a4**2 + d3**2
    LL46 = a4**2 + d5**2
    L24 = jnp.sqrt(LL24)
    L46 = jnp.sqrt(LL46)

    thetaH46 = jnp.arctan(d5 / a4)
    theta342 = jnp.arctan(d3 / a4)
    theta46H = jnp.arctan(a4 / d5)

    q_min = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = jnp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    
    # q_min = jnp.array([-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671])
    # q_max = jnp.array([2.9671, 1.8326, 2.9671, 0.0873, 2.9671, 3.8223, 2.9671])

    invalid_solution |= (q7 <= q_min[6]) | (q7 >= q_max[6])

    q_all = q_all.at[:, 6].set(q7)

    R_EE = O_T_EE[:3, :3]
    z_EE = O_T_EE[:3, 2]
    p_EE = O_T_EE[:3, 3]
    p_7 = p_EE - d7e * z_EE

    x_EE_6 = jnp.array([jnp.cos(q7 - jnp.pi / 4), -jnp.sin(q7 - jnp.pi / 4), 0.0])
    x_6 = R_EE @ x_EE_6
    x_6 /= jnp.linalg.norm(x_6)
    p_6 = p_7 - a7 * x_6

    p_2 = jnp.array([0.0, 0.0, d1])
    V26 = p_6 - p_2
    LL26 = jnp.dot(V26, V26)
    L26 = jnp.sqrt(LL26)
    
    invalid_solution |= (L24 + L46 < L26) | (L24 + L26 < L46) | (L26 + L46 < L24)

    theta246 = jnp.arccos((LL24 + LL46 - LL26) / (2.0 * L24 * L46))
    q4 = theta246 + thetaH46 + theta342 - 2.0 * jnp.pi

    invalid_solution |= (q4 <= q_min[3]) | (q4 >= q_max[3])

    q_all = q_all.at[:, 3].set(q4)
    
    # compute q6
    theta462 = jnp.arccos((LL26 + LL46 - LL24) / (2.0 * L26 * L46))
    theta26H = theta46H + theta462
    D26 = -L26 * jnp.cos(theta26H)

    Z_6 = jnp.cross(z_EE, x_6)
    Y_6 = jnp.cross(Z_6, x_6)
    R_6 = jnp.vstack([x_6, Y_6 / jnp.linalg.norm(Y_6), Z_6 / jnp.linalg.norm(Z_6)]).T
    V_6_62 = R_6.T @ (-V26)

    Phi6 = jnp.arctan2(V_6_62[1], V_6_62[0])
    
    # Use clip to avoid domain errors with asin
    Theta6 = jnp.arcsin(D26 / jnp.sqrt(V_6_62[0]**2 + V_6_62[1]**2))

    q6_0 = jnp.pi - Theta6 - Phi6
    q6_1 = Theta6 - Phi6
    q6 = jnp.array([q6_0, q6_1])
    
    for i in range(2):
        q6 = q6.at[i].set(jnp.where(q6[i] <= q_min[5], q6[i] + 2.0 * jnp.pi, q6[i]))
        q6 = q6.at[i].set(jnp.where(q6[i] >= q_max[5], q6[i] - 2.0 * jnp.pi, q6[i]))
        
        invalid = (q6[i] <= q_min[5]) | (q6[i] >= q_max[5])
        invalid_solution = invalid_solution.at[2*i].set(
            invalid_solution[2*i] | invalid
        )
        invalid_solution = invalid_solution.at[2*i + 1].set(
            invalid_solution[2*i + 1] | invalid
        )

        q_all = q_all.at[2 * i, 5].set(q6[i])
        q_all = q_all.at[2 * i + 1, 5].set(q6[i])
        
    invalid_solution |= ~jnp.isfinite(q_all[0, 5])
    invalid_solution |= ~jnp.isfinite(q_all[2, 5])
    
    # compute q1 & q2
    thetaP26 = 3.0 * jnp.pi / 2.0 - theta462 - theta246 - theta342
    thetaP = jnp.pi - thetaP26 - theta26H
    LP6 = L26 * jnp.sin(thetaP26) / jnp.sin(thetaP)
    
    z_5_all = jnp.empty((4, 3))
    V2P_all = jnp.empty((4, 3))
    
    for i in range(2):
        z_6_5 = jnp.array([jnp.sin(q6[i]), jnp.cos(q6[i]), 0])
        z_5 = R_6 @ z_6_5
        V2P = p_6 - LP6 * z_5 - p_2

        z_5_all = z_5_all.at[2 * i].set(z_5)
        z_5_all = z_5_all.at[2 * i + 1].set(z_5)

        V2P_all = V2P_all.at[2 * i].set(V2P)
        V2P_all = V2P_all.at[2 * i + 1].set(V2P)

        L2P = jnp.linalg.norm(V2P)
    
        # Is singular
        invalid = jnp.abs(V2P[2] / L2P) > 0.999
        invalid_solution = invalid_solution.at[2*i].set(invalid_solution[2*i] | invalid)
        invalid_solution = invalid_solution.at[2*i + 1].set(invalid_solution[2*i + 1] | invalid)

        q_all = q_all.at[2 * i, 0].set(jnp.atan2(V2P[1], V2P[0]))
        q_all = q_all.at[2*i, 1].set(jnp.arccos(V2P[2] / L2P))
        
        q_all = q_all.at[2*i+1, 0].set(jnp.where(q_all[2*i, 0] < 0,
                                                q_all[2*i, 0] + jnp.pi,
                                                q_all[2*i, 0] - jnp.pi))
        
        q_all = q_all.at[2 * i + 1, 1].set(-q_all[2 * i, 1])
   
    for i in range(4):
        invalid = (q_all[i, 0] <= q_min[0]) | (q_all[i, 0] >= q_max[0]) | \
                  (q_all[i, 1] <= q_min[1]) | (q_all[i, 1] >= q_max[1])
        
        invalid_solution = invalid_solution.at[i].set(invalid_solution[i] | invalid)

        z_3 = V2P_all[i] / jnp.linalg.norm(V2P_all[i])
        Y_3 = -jnp.cross(V26, V2P_all[i])
        y_3 = Y_3 / jnp.linalg.norm(Y_3)
        x_3 = jnp.cross(y_3, z_3)
        
        c1 = jnp.cos(q_all[i, 0])
        s1 = jnp.sin(q_all[i, 0])
        c2 = jnp.cos(q_all[i, 1])
        s2 = jnp.sin(q_all[i, 1])

        R_1 = jnp.array([[c1, -s1, 0.0],
                         [s1,  c1, 0.0],
                         [0.0, 0.0, 1.0]])
        R_1_2 = jnp.array([[ c2, -s2, 0.0],
                           [0.0, 0.0, 1.0],
                           [-s2, -c2, 0.0]])
        R_2 = R_1 @ R_1_2
        x_2_3 = R_2.T @ x_3
        q_all = q_all.at[i, 2].set(jnp.atan2(x_2_3[2], x_2_3[0]))
        
        invalid = (q_all[i, 2] <= q_min[2]) | (q_all[i, 2] >= q_max[2])
        invalid_solution = invalid_solution.at[i].set(invalid_solution[i] | invalid)
        
        # compute q5
        VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5_all[i]
        c6 = jnp.cos(q_all[i, 5])
        s6 = jnp.sin(q_all[i, 5])
        R_5_6 = jnp.array([[c6, -s6,   0.0],
                           [0.0, 0.0, -1.0],
                           [s6,  c6,   0.0]])
        R_5 = R_6 @ R_5_6.T
        V_5_H4 = R_5.T @ VH4
        
        q_all = q_all.at[i, 4].set(-jnp.arctan2(V_5_H4[1], V_5_H4[0]))
        
        invalid = (q_all[i, 4] <= q_min[4]) | (q_all[i, 4] >= q_max[4])
        invalid_solution = invalid_solution.at[i].set(invalid_solution[i] | invalid)

    return q_all, invalid_solution


def analytic_ik_case_consistent(O_T_EE, q7, q_actual):
    """
    "Case-Consistent" inverse kinematics w.r.t. End Effector Frame (using Franka Hand data)
    This finds an IK solution for the given end-effector position, choosing a solution
    that is consistent with the current joint configuration `q_actual`.
    """
    q = jnp.full(7, jnp.nan)
    invalid = False

    # constants
    d1 = 0.3330
    d3 = 0.3160
    d5 = 0.3840
    d7e = 0.2104
    a4 = 0.0825
    a7 = 0.0880

    LL24 = a4**2 + d3**2
    LL46 = a4**2 + d5**2
    L24 = jnp.sqrt(LL24)
    L46 = jnp.sqrt(LL46)
    
    thetaH46 = jnp.arctan(d5/a4)
    theta342 = jnp.arctan(d3/a4)
    theta46H = jnp.arctan(a4/d5)
    
    q_min = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = jnp.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    
    # return NAN if input q7 is out of range
    invalid |= (q7 <= q_min[6]) | (q7 >= q_max[6])
    
    q = q.at[6].set(q7)

    # FK for getting current case id
    c1_a = jnp.cos(q_actual[0]); s1_a = jnp.sin(q_actual[0])
    c2_a = jnp.cos(q_actual[1]); s2_a = jnp.sin(q_actual[1])
    c3_a = jnp.cos(q_actual[2]); s3_a = jnp.sin(q_actual[2])
    c4_a = jnp.cos(q_actual[3]); s4_a = jnp.sin(q_actual[3])
    c5_a = jnp.cos(q_actual[4]); s5_a = jnp.sin(q_actual[4])
    c6_a = jnp.cos(q_actual[5]); s6_a = jnp.sin(q_actual[5])

    As_a = jnp.zeros((7, 4, 4))
    As_a = As_a.at[0].set(jnp.array([[   c1_a, -s1_a,  0.0,  0.0],
                                     [   s1_a,  c1_a,  0.0,  0.0],
                                     [    0.0,   0.0,  1.0,   d1],
                                     [    0.0,   0.0,  0.0,  1.0]]))
    As_a = As_a.at[1].set(jnp.array([[   c2_a, -s2_a,  0.0,  0.0],
                                     [    0.0,   0.0,  1.0,  0.0],
                                     [  -s2_a, -c2_a,  0.0,  0.0],
                                     [    0.0,   0.0,  0.0,  1.0]]))
    As_a = As_a.at[2].set(jnp.array([[   c3_a, -s3_a,  0.0,  0.0],
                                     [    0.0,   0.0, -1.0,  -d3],
                                     [   s3_a,  c3_a,  0.0,  0.0],
                                     [    0.0,   0.0,  0.0,  1.0]]))
    As_a = As_a.at[3].set(jnp.array([[   c4_a, -s4_a,  0.0,   a4],
                                     [    0.0,   0.0, -1.0,  0.0],
                                     [   s4_a,  c4_a,  0.0,  0.0],
                                     [    0.0,   0.0,  0.0,  1.0]]))
    As_a = As_a.at[4].set(jnp.array([[    1.0,   0.0,  0.0,  -a4],
                                     [    0.0,   1.0,  0.0,  0.0],
                                     [    0.0,   0.0,  1.0,  0.0],
                                     [    0.0,   0.0,  0.0,  1.0]]))
    As_a = As_a.at[5].set(jnp.array([[   c5_a, -s5_a,  0.0,  0.0],
                                     [    0.0,   0.0,  1.0,   d5],
                                     [  -s5_a, -c5_a,  0.0,  0.0],
                                     [    0.0,   0.0,  0.0,  1.0]]))
    As_a = As_a.at[6].set(jnp.array([[   c6_a, -s6_a,  0.0,  0.0],
                                     [    0.0,   0.0, -1.0,  0.0],
                                     [   s6_a,  c6_a,  0.0,  0.0],
                                     [    0.0,   0.0,  0.0,  1.0]]))

    Ts_a = jnp.zeros((7, 4, 4))
    Ts_a = Ts_a.at[0].set(As_a[0])
    for j in range(1, 7):
        Ts_a = Ts_a.at[j].set(Ts_a[j - 1] @ As_a[j])

    # identify q6 case
    V62_a = Ts_a[1][:3, 3] - Ts_a[6][:3, 3]
    V6H_a = Ts_a[4][:3, 3] - Ts_a[6][:3, 3]
    Z6_a = Ts_a[6][:3, 2]
    is_case6_0 = (jnp.dot(jnp.cross(V6H_a, V62_a), Z6_a) <= 0)

    # identify q1 case
    is_case1_1 = (q_actual[1] < 0)
    
    # IK: compute p_6
    R_EE = O_T_EE[:3, :3]
    z_EE = O_T_EE[:3, 2]
    p_EE = O_T_EE[:3, 3]
    p_7 = p_EE - d7e * z_EE
    
    x_EE_6 = jnp.array([jnp.cos(q7 - jnp.pi/4), -jnp.sin(q7 - jnp.pi/4), 0.0])
    x_6 = R_EE @ x_EE_6
    x_6 /= jnp.linalg.norm(x_6)
    p_6 = p_7 - a7 * x_6
    
    # IK: compute q4
    p_2 = jnp.array([0.0, 0.0, d1])
    V26 = p_6 - p_2
    
    LL26 = jnp.dot(V26, V26)
    L26 = jnp.sqrt(LL26)
    
    invalid |= (L24 + L46 < L26) | (L24 + L26 < L46) | (L26 + L46 < L24)
    
    theta246 = jnp.arccos((LL24 + LL46 - LL26) / (2.0 * L24 * L46))
    q4 = theta246 + thetaH46 + theta342 - 2.0 * jnp.pi
    invalid |= (q4 <= q_min[3]) | (q4 >= q_max[3])
    q = q.at[3].set(q4)
    
    # IK: compute q6
    theta462 = jnp.arccos((LL26 + LL46 - LL24) / (2.0 * L26 * L46))
    theta26H = theta46H + theta462
    D26 = -L26 * jnp.cos(theta26H)
    
    Z_6 = jnp.cross(z_EE, x_6)
    Y_6 = jnp.cross(Z_6, x_6)
    R_6 = jnp.vstack([x_6, Y_6 / jnp.linalg.norm(Y_6), Z_6 / jnp.linalg.norm(Z_6)]).T
    V_6_62 = R_6.T @ (-V26)

    Phi6 = jnp.arctan2(V_6_62[1], V_6_62[0])
    Theta6 = jnp.arcsin(D26 / jnp.sqrt(V_6_62[0]**2 + V_6_62[1]**2))
    
    q6 = jnp.where(is_case6_0, jnp.pi - Theta6 - Phi6, Theta6 - Phi6)
    
    q6 = jnp.where(q6 <= q_min[5], q6 + 2.0 * jnp.pi, q6)
    q6 = jnp.where(q6 >= q_max[5], q6 - 2.0 * jnp.pi, q6)
    
    invalid |= (q6 <= q_min[5]) | (q6 >= q_max[5])
    q = q.at[5].set(q6)

    # IK: compute q1 & q2
    thetaP26 = 3.0 * jnp.pi / 2.0 - theta462 - theta246 - theta342
    thetaP = jnp.pi - thetaP26 - theta26H
    LP6 = L26 * jnp.sin(thetaP26) / jnp.sin(thetaP)
    
    z_6_5 = jnp.array([jnp.sin(q[5]), jnp.cos(q[5]), 0.0])
    z_5 = R_6 @ z_6_5
    V2P = p_6 - LP6 * z_5 - p_2
    
    L2P = jnp.linalg.norm(V2P)
    
    singular = jnp.abs(V2P[2] / L2P) > 0.999
    q0 = jnp.where(singular, q_actual[0], jnp.arctan2(V2P[1], V2P[0]))
    q1 = jnp.where(singular, 0.0, jnp.arccos(V2P[2] / L2P))

    q0_final = jnp.where(is_case1_1, jnp.where(q0 < 0.0, q0 + jnp.pi, q0 - jnp.pi), q0)
    q1_final = jnp.where(is_case1_1, -q1, q1)
    
    q = q.at[0].set(q0_final)
    q = q.at[1].set(q1_final)

    invalid |= (q[0] <= q_min[0]) | (q[0] >= q_max[0]) | (q[1] <= q_min[1]) | (q[1] >= q_max[1])
    
    # IK: compute q3
    z_3 = V2P / jnp.linalg.norm(V2P)
    Y_3 = -jnp.cross(V26, V2P)
    y_3 = Y_3 / jnp.linalg.norm(Y_3)
    x_3 = jnp.cross(y_3, z_3)
    
    c1 = jnp.cos(q[0]); s1 = jnp.sin(q[0])
    R_1 = jnp.array([[ c1, -s1, 0.0], [ s1,  c1, 0.0], [0.0, 0.0, 1.0]])
    
    c2 = jnp.cos(q[1]); s2 = jnp.sin(q[1])
    R_1_2 = jnp.array([[ c2, -s2, 0.0], [0.0, 0.0, 1.0], [-s2, -c2, 0.0]])
    
    R_2 = R_1 @ R_1_2
    x_2_3 = R_2.T @ x_3
    q2_val = jnp.arctan2(x_2_3[2], x_2_3[0])
    q = q.at[2].set(q2_val)
    
    invalid |= (q[2] <= q_min[2]) | (q[2] >= q_max[2])
    
    # IK: compute q5
    VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5
    c6 = jnp.cos(q[5]); s6 = jnp.sin(q[5])
    R_5_6 = jnp.array([[ c6, -s6, 0.0], [0.0, 0.0, -1.0], [ s6,  c6, 0.0]])
    R_5 = R_6 @ R_5_6.T
    V_5_H4 = R_5.T @ VH4
    
    q4_val = -jnp.arctan2(V_5_H4[1], V_5_H4[0])
    q = q.at[4].set(q4_val)

    invalid |= (q[4] <= q_min[4]) | (q[4] >= q_max[4])
    
    return q, invalid