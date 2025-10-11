import jax
import jax.numpy as jnp
import jaxlie
from kinematics.ik import analytic_ik, analytic_ik_case_consistent

def rgb_to_hex(r, g, b):
    """Converts RGB color values [0 - 255] to a hexadecimal value 0x."""
    return (r << 16) | (g << 8) | b

def white_hex():
    """Returns the hexadecimal color code for white."""
    return 

def purple_hex():
    """Returns the hexadecimal color code for purple."""
    return rgb_to_hex(128, 0, 128)

def gray_hex():
    """Returns the hexadecimal color code for gray."""
    # return rgb_to_hex(255, 255, 255)
    return rgb_to_hex(250, 250, 250)

def dark_blue_grey():
    return 0x2c3e50

def matrix_to_xyzquat(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a 4x4 transformation matrix to a block pose representation.
    Returns:
        A block pose representation (x, y, z, qx, qy, qz, qw).
    """
    assert matrix.shape == (4, 4), "Input matrix must be 4x4."

    # Extract translation
    translation = matrix[:3, 3]

    # Extract rotation
    rotation = jaxlie.SO3.from_matrix(matrix[:3, :3])
    quat = rotation.as_quaternion_xyzw()

    return jnp.concatenate([translation, quat])
    
def matrix_to_xyzyaw(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a 4x4 transformation matrix to a block pose representation.
    Returns:
        A block pose representation (x, y, z, yaw).
    """
    assert matrix.shape == (4, 4), "Input matrix must be 4x4."

    # Extract translation
    translation = matrix[:3, 3]

    # Extract rotation
    rotation = jaxlie.SO3.from_matrix(matrix[:3, :3])
    yaw = rotation.compute_yaw_radians()

    return jnp.concatenate([translation, jnp.array([yaw])])

def yaw_to_quat_xyz(place_poses):
    """
    From place_poses (..., 4) [x y z yaw] to  (..., 7) [x y z quat]
        
    """
    assert place_poses.shape[-1] == 4, f'Expected last dim to be 4, got {place_poses.shape[-1]}'
    batch_shape = place_poses.shape[:-1]

    place_poses = place_poses.reshape(-1, 4)
    xyz = place_poses[:, :3]
    yaws = place_poses[:, 3]

    # First rotate 180 degrees about x, then apply yaw about z
    # These jaxlie funcs work for batched/unbatched
    rot_x = jaxlie.SO3.from_x_radians(jnp.pi)
    rotations = jaxlie.SO3.from_z_radians(yaws) @ rot_x
    quats = rotations.as_quaternion_xyzw()
    
    place_poses = jnp.concatenate([xyz, quats], axis=-1)
    place_poses = place_poses.reshape(batch_shape + (-1,))
    
    return place_poses

def grasp_to_q(grasp_pose, nearest_q=jnp.array([0., -jnp.pi/4, 0., -3*jnp.pi/4, 0., jnp.pi/2, jnp.pi/4, 0., 0.])):
    '''
    grasp_pose: [x y z quat]
    returns q, error
    '''
    assert grasp_pose.shape == (7,), f'Expected grasp_pose to be (7,), got {grasp_pose.shape}'
    
    rot = jaxlie.SO3.from_quaternion_xyzw(grasp_pose[3:]).as_matrix()
    pos = grasp_pose[:3]
    O_T_EE = jnp.eye(4)
    O_T_EE = O_T_EE.at[:3, :3].set(rot)
    O_T_EE = O_T_EE.at[:3, 3].set(pos)
    
    # Shift O_T_EE by -0.1 in the intrinsic z direction
    dist_from_mount_to_grasp = 0.0016
    O_T_EE = O_T_EE.at[:3, 3].set(O_T_EE[:3, 3] - dist_from_mount_to_grasp * O_T_EE[:3, 2])

    # qs_0, invalids_0 = analytic_ik_case_consistent(O_T_EE, q7=-2.8973,   q_actual=Simulation.get_neutral_pose())
    # qs_1, invalids_1 = analytic_ik_case_consistent(O_T_EE, q7=-jnp.pi/2, q_actual=Simulation.get_neutral_pose())
    # qs_2, invalids_2 = analytic_ik_case_consistent(O_T_EE, q7= 0,        q_actual=Simulation.get_neutral_pose())
    # qs_3, invalids_3 = analytic_ik_case_consistent(O_T_EE, q7= jnp.pi/2, q_actual=Simulation.get_neutral_pose())
    # qs_4, invalids_4 = analytic_ik_case_consistent(O_T_EE, q7= 2.8973,   q_actual=Simulation.get_neutral_pose())
    # qs = jnp.array([qs_0, qs_1, qs_2, qs_3, qs_4])
    # invalids = jnp.array([invalids_0, invalids_1, invalids_2, invalids_3, invalids_4])
    
    # Do it with different q7 poses
    analytic_ik_vmap = jax.vmap(analytic_ik, in_axes=(None, 0))
    q7s_to_try = jnp.linspace(-2.8970, 2.8970, 9)
    qs, invalids = analytic_ik_vmap(O_T_EE, q7s_to_try)
    qs = qs.reshape(-1, 7)
    invalids = invalids.reshape(-1,)

    # Get valid q_all that is closest to neutral pose
    neutral = jnp.array([0., -jnp.pi/4, 0., -3*jnp.pi/4, 0., jnp.pi/2, jnp.pi/4, 0., 0.])
    joint_distance = jnp.linalg.norm(qs - nearest_q[None, :7], axis=-1)
    joint_distance = jnp.where(invalids, jnp.inf, joint_distance)
    idx = jnp.argmin(joint_distance)
    q_final = qs[idx]
    sol_final = invalids[idx]

    # Pad q with extra [0.1, 0.1]
    q_final = jnp.pad(q_final, (0, 2), constant_values=0.0)
    return q_final, jnp.where(sol_final, jnp.inf, 0.0)

def grasp_to_q2(grasp_pose):
    '''
    grasp_pose: [x y z quat]
    returns q, error
    '''
    assert grasp_pose.shape == (7,), f'Expected grasp_pose to be (7,), got {grasp_pose.shape}'
    
    neutral = jnp.array([0., -jnp.pi/4, 0., -3*jnp.pi/4, 0., jnp.pi/2, jnp.pi/4, 0., 0.])
    
    rot = jaxlie.SO3.from_quaternion_xyzw(grasp_pose[3:]).as_matrix()
    pos = grasp_pose[:3]
    O_T_EE = jnp.eye(4)
    O_T_EE = O_T_EE.at[:3, :3].set(rot)
    O_T_EE = O_T_EE.at[:3, 3].set(pos)
    
    # Shift O_T_EE by -0.1 in the intrinsic z direction
    dist_from_mount_to_grasp = 0.0016
    O_T_EE = O_T_EE.at[:3, 3].set(O_T_EE[:3, 3] - dist_from_mount_to_grasp * O_T_EE[:3, 2])

    # Do it with different q7 poses
    analytic_ik_vmap = jax.vmap(analytic_ik_case_consistent, in_axes=(None, 0, None))
    q7s_to_try = jnp.linspace(-2.8970, 2.8970, 9)
    qs, invalids = analytic_ik_vmap(O_T_EE, q7s_to_try, neutral)
    qs = qs.reshape(-1, 7)
    invalids = invalids.reshape(-1,)

    # Get valid q_all that is closest to neutral pose
    joint_err = jnp.abs(qs[:, 6] - 0.0)  
    idx = jnp.argmin(joint_err)
    q_final = qs[idx]
    sol_final = invalids[idx]

    # Pad q with extra [0.1, 0.1]
    q_final = jnp.pad(q_final, (0, 2), constant_values=0.0)
    return q_final, jnp.where(sol_final, jnp.inf, 0.0)

def interpolate_xyzyaw(start, end, num_steps):
    """
    Interpolates between two poses in (x, y, z, yaw) format.
    Args:
        start: Starting pose (x, y, z, yaw).
        end: Ending pose (x, y, z, yaw).
        num_steps: Number of interpolation steps.
    Returns:
        An array of interpolated poses of shape (num_steps, 4).
    """
    assert start.shape == (4,), "Start pose must be of shape (4,)."
    assert end.shape == (4,), "End pose must be of shape (4,)."
    assert num_steps >= 2, "Number of steps must be at least 2."

    # Linearly interpolate x, y, z
    xyz_interp = jnp.linspace(start[:3], end[:3], num_steps)

    # Interpolate yaw considering angle wrapping
    start_yaw = start[3]
    end_yaw = end[3]
    
    # Compute the shortest angular difference
    delta_yaw = ((end_yaw - start_yaw + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    yaw_interp = start_yaw + jnp.linspace(0, delta_yaw, num_steps)

    # Combine xyz and yaw
    interp_poses = jnp.concatenate([xyz_interp, yaw_interp[:, None]], axis=-1)

    return interp_poses
    
def q_traj(block_traj):
    '''
    Given a traj of N x [x y z yaw], return N x dofs trajectory
    '''
    
    # First find the z rotation offset between the start and end poses
    z_offsets = jnp.linspace(-jnp.pi, jnp.pi, num=16)
    start = block_traj[0]
    end = block_traj[-1]
    
    # check feasibility of each offset
    def feasible(z_offset):
        z_offset_array = jnp.array([0, 0, 0, z_offset])
        _, start_ik_err = grasp_to_q(yaw_to_quat_xyz(start + z_offset_array))
        _, end_ik_err = grasp_to_q(yaw_to_quat_xyz(end + z_offset_array))
        return jnp.isfinite(start_ik_err) & jnp.isfinite(end_ik_err)
    
    feasibles = jax.vmap(feasible)(z_offsets)
    
    # Pick offset nearest to 0 that is feasible
    best_idx = jnp.argmin(jnp.where(feasibles, jnp.abs(z_offsets), jnp.inf))
    best_z_offset = z_offsets[best_idx]
    
    # Now find the q pose (with best yaw offset) for each block pose in the traj
    rotated_block_traj = block_traj + jnp.array([0, 0, 0, best_z_offset])
    block_traj_quat = yaw_to_quat_xyz(rotated_block_traj)
    qs, ik_errs = jax.vmap(grasp_to_q)(block_traj_quat)
    
    return qs

def _q_traj_init_single(start_pose, end_pose, num_states):
    """
    Generates a single joint-space trajectory from a start to an end pose.
    Finds a feasible gripper yaw offset, computes start and end joint
    configurations via IK, and linearly interpolates between them.
    """
    upness = 0.02
    up_vec = jnp.array([0, 0, upness, 0])
    start_pose += up_vec
    end_pose += up_vec

    z_offsets = jnp.linspace(-jnp.pi, jnp.pi * 0.5, num=4)

    def feasible(z_offset, start, end):
        z_offset_array = jnp.array([0, 0, 0, z_offset])
        _, start_ik_err = grasp_to_q(yaw_to_quat_xyz(start + z_offset_array))
        _, end_ik_err = grasp_to_q(yaw_to_quat_xyz(end + z_offset_array))
        return jnp.isfinite(start_ik_err) & jnp.isfinite(end_ik_err)

    feasibles = jax.vmap(feasible, in_axes=(0, None, None))(z_offsets, start_pose, end_pose)
    best_idx = jnp.argmin(jnp.where(feasibles, jnp.abs(z_offsets), jnp.inf))
    best_z_offset = z_offsets[best_idx]

    rotated_start = start_pose + jnp.array([0, 0, 0, best_z_offset])
    rotated_end = end_pose + jnp.array([0, 0, 0, best_z_offset])

    start_q, _ = grasp_to_q(yaw_to_quat_xyz(rotated_start))
    end_q, _ = grasp_to_q(yaw_to_quat_xyz(rotated_end), nearest_q=start_q)

    q_traj = jnp.linspace(start_q, end_q, num_states + 2)
    return q_traj

def q_traj_init(start_state, end_state, num_states):
    """
    Generates initial joint-space trajectories for a multi-block task.

    This function creates `2 * num_blocks - 1` trajectories:
    - Pick-and-place trajectories at even indices (0, 2, ...), moving from a
      block's start pose to its end pose.
    - Return trajectories at odd indices (1, 3, ...), moving from the previous
      block's end pose to the next block's start pose.

    It finds a feasible gripper yaw offset for each trajectory segment, computes
    the start and end joint configurations via inverse kinematics, and then
    linearly interpolates between them in joint space.

    Args:
        start_state (jnp.ndarray): Initial poses of the blocks.
            Shape: `(num_blocks, 4)` where the last dim is `(x, y, z, yaw)`.
        end_state (jnp.ndarray): Final (goal) poses of the blocks.
            Shape: `(num_blocks, 4)`.
        num_states (int): The number of intermediate points for each trajectory.

    Returns:
        jnp.ndarray: The generated joint-space trajectories, including endpoints.
            Shape: `(2 * num_blocks - 1, num_states + 2, 7)`.
    """
    num_blocks = start_state.shape[0]
    num_trajs = 2 * num_blocks - 1

    # 1. Generate pick-and-place trajectories
    pick_place_q_trajs = jax.vmap(_q_traj_init_single, in_axes=(0, 0, None))(
        start_state, end_state, num_states
    )

    # 2. Generate return trajectories by interpolating in joint space
    return_start_qs = pick_place_q_trajs[:-1, -1, :]  # End of each pick-place traj
    return_end_qs = pick_place_q_trajs[1:, 0, :]    # Start of next pick-place traj

    return_q_trajs = jax.vmap(jnp.linspace, in_axes=(0, 0, None))(
        return_start_qs, return_end_qs, num_states + 2
    )

    # 3. Interleave pick-and-place and return trajectories
    q_trajs = jnp.zeros((num_trajs, num_states + 2, pick_place_q_trajs.shape[-1]))
    q_trajs = q_trajs.at[::2, :, :].set(pick_place_q_trajs)
    q_trajs = q_trajs.at[1::2, :, :].set(return_q_trajs)
    
    return q_trajs[:, :, :7]
    
if __name__ == '__main__':
    
    neutral = jnp.array([0., -jnp.pi/4, 0., -3*jnp.pi/4, 0., jnp.pi/2, jnp.pi/4, 0., 0.])
    
    # Use analytic_ik() to get pose at in x axis at z = 0.1
    O_T_EE = jnp.eye(4)
    O_T_EE = O_T_EE.at[2, 3].set(0.1)
    O_T_EE = O_T_EE.at[:3, :3].set(jnp.array([[1, 0, 0],
                                              [0, -1, 0],
                                              [0, 0, -1]]))
    
    q, err = analytic_ik_case_consistent(O_T_EE, q7=0, q_actual=neutral)
    
    print("IK solution at [0, 0, 0.1]:", q, "err:", err)
    
def interp(qs, dist_per_step):
    '''Given qs (N, 7 or 9), make one interp step per dist_per_step, keep original endpoints'''
    
    # Calculate cumulative distances along the path
    diffs = jnp.diff(qs, axis=0)
    dists = jnp.linalg.norm(diffs, axis=1)
    cum_dists = jnp.concatenate([jnp.array([0]), jnp.cumsum(dists)])
    total_dist = cum_dists[-1]

    # Create new sample points at fixed distance intervals, ensuring endpoints are included
    interp_dists = jnp.arange(0, total_dist, dist_per_step)
    interp_dists = jnp.append(interp_dists, total_dist)
    
    # Interpolate for each joint dimension
    interp_fn = lambda q_dim: jnp.interp(interp_dists, cum_dists, q_dim)
    new_qs = jax.vmap(interp_fn)(qs.T).T
    
    return new_qs
