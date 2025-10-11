from functools import partial
import argparse

import jax
import jax.numpy as jnp
import time

import numpy as np

from kinematics.kinematics import _fk, get_ee_pose
from spasm.tower_env import TowerSimulation
from spasm.tower_solve import cost as tower_cost, block_block_penetration, spheres_blocks_collision
from spasm.conversions import interp, matrix_to_xyzyaw, q_traj_init
import jaxlie

class TrajOptParams:
    def __init__(self):
        self.trajopt_steps = 40
        self.trajopt_lr = 0.02
        
        self.arm_collision_weight = 0.6 
        self.block_collision_weight = 0.20
        self.orientation_weight = 0.60
        self.shortness_weight = 0.5 
        self.tower_cost_weight = 2.5 # 2.5
        
        self.viewopt = False

def error_to_down(q):
    """Computes the error between the current robot hand pose and a target pose."""
    target_rot = jaxlie.SO3.from_z_radians(jnp.pi)

    current_pose_mat = get_ee_pose(q)
    current_rot = jaxlie.SO3.from_matrix(current_pose_mat[:3, :3])
    
    # Orientation error using logarithm map
    error_rot = target_rot.inverse() @ current_rot
    axis_angle = error_rot.log()
    
    error_vec = axis_angle * 0.4
    # error_vec = pos_error
    return jnp.sum(jnp.abs(error_vec))

def block_traj_blocks_collision(sim, block_traj, blocks, scale):
    """
    Computes collision between a block trajectory and a collection of static blocks.
    block_traj: (T, 4)
    blocks: (num_blocks, 4)
    """
    assert block_traj.ndim == 2 and block_traj.shape[1] == 4, f'Expected (T, 4), got {block_traj.shape}'
    assert blocks.ndim == 2 and blocks.shape[1] == 4, f'Expected (num_blocks, 4), got {blocks.shape}'
        
    # vmap over the trajectory poses, for each pose, vmap over the static blocks
    # Shape (T, num_blocks)
    penetrations = jax.vmap(
        lambda traj_pose: jax.vmap(
            block_block_penetration, in_axes=(None, None, 0, None)
        )(sim, traj_pose, blocks, scale)
    )(block_traj)
    return penetrations.sum(axis=0)

def sweep_spheres_collision(spheres_xyz, spheres_radii, s2, s2r):
    assert spheres_xyz.shape[1] == 3, f'Expected (N, 3), got {spheres_xyz.shape}' # (N, 3)
    assert spheres_radii.ndim == 1, f'Expected (N,), got {spheres_radii.shape}' # (N,)
    assert s2.shape[1] == 3
    assert s2r.ndim == 1
    
    block_centers = s2
    dists = jnp.linalg.norm(spheres_xyz[:, None, :] - block_centers[None, :, :], axis=-1) # (num_spheres, num_blocks)
    dists = dists - (spheres_radii[:, None] + s2r[None, :]) - 10e-2
    dists = jnp.where(dists < 0, -dists, 0.0) # Only care about penetration
    return (dists).sum()

def cost(params: TrajOptParams, sim: TowerSimulation, initial_state, q_trajs, i):
    """
    Computes the total cost for a batch of trajectories.
    q_trajs: (num_trajs, T, 7)
    """
    num_blocks = sim.num_blocks
    num_trajs = 2 * num_blocks - 1
    assert q_trajs.ndim == 3, f'Expected (num_trajs, T, 7), got {q_trajs.shape}'
    assert q_trajs.shape[0] == num_trajs
    assert q_trajs.shape[2] == 7, f'Expected (num_trajs, T, 7), got {q_trajs.shape}'
    assert initial_state.shape == (num_trajs, 7), f'Expected (num_trajs, 7), got {initial_state.shape}'

    # The conversion from q to block pose should return 4 values (x,y,z,yaw)
    q_to_block = lambda q: matrix_to_xyzyaw(get_ee_pose(q))
    q_to_block_vmap = jax.vmap(jax.vmap(q_to_block))
    
    # --- Convert q to poses ---
    ee_poses_xyzyaw = q_to_block_vmap(q_trajs)
    initial_poses = ee_poses_xyzyaw[:, 0, :]
    final_poses = ee_poses_xyzyaw[:, -1, :]
    
    # --- Robot Arm Collision Cost ---
    def arm_collision_cost_fn(q_traj, traj_idx):
        """
        Computes collision cost for the arm of a single trajectory against other blocks and obstacles.
        - Trajectory `traj_idx` arm collides with initial blocks `j > block_idx`.
        - Trajectory `traj_idx` arm collides with final blocks `j < block_idx`.
        - Trajectory `traj_idx` arm collides with obstacles.
        """
        block_idx = traj_idx // 2
        def single_q_cost(q):
            # (num_spheres, 3) and (num_spheres,)
            spheres, radii = _fk(q)

            # Arm for trajectory `traj_idx` collides with initial blocks `j` where `j > block_idx`
            initial_mask = jnp.arange(num_blocks) > block_idx
            cost_initial = spheres_blocks_collision(sim, spheres, radii, initial_poses[::2])
            assert cost_initial.shape == initial_mask.shape, f"cost_initial shape should be {initial_mask.shape}, got {cost_initial.shape}"
            cost_initial = jnp.sum(cost_initial * initial_mask)

            # Arm for trajectory `traj_idx` collides with final blocks `j` where `j < block_idx`
            final_mask = jnp.arange(num_blocks) < block_idx
            cost_final = spheres_blocks_collision(sim, spheres, radii, final_poses[::2])
            assert cost_final.shape == final_mask.shape, f"cost_final shape should be {final_mask.shape}, got {cost_final.shape}"
            cost_final = jnp.sum(cost_final * final_mask)

            # Arm collision with obstacles
            cost_obstacles = sweep_spheres_collision(spheres, radii, sim.obstacle_poses, sim.obstacle_radii)
            
            # Collision with ground
            cost_ground = jax.nn.relu(radii - spheres[:, 2] + 2e-2).sum() * 10
            
            return cost_initial + cost_final + cost_obstacles + cost_ground
        
        return jax.vmap(single_q_cost)(q_traj).sum()

    # Vmap over trajectories
    arm_collision_cost = jax.vmap(arm_collision_cost_fn, in_axes=(0, 0))(q_trajs, jnp.arange(num_trajs)).sum()
    arm_collision_cost *= params.arm_collision_weight

    # --- Robot Orientation Cost ---
    orientation_cost = (jax.vmap(jax.vmap(error_to_down))(q_trajs[:, [0, -1], :])).sum() * params.orientation_weight

    # --- Held Block Collision Cost ---
    def held_block_collision_cost_fn(ee_poses, block_idx):
        """
        Computes collision cost for the held block.
        - The final pose of the held block for trajectory `block_idx` collides with final poses of blocks `j < block_idx`.
        - The trajectory of the held block for trajectory `block_idx` collides with initial poses of blocks `j > block_idx`.
        """
        assert ee_poses.ndim == 2 and ee_poses.shape[1] == 4, f'Expected (T, 4), got {ee_poses.shape}'
        
        # Collision of block trajectory `block_idx` with initial state blocks `j > block_idx`
        initial_mask = jnp.arange(num_blocks) > block_idx
        # Vmap over the trajectory poses
        traj_block_collision = block_traj_blocks_collision(sim, ee_poses, initial_poses[::2], 1.0)
        assert traj_block_collision.shape == initial_mask.shape, f"traj_block_collision shape should be {initial_mask.shape}, got {traj_block_collision.shape}"
        traj_block_collision = jnp.sum(traj_block_collision * initial_mask)
        
        # Collision of final block pose `block_idx` with final poses `j < block_idx`
        final_mask = jnp.arange(num_blocks) < block_idx
        final_block_collision = block_traj_blocks_collision(sim, ee_poses, final_poses[::2], 1.2)
        assert final_block_collision.shape == final_mask.shape, f"final_block_collision shape should be {final_mask.shape}, got {final_block_collision.shape}"
        final_block_collision = jnp.sum(final_block_collision * final_mask)
        
        # Collision with ground
        cost_ground = jax.nn.relu(-ee_poses[:, 2] + 2e-2 + sim.block_height).sum() * 10

        return final_block_collision + traj_block_collision + cost_ground

    held_block_collision_cost = jax.vmap(held_block_collision_cost_fn, in_axes=(0, 0))(
        ee_poses_xyzyaw[::2], jnp.arange(num_blocks)
    ).sum()
    held_block_collision_cost *= params.block_collision_weight

    # --- Trajectory Shortness Cost ---
    # Minimize distance between consecutive points, excluding the fixed start point
    
    shortness_cost = jnp.linalg.norm(q_trajs[:, 1:, :] -  q_trajs[:, :-1, :], axis=-1).sum()
    shortness_cost += jnp.linalg.norm(q_trajs[:, 0, :] - initial_state, axis=-1).sum() # ::2??
    
    second_last_pose = jax.lax.stop_gradient(q_trajs[:, -2, :])
    last_pose = q_trajs[:, -1, :]
    diff = jnp.linalg.norm(last_pose - second_last_pose, axis=-1).sum()
    # shortness_cost += diff * 100
    
    shortness_cost *= params.shortness_weight

    # --- Final Tower Cost ---
    tower_cost_val = tower_cost(sim, final_poses[::2], initial_state[::2]) * params.tower_cost_weight
    
    if params.viewopt:
        jax.debug.print("Arm collision cost: {:.2f}, Orientation cost: {:.2f}, Held block collision cost: {:.2f}, Shortness cost: {:.2f}, Tower cost: {:.2f}",
                   arm_collision_cost, orientation_cost, held_block_collision_cost, shortness_cost, tower_cost_val)
    
    # FIXME
    # tower_schedule = (1 - jnp.clip(i / 15, 0.0, 1.0))
    # tower_schedule = 1 / (1.0 - i / 40)
    # tower_schedule *= 0.4
    shortness_schedule = jnp.where((i > 0) & (i < 20), 1 - i / 20, 0.0)
    total_cost = 1.6 * tower_cost_val + arm_collision_cost + orientation_cost + held_block_collision_cost + shortness_cost * shortness_schedule
    return total_cost

@partial(jax.jit, static_argnames=('params', 'sim'))
def opt(params: TrajOptParams, sim: TowerSimulation, initial_state, final_state):
    """
    Optimizes a single trajectory segment. Only the interpolated points are modified.
    initial_state: (num_blocks, 4)
    final_state: (num_blocks, 4)
    """
    
    # Num interpolation points
    T = 10
    num_blocks = sim.num_blocks
    num_trajs = 2 * num_blocks - 1
     
    # (num_trajs, T, 4)
    q_trajs_init = q_traj_init(initial_state, final_state, T)
    assert q_trajs_init.shape == (num_trajs, T + 2, 7), f'Expected ({num_trajs}, {T+2}, 7), got {q_trajs_init.shape}'
    
    # def schedule_lr(init_lr, step, total_steps):
    #     return 1.0 / jnp.exp(step / total_steps * 3) * init_lr
    
    def schedule_lr(init_lr, step, total_steps):
        return (1.0 - step / total_steps) * init_lr
    
    def opt_step(i, q_trajs):

        lr = schedule_lr(params.trajopt_lr, i, params.trajopt_steps)
        grad = jax.grad(cost, argnums=3)(params, sim, q_trajs[:, 0, :], q_trajs[:, 1:, :], i)
        q_trajs = q_trajs.at[:, 1:, :].add(grad * -lr)
        
        # Set the return trajectories' start and end
        # The start of a return trajectory is the end of the previous pick-and-place trajectory.
        return_starts = q_trajs[::2, -1, :][:-1]
        q_trajs = q_trajs.at[1::2, 0, :].set(return_starts)

        # The end of a return trajectory is the start of the next pick-and-place trajectory.
        return_ends = q_trajs[::2, 0, :][1:]
        q_trajs = q_trajs.at[1::2, -1, :].set(return_ends)

        if params.viewopt:
            def callback(q_trajs):
                final_poses = jax.vmap(lambda q: matrix_to_xyzyaw(get_ee_pose(q)))(q_trajs[::2, -1, :])
                sim.set_state(final_poses)
                sim.draw_trajs(q_trajs)
                sim.render()
            jax.debug.callback(callback, q_trajs)
        
        return q_trajs
    
    opt_q_traj = jax.lax.fori_loop(0, params.trajopt_steps, opt_step, q_trajs_init)
    return opt_q_traj
    
if __name__ == '__main__':
    # with jax.log_compiles():
    from kinematics.util import jax_cache_on
    jax_cache_on()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', action='store_true', help='Time benchmark.')
    parser.add_argument('--viewopt', action='store_true', help='Enable view optimization.')
    args = parser.parse_args()

    try:
        data = jnp.load('saved/tower.npz')
        solutions = data['opt_particles'] # (num_solutions, num_blocks, 4)
        init_state = data['init_state']   # (num_blocks, 4)
    except FileNotFoundError:
        print("Could not find 'saved/tower.npz'. Please run 'spasm/tower_solve.py' first.")
        exit()
        
    # solutions[0, :, 0] += 0.1
    # solutions[0, :, 1] += 0.15

    params = TrajOptParams()
    params.viewopt = args.viewopt
    sim = TowerSimulation(num_blocks=init_state.shape[0], num_obs=10)
    sim.z_error_mul = 5.0
    sim.set_state(solutions[0])
    sim.render()
    
    if sim.num_obs == 0:
        params.trajopt_steps = 10
    if sim.num_obs == 1:
        params.trajopt_steps = 30
        
    # Warm up
    if args.bench:
        opt_traj = opt(params, sim, init_state, solutions[0])
        opt_traj.block_until_ready()
    
    # (num_trajs, T, 4), (num_trajs, 4)
    begin = time.perf_counter()
    for _ in range(10 if args.bench else 1):
        # Shape (num_trajs, T, 7)
        q_opt_traj = opt(params, sim, init_state, solutions[0])
        q_opt_traj.block_until_ready()
    end = time.perf_counter()
       
    jnp.save('saved/tower_traj.npy', q_opt_traj)
    
    if args.bench:
        print(f"Average optimization time: {(end - begin) * 1000 / 10:.2f} ms")
    
    # q_traj_towerj = jax.jit(q_traj_tower)
    
    q_to_block = lambda q: matrix_to_xyzyaw(get_ee_pose(q))
    q_to_block_jit = jax.jit(q_to_block)
    
    sim.render()
    time.sleep(1)
    
    # No trajs in final rollout
    sim.vis['traj'].delete()
    
    while True:
        # qs = [q_traj_towerj(traj) for traj in opt_traj]
        sim.set_state(init_state)
        sim.render()
                
        for traj_idx in range(q_opt_traj.shape[0]):
            block_idx = traj_idx // 2
            is_pick_place = (traj_idx % 2 == 0)
            
            q_interp = interp(q_opt_traj[traj_idx], 0.03) # FIXME 0.03
            
            for time_idx in range(q_interp.shape[0]):
                ee = get_ee_pose(q_interp[time_idx])
                if is_pick_place:
                    sim.block_poses_matrix[block_idx] = np.asarray(ee)
                sim.set_robot_pose(q_interp[time_idx])
                sim.render()
                
            # Set to final state
            if is_pick_place:
                sim.set_one_state(block_idx, q_to_block_jit(q_opt_traj[traj_idx, -1]))
                sim.render()
                time.sleep(0.5)

            # # Final state in traj is actually this
            # sim.set_one_state(block_idx, final_state[block_idx])
            # # sim.set_one_state(block_idx, q_to_block_jit(q_opt_traj[block_idx, -1]))
            # sim.set_one_state(block_idx, final_state[block_idx])
            # sim.render()
            
        # Draw only final state without trajs
        sim.vis['traj'].delete()
        sim.set_robot_pose(sim.get_neutral_pose())
            
        time.sleep(2)

