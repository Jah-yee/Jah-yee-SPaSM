import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
from matplotlib.pyplot import step
import numpy as np
import jaxlie
from kinematics.kinematics import _fk, get_ee_pose
from spasm.conversions import (grasp_to_q, interp, matrix_to_xyzyaw, q_traj_init,
                               yaw_to_quat_xyz)
from spasm.solve import cost as tetris_cost
from spasm.solve import sphere_sphere_penetration, sphere_wall_penetration
from spasm.tetris_env import Simulation, _block_pose_to_spheres


class TrajOptParams:
    def __init__(self):
        self.trajopt_steps = 20 #10 FIXME
        self.trajopt_lr = 0.003 # 0.005

        self.arm_collision_weight = 0.005
        self.block_collision_weight = 1.20
        self.orientation_weight = 0.05
        self.shortness_weight = 0.50 # 0.06
        self.tetris_cost_weight = 0.003

        self.viewopt = False
        
        self.opt = 'linear'


def error_to_down(q):
    """Computes the error between the current robot hand pose and a target pose."""
    target_rot = jaxlie.SO3.from_z_radians(jnp.pi)

    current_pose_mat = get_ee_pose(q)
    current_rot = jaxlie.SO3.from_matrix(current_pose_mat[:3, :3])

    # Orientation error using logarithm map
    error_rot = target_rot.inverse() @ current_rot
    axis_angle = error_rot.log()

    error_vec = axis_angle * 0.4
    return jnp.sum(jnp.abs(error_vec))

# Schleem
schleem = 5e-2 # schleem is 10e-2

def block_traj_collision(sim, block_spheres_traj, other_blocks_spheres):
    """
    block_spheres_traj: (T, num_spheres, 4)
    other_blocks_spheres: (num_other_blocks, num_spheres, 4)
    Returns: (num_other_blocks,)
    """
    def single_step_collision(block_spheres):
        # block_spheres: (num_spheres, 4)
        # other_blocks_spheres: (num_other_blocks, num_spheres, 4)
        # -> (num_other_blocks)
        penetrations = jax.vmap(
            lambda other_spheres: sphere_sphere_penetration(block_spheres, other_spheres, schleem).sum()
        )(other_blocks_spheres)
        return penetrations.sum()

    return jax.vmap(single_step_collision)(block_spheres_traj).sum()


def cost(params: TrajOptParams, sim: Simulation, initial_state_q, q_trajs):
    """
    Computes the total cost for a batch of trajectories.
    q_trajs: (num_trajs, T, 7)
    initial_state: (num_blocks, 4)
    """
    num_blocks = sim.num_blocks
    num_trajs = 2 * num_blocks - 1

    # Reconstruct full q_trajs. The optimized part is q_trajs, the start is fixed.
    # q_trajs = jnp.concatenate([initial_state_q[:, None, :], q_trajs], axis=1)

    assert q_trajs.ndim == 3, f'Expected (num_trajs, T, 7), got {q_trajs.shape}'
    assert q_trajs.shape[0] == num_trajs
    assert q_trajs.shape[2] == 7, f'Expected (num_trajs, T, 7), got {q_trajs.shape}'
    assert initial_state_q.shape == (num_trajs, 7), f'Expected initial_state_q shape ({num_trajs}, 7), got {initial_state_q.shape}'

    q_to_block = lambda q: matrix_to_xyzyaw(get_ee_pose(q))
    q_to_block_vmap = jax.vmap(jax.vmap(q_to_block))

    # --- Convert q to poses ---
    ee_poses_xyzyaw = q_to_block_vmap(q_trajs[::2])
    initial_poses = jax.vmap(q_to_block)(q_trajs[::2, 0, :])
    final_poses = ee_poses_xyzyaw[:, -1, :]

    # --- Robot Arm Collision Cost ---
    def arm_collision_cost_fn(q_traj, traj_idx):
        block_idx = traj_idx // 2
        def single_q_cost(q):
            spheres, radii = _fk(q)
            arm_spheres = jnp.hstack([spheres, radii[:, None]])

            # Arm collides with initial blocks j > block_idx
            initial_mask = jnp.arange(num_blocks) > block_idx
            initial_spheres = jax.vmap(_block_pose_to_spheres, in_axes=(0, 0))(sim.block_spheres, initial_state)
            cost_initial = jax.vmap(lambda s: sphere_sphere_penetration(arm_spheres, s, 2e-2).sum())(initial_spheres)
            cost_initial = jnp.sum(cost_initial * initial_mask)

            # Arm collides with final blocks j < block_idx
            final_mask = jnp.arange(num_blocks) < block_idx
            final_spheres = jax.vmap(_block_pose_to_spheres, in_axes=(0, 0))(sim.block_spheres, final_poses)
            cost_final = jax.vmap(lambda s: sphere_sphere_penetration(arm_spheres, s, 2e-2).sum())(final_spheres)
            cost_final = jnp.sum(cost_final * final_mask)

            # Arm collision with ground
            cost_ground = jax.nn.relu(-arm_spheres[:, 2] + arm_spheres[:, 3] + 6e-2).sum()

            # Arm collision with walls
            cost_walls = sphere_wall_penetration(arm_spheres, sim, 0).sum()

            return cost_initial + cost_final + cost_walls + cost_ground
        return jax.vmap(single_q_cost)(q_traj).sum()

    arm_collision_cost = jax.vmap(arm_collision_cost_fn, in_axes=(0, 0))(q_trajs, jnp.arange(num_trajs)).sum()
    arm_collision_cost *= params.arm_collision_weight

    # --- Robot Orientation Cost ---
    orientation_cost = (jax.vmap(jax.vmap(error_to_down))(q_trajs[:, [0, -1], :])).sum() * params.orientation_weight

    # --- Held Block Collision Cost ---
    def held_block_collision_cost_fn(ee_poses, block_idx):
        block_spheres = sim.block_spheres[block_idx]
        block_spheres_traj = jax.vmap(_block_pose_to_spheres, in_axes=(None, 0))(block_spheres, ee_poses)

        # Collision with initial blocks j > block_idx
        initial_mask = jnp.arange(num_blocks) > block_idx
        initial_spheres = jax.vmap(_block_pose_to_spheres, in_axes=(0, 0))(sim.block_spheres, initial_poses)
        
        # traj_block_collision is (num_blocks)
        traj_block_collision = jax.vmap(lambda s: block_traj_collision(sim, block_spheres_traj, s[None, ...]))(initial_spheres)
        traj_block_collision = jnp.sum(traj_block_collision * initial_mask)

        # Collision of final block pose with final poses j < block_idx
        final_mask = jnp.arange(num_blocks) < block_idx
        final_spheres = jax.vmap(_block_pose_to_spheres, in_axes=(0, 0))(sim.block_spheres, final_poses)
        final_block_collision = jax.vmap(lambda s: sphere_sphere_penetration(block_spheres_traj[-1], s, schleem).sum())(final_spheres)
        final_block_collision = jnp.sum(final_block_collision * final_mask)

        # Wall collision for held block
        wall_collision = sphere_wall_penetration(block_spheres_traj.reshape(-1, 4), sim).sum()

        # Ground collision for held block
        ground_collision = jax.nn.relu(-block_spheres_traj[..., 2] + block_spheres_traj[..., 3] + 6e-2).sum() * 10

        return traj_block_collision + final_block_collision + wall_collision + ground_collision

    held_block_collision_cost = jax.vmap(held_block_collision_cost_fn, in_axes=(0, 0))(
        ee_poses_xyzyaw, jnp.arange(num_blocks)
    ).sum()
    held_block_collision_cost *= params.block_collision_weight

    # --- Trajectory Shortness Cost ---
    shortness_cost = jnp.linalg.norm(q_trajs[:, 1:, :] - q_trajs[:, :-1, :], axis=-1).sum()
    shortness_cost += jnp.linalg.norm(q_trajs[:, 0, :] - initial_state_q, axis=-1).sum()
    
    shortness_cost *= params.shortness_weight

    # --- Final Tetris Cost ---
    tetris_cost_val = tetris_cost(params, sim, final_poses) * params.tetris_cost_weight
    
    if params.viewopt:
        jax.debug.print('Arm collision cost: {:.4f}, Orientation cost: {:.4f}, Held block collision cost: {:.4f}, Shortness cost: {:.4f}, Tetris cost: {:.4f}',
                  arm_collision_cost, orientation_cost, held_block_collision_cost, shortness_cost, tetris_cost_val)
    
    total_cost = arm_collision_cost + orientation_cost + held_block_collision_cost + shortness_cost + tetris_cost_val
    
    return total_cost


@partial(jax.jit, static_argnames=('params', 'sim'))
def opt(params: TrajOptParams, sim: Simulation, initial_state, final_state):
    T = 8
    num_blocks = sim.num_blocks
    num_trajs = 2 * num_blocks - 1

    q_trajs_init = q_traj_init(initial_state, final_state, T)
    assert q_trajs_init.shape == (num_trajs, T + 2, 7), f'Expected ({num_trajs}, {T+2}, 7), got {q_trajs_init.shape}'

    def schedule_lr(init_lr, step, total_steps):
        return (1.0 - step / total_steps) * init_lr

    def opt_step(i, q_trajs):
        lr = schedule_lr(params.trajopt_lr, i, params.trajopt_steps)
        grad = jax.grad(cost, argnums=3)(params, sim, q_trajs[:, 0, :], q_trajs[:, 1:, :])
        
        # Multiply grad for joint 1, 3
        grad = grad.at[:, :, 1].multiply(2)
        grad = grad.at[:, :, 3].multiply(2)
        
        # Print grad
        if params.viewopt:
            jax.debug.print('grad {}', grad)
        
        # Apply gradients
        grad = jnp.nan_to_num(grad, nan=0.0)
        q_trajs = q_trajs.at[:, 1:-1, :].add(-params.trajopt_lr * grad[:, :-1, :])
        
        # Set the return trajectories' start and end
        # The start of a return trajectory is the end of the previous pick-and-place trajectory.
        return_starts = q_trajs[::2, -1, :][:-1]
        q_trajs = q_trajs.at[1::2, 0, :].set(return_starts)

        # The end of a return trajectory is the start of the next pick-and-place trajectory.
        return_ends = q_trajs[::2, 0, :][1:]
        q_trajs = q_trajs.at[1::2, -1, :].set(return_ends)

        if params.viewopt:
            def callback(i, q_trajs):
                if i % 2 != 0:
                    return
                final_poses = jax.vmap(lambda q: matrix_to_xyzyaw(get_ee_pose(q)))(q_trajs[::2, -1, :])
                sim.set_state(final_poses)
                sim.draw_trajs(q_trajs)
                sim.render()
                print('Render step')
            jax.debug.callback(callback, i, q_trajs)
            
        return q_trajs

    opt_q_traj = jax.lax.fori_loop(0, params.trajopt_steps, opt_step, q_trajs_init)

    return opt_q_traj


if __name__ == '__main__':
    from kinematics.util import jax_cache_on
    jax_cache_on()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_blocks', type=int, default=5, help="Number of blocks")
    parser.add_argument('--bench', action='store_true', help='Time benchmark.')
    parser.add_argument('--viewopt', action='store_true', help='Enable view optimization.')
    args = parser.parse_args()

    try:
        solutions = jnp.load('saved/tetris.npy')
    except FileNotFoundError:
        print("Could not find 'saved/tetris.npy'. Please run 'spasm/solve.py' first.")
        exit()

    params = TrajOptParams()
    params.viewopt = args.viewopt
    sim = Simulation(num_blocks=solutions.shape[0])
    sim.render()

    initial_state = jnp.array(sim.block_poses_original)
    final_state = solutions

    # warm up
    if args.bench:
        opt_traj = opt(params, sim, initial_state, final_state)
        opt_traj.block_until_ready()

    begin = time.perf_counter()
    for _ in range(10 if args.bench else 1):
        q_opt_traj = opt(params, sim, initial_state, final_state)
        q_opt_traj.block_until_ready()
    end = time.perf_counter()

    if args.bench:
        print(f"Time: {(end - begin) * 1000 / 10:.2f} ms")

    # Save q_opt_traj
    jnp.save('saved/tetris_traj.npy', q_opt_traj)
    print("Saved trajectory to 'saved/tetris_traj.npy'")
    
    q_to_block = lambda q: matrix_to_xyzyaw(get_ee_pose(q))
    q_to_block_jit = jax.jit(q_to_block)

    while True:
        sim.set_state(initial_state)
        sim.render()
                
        for traj_idx in range(q_opt_traj.shape[0]):
            block_idx = traj_idx // 2
            is_pick_place = (traj_idx % 2 == 0)
            
            q_interp = interp(q_opt_traj[traj_idx], 0.3)
            
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
        time.sleep(1)

