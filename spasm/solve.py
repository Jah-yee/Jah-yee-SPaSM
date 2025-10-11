import copy
from functools import partial
import os
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import time

from kinematics.kinematics import get_ee_pose
from spasm.conversions import grasp_to_q, interpolate_xyzyaw, matrix_to_xyzyaw, q_traj, yaw_to_quat_xyz
from spasm.tetris_env import Simulation, _block_pose_to_spheres, block_pose_to_spheres
# from spasm.trajopt import generate_trajectory

class SpasmParams:
    def __init__(self):
        self.sampling_batch = 512
        self.opt_batch = 64
        self.opt_steps = 15
        self.viewopt = False
        self.noopt = False
        
        self.lr = 0.6
        self.quadratic_lr = 6
        self.quadratic_opt_steps = 3
        
        self.cost_thresh = 0.44
        
        self.opt = 'linear'  # 'linear' or 'quadratic'
    
    def _tree_flatten(self):
        # Jax-able types (numpy, jnp)
        children = None
        
        # Others
        aux_data = {k: v for k, v in self.__dict__.items()}  
        
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

def sphere_sphere_penetration(spheres1, spheres2, margin=0.010):
    """
    Computes the penetration between two sets of spheres.
    Assumes no self-collision between spheres1 and spheres2!

    spheres1: (N, 4) [x, y, z, r]
    spheres2: (M, 4) [x, y, z, r]
    return penetration: (N)
    """
    assert spheres1.shape[1] == 4, f"spheres1 should have shape (N, 4), got {spheres1.shape}"
    assert spheres2.shape[1] == 4, f"spheres2 should have shape (M, 4), got {spheres2.shape}"

    # Expand dims to broadcast
    spheres1 = spheres1[:, jnp.newaxis, :]  # (N, 1, 4)
    spheres2 = spheres2[jnp.newaxis, :, :]  # (1, M, 4)

    # Pairwise distances Shape: (N, M)
    dist = jnp.linalg.norm(spheres1[..., :3] - spheres2[..., :3], axis=-1)
    
    # Pairwise radii sum (N, M)
    radii_sum = spheres1[..., 3] + spheres2[..., 3]
    
    # Penetration FIXME changed something
    penetration = jnp.maximum(-margin, radii_sum - dist) + margin

    return jnp.abs(penetration)
    
def sphere_wall_penetration(spheres, sim, rizz_aura=0.010):
    """
    Computes the collision cost between a set of spheres and the goal region walls.
    The walls are infinitely thick, facing away from the goal_position.

    Args:
        spheres (jnp.ndarray): Shape (N, 4) where each row is (x, y, z, r).
        sim: The simulation object containing goal_position and goal_dims.

    Returns:
        penetration of shape (N,)
    """
    goal_pose = sim.goal_position
    goal_dims = sim.goal_dims
    
    sphere_centers = spheres[..., :3]
    sphere_radii = spheres[..., 3]

    # Goal region boundaries
    x_max_b = goal_pose[0] + goal_dims[0] / 2.0
    x_min_b = goal_pose[0] - goal_dims[0] / 2.0
    y_max_b = goal_pose[1] + goal_dims[1] / 2.0
    y_min_b = goal_pose[1] - goal_dims[1] / 2.0

    # Penetration for each of the 4 walls (distance of sphere center from boundary if outside)
    aura = rizz_aura * 2
    pen_x_pos = jnp.maximum(-aura, sphere_centers[..., 0] - x_max_b + sphere_radii) + aura
    pen_x_neg = jnp.maximum(-aura, x_min_b - sphere_centers[..., 0] + sphere_radii) + aura
    pen_y_pos = jnp.maximum(-aura, sphere_centers[..., 1] - y_max_b + sphere_radii) + aura
    pen_y_neg = jnp.maximum(-aura, y_min_b - sphere_centers[..., 1] + sphere_radii) + aura
        
    # Shape: (4, N)
    xy_penetration = jnp.stack([pen_x_pos, pen_x_neg, pen_y_pos, pen_y_neg])
    xy_penetration = jnp.max(xy_penetration, axis=0)  # Shape: (N,)
    
    # Now add on z penetration
    wall_penetration = jnp.maximum(0, sim.block_z + 0.07 - spheres[..., 2] + sphere_radii)
    floor_penetration = jnp.maximum(0, 0.00 - spheres[..., 2] + sphere_radii)
    z_penetration = jnp.where(xy_penetration > rizz_aura + 1e-6, wall_penetration * 0.6, floor_penetration)

    return jnp.abs(xy_penetration) + jnp.abs(z_penetration)

def cost(params, sim, block_poses):
    '''
    Wall collision and sphere collision cost
    block_poses: (num_blocks, 4)
    return: scalar cost
    '''
    assert block_poses.shape == (sim.num_blocks, 4), f"block_poses should be of shape ({sim.num_blocks}, 4), got {block_poses.shape}"

    # Transform by place_poses
    sphere_poses = block_pose_to_spheres(sim, block_poses)

    # Get all sphere-sphere distances
    sphere_penetration = 0
    block_i = []
    block_j = []
    for i in range(len(sphere_poses)):
        for j in range(len(sphere_poses)):
            if i == j:
                continue
            block_i.append(i)
            block_j.append(j)
    
    ssp_func = lambda i, j: sphere_sphere_penetration(sphere_poses[i], sphere_poses[j])
    ssp = jax.vmap(ssp_func)(jnp.array(block_i), jnp.array(block_j))
    sphere_penetration += ssp.sum() if params.opt == 'linear' else (ssp**2).sum()

    # Get sphere-wall distances
    wall_penetration = sphere_wall_penetration(sphere_poses.reshape(-1, 4), sim)
    wall_penetration = wall_penetration.sum() if params.opt == 'linear' else (wall_penetration**2).sum()
    
    return wall_penetration * 3 + sphere_penetration * 0.5 

def clip_particle(sim, block_poses):
    
    # Clip place_poses by xy
    xy = block_poses[..., :2] # (num_blocks, 2)
    yaw = block_poses[..., 3, None] # (num_blocks, 1)
    half_dims = sim.goal_dims[:2] / 2.0
    
    xy_clipped = jnp.clip(xy - sim.goal_position[:2], -half_dims, half_dims) \
                     + sim.goal_position[:2]
    
    z_clipped = jnp.full((xy_clipped.shape[0], 1), sim.block_z)

    yaw = (yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi  # Wrap yaw to [-pi, pi]

    return jnp.concatenate([xy_clipped, z_clipped, yaw], axis=-1)

def sample_particles(params: SpasmParams, sim, key):

    # Sample block poses
    xy = jax.random.uniform(key, (params.sampling_batch, sim.num_blocks, 2), 
                            minval=-sim.goal_dims[:2] / 2, 
                            maxval=sim.goal_dims[:2] / 2)
    
    xy += sim.goal_position[:2]
    zs = jnp.full((params.sampling_batch, sim.num_blocks, 1), sim.block_z)
    
    yaws = jax.random.uniform(key, (params.sampling_batch, sim.num_blocks, 1), 
                              minval=-jnp.pi, maxval=jnp.pi)
    
    # (N, num_blocks, 4)
    block_poses = jnp.concatenate([xy, zs, yaws], axis=-1)
        
    # Get top k particles
    errors = jax.vmap(cost, in_axes=(None, None, 0))(params, sim, block_poses)
    top_indices = jnp.argsort(errors)[:params.opt_batch]
    top_place_poses = block_poses[top_indices]
            
    # (N, num_blocks, 4)
    return top_place_poses

def opt_step(i, params: SpasmParams, sim, particle):
    # Grad
    J = jax.grad(cost, argnums=2)(params, sim, particle)
    
    # Linear decay
    if params.opt == 'linear':
        lr = (1 - i / params.opt_steps) * params.lr
        delta = -J * 5e-3
        delta = delta.at[..., -1].mul(1.0e3) 
        particle_new = particle + delta * lr
    else:
        # lr = (1 - i / params.opt_steps) * params.quadratic_lr
        lr = params.quadratic_lr
        delta = -J * 5e-3
        delta = delta.at[..., -1].mul(250) 
        particle_new = particle + delta * lr    
    
    particle_new = clip_particle(sim, particle_new)
            
    if params.viewopt:
        def callback(particle, i):
            print('viewing step', i)
            sim.set_state(particle)
            sim.render()
        jax.debug.callback(callback, particle_new, i)
    
    return particle_new

def project(params: SpasmParams, sim, particles):
    
    quad_opt_params = copy.copy(params)
    quad_opt_params.opt = 'quadratic'
    
    opt_stepp = lambda i, particle: opt_step(i, params, sim, particle)
    quad_opt_stepp = lambda i, particle: opt_step(i, quad_opt_params, sim, particle)
    
    def opt(particle): 
        particle = jax.lax.fori_loop(0, params.opt_steps, opt_stepp, particle)
        particle = jax.lax.fori_loop(0, quad_opt_params.quadratic_opt_steps, quad_opt_stepp, particle)
        return particle
    
    return jax.vmap(opt)(particles)
    
@partial(jax.jit, static_argnames=['params', 'sim'])
def solve(params: SpasmParams, sim, key):
    
    cost_fun = jax.vmap(cost, in_axes=(None, None, 0))
    
    def solve_once(state):
        key, _, _ = state
    
        # (opt_batch, num_blocks, 4)
        particles = sample_particles(params, sim, key)
        opt_particles = project(params, sim, particles)        
        opt_error = cost_fun(params, sim, opt_particles)
        min_idx = jnp.argmin(opt_error)
        min_particle = opt_particles[min_idx]
        min_error = opt_error[min_idx]
        
        _, new_key = jax.random.split(key)
        
        return new_key, min_particle, min_error
    
    def cond(state):
        # Do this until the best error is < params.cost_thresh
        key, min_particle, min_error = state
        return min_error > params.cost_thresh

    _, min_particle, min_error = jax.lax.while_loop(cond, solve_once, (key, jnp.zeros((sim.num_blocks, 4)), jnp.inf))
    # _, min_particle, min_error = solve_once((key, jnp.zeros((sim.num_blocks, 4)), jnp.inf))
    return min_particle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_blocks', type=int, default=5, help="Number of blocks")
    parser.add_argument('--noopt', action='store_true', help="Disable optimization")
    parser.add_argument('--cached', action='store_true', help="view cached solution")
    parser.add_argument('--viewopt', action='store_true', help="View opt steps")
    parser.add_argument('--bench', action='store_true', help="Enable benchmarking")
    args = parser.parse_args()
    
    # Call solve
    from kinematics.util import jax_cache_on
    jax_cache_on()
    # jax.config.update("jax_debug_nans", True)

    sim = Simulation(num_blocks=args.num_blocks)
    params = SpasmParams()
    key = jax.random.key(int(time.time()))

    if args.num_blocks == 3:
        # 3-tetris
        params.sampling_batch = 512
        params.opt_batch = 64
        params.opt_steps = 25
        
    elif args.num_blocks == 5:
        # 5-tetris
        params.sampling_batch = 4096
        params.opt_batch = 256
        params.opt_steps = 25
        params.cost_thresh = 0.42
        
    elif args.num_blocks == 8:
        # 8-tetris
        params.sampling_batch = 2048 * 128
        params.opt_batch = 256
        params.opt_steps = 50
        params.cost_thresh = 0.66
        
    else:
        raise ValueError('num_blocks must be one of 3, 5, 10.')

    if args.noopt:
        params.noopt = True
        
    if args.viewopt:
        params.opt_batch = 1
        params.viewopt = True
    
    # Solve
    if not args.cached:
        if args.bench:
            # Warmup
            opt_particle = solve(params, sim, key)
            opt_particle.block_until_ready()

        start_time = time.perf_counter()
        for _ in range(1 if not args.bench else 10):
            opt_particle = solve(params, sim, key)
        opt_particle.block_until_ready()
        total_time = time.perf_counter() - start_time
        
        if args.bench:
            # Stats make it tempting to prematurely optimize
            print(f"Total time: {total_time * 1000 / 10:.2f} ms")
        
        # Save solutions
        os.makedirs('saved', exist_ok=True)
        jnp.save('saved/tetris.npy', opt_particle)

    print("Loading...")
    opt_particle = jnp.load('saved/tetris.npy')
    print("Loaded!")
    
    cost_jit = jax.jit(cost, static_argnames=('params', 'sim'))
    print('Best error', cost_jit(params, sim, opt_particle))
    
    # for _ in range(100):
    #     opt_particle = solve(params, sim, key)
    #     print('Best error', cost_jit(params, sim, opt_particle))
    #     key, _ = jax.random.split(key)
    
    # Show best solution
    sim.set_state(opt_particle)
    sim.render()
    time.sleep(2)
    sim.reset_state()
    
    
    # Shape (T, dof)
    # gen_traj_jit = jax.jit(generate_trajectory, static_argnames=['params', 'sim'])
    # traj = generate_trajectory(params, sim, opt_particles[0])
    
    # sim.animate_trajectory(traj)
    # sim.reset_state()
    # sim.animate_trajectory(traj)
    
    
    # Visualize the final lowest error state
    # Show the blocks in theri final positions, and then loop the arm position for each (while true)
    # import time

    # for i, e in enumerate(opt_errors):
    #     print(f'    {i}: {e.item()}')

    # for i in range(100000000):
    #     best_block_poses = opt_particles[i % 1] #len(opt_particles)]
    #     sim.set_state(best_block_poses)

    #     # for q in best_q:
    #     #     # sim.set_robot_pose(q)
    #     sim.render()
    #     time.sleep(0.5)