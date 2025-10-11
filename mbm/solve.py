import jax
import jax.numpy as jnp
import jaxlie
import time
from functools import partial

from kinematics.kinematics import fk
from mbm.vis import MBMVisualizer, all_retractions
import argparse

class TrajOptParams:
    def __init__(self):
        self.max_steps = 5
        self.init_lr = 0.6
        
        self.pen_weight = 0.2  
        self.short_weight = 0.1
        
        self.viewopt = False
        
        self.sample_batch = 10000 # 16
        self.opt_batch = 2

# Margin around each obstacle
margin = 0.01

def sample_prolapsed_hs(key, a, b, num_samples, radius):
    """Samples in an ellipse where the ends are a b"""
    distance = jnp.linalg.norm(b - a)
    string_distance = jnp.hypot(distance/2, radius) * 2
    longwise_radius = (string_distance - distance) / 2
    
    # Sample in sphere at (a + b) / 2
    dof = a.shape[0]
    samples_unit_ball = jax.random.ball(key, dof, shape=(num_samples,)) * radius

    # Then in direction of (b - a) scale by longwise_radius / radius
    major_axis_dir = b - a
    major_scale = major_axis_dir * (longwise_radius / radius)
    
    return (a + b) / 2 + samples_unit_ball * major_scale

def mid_init(params, viz, mid):
    jojo_part1 = jnp.linspace(viz.q_start, mid, 7)
    jojo_part2 = jnp.linspace(mid, viz.q_goal, 15)[1:-1]
    q_traj = jnp.concatenate([jojo_part1, jojo_part2])
    
    # Clip to joint limits
    q_traj = jnp.clip(q_traj, viz.qmins, viz.qmaxes)
    
    return q_traj

# def retract_init(params, viz):
#     p = jnp.pi

#     s1 = jnp.array([0.0, -1.9, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
#     # s2 = jnp.array([0.0, -1.7, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
    
#     jojo_part1 = jnp.linspace(viz.q_start, s1, 10)[1:]
#     # jojo_part2 = jnp.linspace(s1, s2, 3)[1:]
#     jojo_part5 = jnp.linspace(s1, viz.q_goal, params.interps - 3)[1:-1]
#     q_traj = jnp.concatenate([jojo_part1, jojo_part5])

#     # return jnp.linspace(viz.q_start, viz.q_goal, params.interps + 6)[1:-1]
#     return q_traj

def sphere_box_penetration(sphere_pos, sphere_rad, box_pos, box_quat_xyzw, box_dims, margin):
    """
    Computes penetration depth between a sphere and a box.
    
    Args:
        sphere_pos (jnp.ndarray): Sphere center position (3,).
        sphere_rad (float): Sphere radius.
        box_pos (jnp.ndarray): Box center position (3,).
        box_quat_xyzw (jnp.ndarray): Box orientation as a quaternion [x, y, z, w] (4,).
        box_dims (jnp.ndarray): Box dimensions [dx, dy, dz] (3,).

    Returns:
        float: Penetration depth. Positive if penetrating, 0 otherwise.
    """
    # Create SO3 object for box rotation and its inverse
    rot_box = jaxlie.SO3.from_quaternion_xyzw(box_quat_xyzw)
    rot_box_inv = rot_box.inverse()

    # Transform sphere center to box's local frame
    sphere_pos_in_box_frame = rot_box_inv @ (sphere_pos - box_pos)

    # Box half dimensions
    half_dims = box_dims / 2.0

    # Find the closest point on the AABB (in box frame) to the sphere center
    closest_point_in_box_frame = jnp.clip(sphere_pos_in_box_frame, -half_dims, half_dims)

    # Distance from sphere center to the box surface
    # Add a small epsilon for gradient stability when the point is inside the box.
    distance = jnp.linalg.norm(sphere_pos_in_box_frame - closest_point_in_box_frame + 1e-7)

    # Penetration is the difference between radius and distance, clamped at 0
    penetration = jnp.maximum(0.0, sphere_rad - distance + margin)
    return penetration

def sphere_cylinder_penetration(sphere_pos, sphere_rad, cyl_pos, cyl_dims, margin):
    """
    Computes penetration depth between a sphere and an upright cylinder (Z-axis up).
    Cylinders in MBM are oriented with height along their local Z axis, but are placed upright in the world.
    The provided dimensions are [height, radius].

    Args:
        sphere_pos (jnp.ndarray): Sphere center position (3,).
        sphere_rad (float): Sphere radius.
        cyl_pos (jnp.ndarray): Cylinder center position (3,).
        cyl_dims (jnp.ndarray): Cylinder dimensions [height, radius] (2,).

    Returns:
        float: Penetration depth. Positive if penetrating, 0 otherwise.
    """
    cyl_height, cyl_rad = cyl_dims[0], cyl_dims[1]
    
    # Vector from cylinder center to sphere center
    delta = sphere_pos - cyl_pos
    
    # Distance in the XY plane (radial distance)
    dist_xy = jnp.linalg.norm(delta[:2]) 
    
    # Vertical distance from cylinder center
    dist_z = delta[2]

    # Find the closest point on the cylinder to the sphere center
    # 1. Clamp the vertical distance to be within the cylinder's height
    closest_z = jnp.clip(dist_z, -cyl_height / 2.0, cyl_height / 2.0)
    
    # 2. Clamp the radial distance to be on the cylinder's radius
    # We find the point on the cylinder's axis and then move out radially.
    # A small epsilon is added to avoid division by zero if the sphere is on the axis.
    radial_dir = delta[:2] / (dist_xy + 1e-6)
    closest_xy = radial_dir * jnp.minimum(dist_xy, cyl_rad)

    # Combine to get the closest point on the cylinder surface
    closest_point_on_cyl = cyl_pos + jnp.array([closest_xy[0], closest_xy[1], closest_z])

    # Distance from sphere center to the cylinder surface
    # Add a small epsilon for gradient stability.
    distance = jnp.linalg.norm(sphere_pos - closest_point_on_cyl + 1e-8)
    
    # Penetration is the difference between radius and distance, clamped at 0
    penetration = jnp.maximum(0.0, sphere_rad - distance + margin)
    return penetration

def total_penetration(q: jnp.ndarray, viz: MBMVisualizer, margin) -> float:
    """
    Computes the total penetration of the robot arm with the scene obstacles.

    Args:
        q (jnp.ndarray): Robot joint configuration (9,).
        viz (MBMVisualizer): Visualizer object containing the scene.

    Returns:
        float: Sum of all penetration depths.
    """
    # Get robot arm spheres
    robot_spheres_pos, robot_spheres_radii = fk(q)

    # --- vmap functions for batch computation ---
    
    # vmap over robot spheres for a single box
    one_sphere_multi_box = jax.vmap(sphere_box_penetration, in_axes=(None, None, 0, 0, 0, None))
    # vmap over all boxes
    many_spheres_multi_box = jax.vmap(one_sphere_multi_box, in_axes=(0, 0, None, None, None, None))

    # vmap over robot spheres for a single cylinder
    one_sphere_multi_cyl = jax.vmap(sphere_cylinder_penetration, in_axes=(None, None, 0, 0, None))
    # vmap over all cylinders
    many_spheres_multi_cyl = jax.vmap(one_sphere_multi_cyl, in_axes=(0, 0, None, None, None))

    # --- Calculate penetrations ---
    
    # Box penetrations (#spheres, #boxes)
    box_pens = many_spheres_multi_box(
        robot_spheres_pos, robot_spheres_radii, viz.box_poses, viz.box_quats, viz.box_dims, margin
    )
    box_pen = jnp.abs(box_pens**1).sum()

    # Cylinder penetrations (#spheres, #cylinders)
    cyl_pens = many_spheres_multi_cyl(
        robot_spheres_pos, robot_spheres_radii, viz.cylinder_poses, viz.cylinder_dims, margin
    )
    cyl_pen = jnp.abs(cyl_pens**1).sum()
    
    # jax.debug.print("Box pen: {b}, Cylinder pen: {c}", b=box_pen, c=cyl_pen)
    
    return box_pen + cyl_pen

traj_penetration = jax.vmap(total_penetration, in_axes=(0, None, None))

def pen_cost(params: TrajOptParams, viz: MBMVisualizer, q_inner):
    """
    Computes the total cost for a trajectory.
    q_traj: (T, 9)
    """
    q_traj = jnp.vstack([viz.q_start, q_inner, viz.q_goal])
        
    # Penetration cost
    penetration_per_step = traj_penetration(q_traj, viz, margin)
    total_pen_cost = jnp.sum(penetration_per_step) * params.pen_weight
    
    # jax.debug.print("Pen cost: {p}", p=total_pen_cost)
    
    return total_pen_cost

def short_cost(params: TrajOptParams, viz, q_inner):
    q_traj = jnp.vstack([viz.q_start, q_inner, viz.q_goal])
    
    # Shortness cost (encourage smooth, short paths)
    q_diff = jnp.diff(q_traj, axis=0)
    shortness_cost = (jnp.maximum(jnp.abs(q_diff), 0.2)).sum() * params.short_weight
    
    # jax.debug.print("Short cost: {s}", s=shortness_cost)
    
    return shortness_cost

#                   1    2    3    4    5    6    7    8    9
biases = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0])

def optimize_traj_one(params: TrajOptParams, viz: MBMVisualizer, q_traj_initial):
    """
    Optimizes a trajectory to minimize penetration and length.
    The start and end points of the trajectory are fixed.
    """
    
    def callback(i, inners, pene):
        print('Step', i, 'penetration', pene)
        qs = jnp.vstack([viz.q_start, inners, viz.q_goal])
        viz.render_traj(qs, skip=1, hand_only=False)
        time.sleep(0.7)
        
    if params.viewopt:
        jax.debug.callback(callback, -1, q_traj_initial, 9999)

    pen_grad_fn = jax.grad(pen_cost, argnums=2)
    short_grad_fn = jax.grad(short_cost, argnums=2)

    def opt_step(items):
        q_vars_current, _, i = items

        pen_grad = pen_grad_fn(params, viz, q_vars_current)
        pen_grad *= biases  # Less on last 3 joints (wrist)
        short_grad = short_grad_fn(params, viz, q_vars_current)
        
        # jax.debug.print('Pen grad norm {:.5f}, Short grad norm {:.5f}', jnp.linalg.norm(pen_grad), jnp.linalg.norm(short_grad))
        
        lr = params.init_lr * 1.0 / jnp.exp(i / params.max_steps * 3)
        
        grad_sum = pen_grad + short_grad
        grad_sum = jnp.clip(grad_sum, -0.2, 0.2)
        q_vars_current -= lr * grad_sum

        q_vars_current = jnp.clip(q_vars_current, viz.qmins, viz.qmaxes)
        
        penetration = traj_penetration(q_vars_current, viz, 0).max()
        # jax.debug.print('pen max {:.4f}', penetration)
        
        if params.viewopt:
            jax.debug.callback(callback, i, q_vars_current, penetration)
        
        return q_vars_current, penetration, i + 1
    
    def cond_fn(val):
        _, penetration, i = val
        satisfied = penetration < 1e-6
        # return ~all_grads_zero & (i < params.steps) 
        return (~satisfied) & (i < params.max_steps)

    # Run the optimization loop
    q_vars_final, penetration, steps = jax.lax.while_loop(cond_fn, opt_step, (q_traj_initial, 9999, 0))

    
    return q_vars_final, penetration, steps

def optimize_traj_(key, params: TrajOptParams, viz: MBMVisualizer):
    mid_init_vmap = jax.vmap(mid_init, in_axes=(None, None, 0))
    
    # 1. Sample a batch of free configurations (sample_batch, 9)
    q_sampled_hs1 = sample_prolapsed_hs(key, viz.q_start, viz.q_goal, params.sample_batch, 0.3)
    q_sampled_hs2 = sample_prolapsed_hs(key, viz.q_start, viz.q_goal, params.sample_batch, 1.5)
    q_sampled_hs3 = sample_prolapsed_hs(key, viz.q_start, viz.q_goal, params.sample_batch, 3.1)
    
    base = viz.q_start.at[1].set(0 * jnp.sign(viz.q_start[1])) # 2.9671
    attempt4 = base.at[0].set( 2.9671)
    attempt6 = base.at[0].set(-2.9671)
    
    cricket = viz.q_start.at[0].add(0) \
                         .at[1].set(-0.8 * jnp.sign(viz.q_start[1])) \
                         .at[3].set(3.14 * jnp.sign(viz.q_start[3])) \
                         .at[5].set(3.14 * jnp.sign(viz.q_start[5]))

    # cricket = viz.q_start.at[0].add(-3.14 * jnp.sign(viz.q_start[0])) \
    #                      .at[1].set(1.0 * jnp.sign(viz.q_start[1])) \
    #                      .at[3].set(0.0 * jnp.sign(viz.q_start[3]))

    # q_sampled_hs = jnp.concatenate([q_sampled_hs, 
    #                                 attempt4[None], attempt6[None],
    #                                 ], axis=0)
    uniform_samps = jax.random.uniform(key, (params.sample_batch, 9,), minval=viz.qmins, maxval=viz.qmaxes)
    
    q_sampled_hs = uniform_samps
    
    # q_sampled_hs = jnp.concatenate([q_sampled_hs1, q_sampled_hs2, q_sampled_hs3], axis=0)
    
    q_inits = mid_init_vmap(params, viz, q_sampled_hs)
    
    # 2. Score the samples
    def score_q(q_traj, viz):
        pen = traj_penetration(q_traj, viz, 0).max()
        return pen 
    def traj_length(q_traj):
        q_traj_full = jnp.vstack([viz.q_start, q_traj, viz.q_goal])
        diffs = jnp.diff(q_traj_full, axis=0)
        return jnp.sum(jnp.linalg.norm(diffs, axis=1))

    vmapped_score_q = jax.vmap(score_q, in_axes=(0, None))
    vmapped_traj_length = jax.vmap(traj_length, in_axes=(0,))
    scores = vmapped_score_q(q_inits, viz)
    lengths = vmapped_traj_length(q_inits)

    scores = jnp.where(scores < 1e-4, 0.01 * lengths, scores + 10)

    # 3. Select the best `opt_batch` samples
    _, best_indices = jax.lax.top_k(-scores, params.opt_batch)
    best_sample_trajs = q_inits[best_indices]
    
    # jax.debug.print("Best sample scores: {s}", s=scores[best_indices])

    # 4. Batch-optimize trajectories using the best samples    
    optimize_traj_vmap = jax.vmap(optimize_traj_one, in_axes=(None, None, 0))
    
    # Returns (trajs, penetrations, steps) for the whole batch
    q_trajs_opt, final_penetrations, steps_taken = optimize_traj_vmap(params, viz, best_sample_trajs)

    # 5. Find the best trajectory from the optimized batch
    best_traj_idx = jnp.argmin(final_penetrations)
    
    best_traj = q_trajs_opt[best_traj_idx]
    best_pene = final_penetrations[best_traj_idx]
    steps = steps_taken[best_traj_idx]
    
    # Reconstruct the final trajectory
    best_traj_full = jnp.vstack([viz.q_start, best_traj, viz.q_goal])
        
    return best_traj_full, best_pene, steps

@partial(jax.jit, static_argnames=('params'))
def optimize_traj(key, params: TrajOptParams, viz: MBMVisualizer):
    # repeat until success
    def cond_fun(val):
        _, best_pene, _, _ = val
        # Continue if penetration is greater than a small threshold
        return best_pene > 1e-6

    def body_fun(val):
        key, _, _, _ = val
        key, subkey = jax.random.split(key)
        
        # Run one batch of optimizations
        traj, pene, steps = optimize_traj_(subkey, params, viz)
        
        # The state of the loop is (key, best_penetration, best_trajectory, steps_taken)
        return key, pene, traj, steps

    # Initialize the loop state
    # The initial trajectory shape must match the output of optimize_traj_
    # We can get this by running a dummy initialization
    dummy_mid = jnp.zeros_like(viz.q_start)
    q_inner_dummy = mid_init(params, viz, dummy_mid)
    initial_traj = jnp.vstack([viz.q_start, q_inner_dummy, viz.q_goal])
    
    initial_val = (key, jnp.inf, initial_traj, -1)

    # Run the while loop until a satisfactory trajectory is found
    final_key, final_pene, final_traj, final_steps = jax.lax.while_loop(
        cond_fun, body_fun, initial_val
    )

    return final_traj, final_pene, final_steps

if __name__ == "__main__":
    from kinematics.util import jax_cache_on
    jax_cache_on()
    
    # Check nans
    # jax.config.update("jax_debug_nans", True)
    
    parser = argparse.ArgumentParser(description="Trajectory optimization for MBM.")
    parser.add_argument("--problem", type=str, default="bookshelf_thin", help="Problem name")
    parser.add_argument("--idx", type=int, default=2, help="Scene index")
    parser.add_argument("--viewopt", action="store_true", help="Visualize optimization steps")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    args = parser.parse_args()
    
    viz = MBMVisualizer()
    viz.set_scene(args.problem, idx=args.idx)
    viz.render_env()
    
    print(f"Loaded problem '{ args.problem}' with index {args.idx}")

    params = TrajOptParams()
    params.viewopt = args.viewopt
    key = jax.random.key(2)
    
    # q_sampled_hs1 = sample_prolapsed_hs(key, viz.q_start, viz.q_goal, 100, 5.0)
    # for q in q_sampled_hs1:
    #     viz.render_robot(q)

    # if args.problem.startswith('cage'):
    #     params.max_steps = 30
    
    if args.bench:
        # Warmup
        optimize_traj(key, params, viz.get_problem_data())
        
    
    # --- Optimize the trajectory ---
    start_time = time.perf_counter()
    q_traj_optimized, best_pene, steps = optimize_traj(key, params, viz.get_problem_data())
    q_traj_optimized.block_until_ready()
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    
    if args.bench:
        print(f"Opt took {elapsed_ms:.1f} ms and best penetration {best_pene:.4f} and {steps} steps.")
    else:
        print(f"Best penetration {best_pene} and {steps} steps.")
        
    viz.vis['traj'].delete()
    viz.render_traj(q_traj_optimized, hand_only=False, skip=1, opacity=0.1)
    # while True:
    #     for q in q_traj_optimized:
    #         viz.render_robot(q)
    #         time.sleep(0.1)
    #     time.sleep(0.7)
