import time
import jax
import jax.numpy as jnp
from kinematics.util import purple_hex, white_hex
from kinematics.kinematics import fk_batched, get_ee_pose, get_hand_pose, transform_to_position, get_joint_limits
from viz import init_vis, draw_spheres

def main():
    """
    An example of how to use the fk library.
    """
    from kinematics.util import jax_cache_on
    jax_cache_on()
    
    init_vis()

    mins, maxes = get_joint_limits()
    print('Loaded joint limits min:', mins, '\n                    max:', maxes)
    
    key = time.time_ns()
    q0 = jax.random.uniform(jax.random.PRNGKey(key), shape=mins.shape, minval=mins, maxval=maxes)
    q1 = jax.random.uniform(jax.random.PRNGKey(key + 1), shape=mins.shape, minval=mins, maxval=maxes)
    qs = jnp.linspace(q0, q1, num=10)
    
    positions, radii = fk_batched(qs)

    # Draw the spheres for the last configuration in the trajectory
    for i, (ps, rs) in enumerate(zip(positions, radii)):
        draw_spheres(ps, rs, color=white_hex(), prefix=f"link_{i}_")

    # --- Test single link pose retrieval ---
    ee_pose_batched = jax.jit(jax.vmap(get_ee_pose))
    hand_pose_batched = jax.jit(jax.vmap(get_hand_pose))
    
    transforms = hand_pose_batched(qs)
    positions = transform_to_position(transforms)
    radii = jnp.array([0.01] * positions.shape[0])

    draw_spheres(positions, radii, color=purple_hex(), prefix='hand_')


if __name__ == '__main__':
    main()
