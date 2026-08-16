import time
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from scipy.spatial import transform
from typing import List, Literal

import jax
import jax.numpy as jnp
import jaxlie

from kinematics.kinematics import fk, get_ee_pose, get_joint_limits
from kinematics.viz import draw_spheres
from spasm.conversions import grasp_to_q, gray_hex, rgb_to_hex, yaw_to_quat_xyz


unit_quat = [1.0, 0.0, 0.0, 0.0]

def create_tetris_spheres(shape: str, sph_radius: float) -> jnp.ndarray:
    """
    Creates an array of spheres for a given Tetris shape.

    Args:
        shape: The shape of the Tetris block ('L' or 'O').
        sph_radius: The radius of the spheres.

    Returns:
        An array of spheres, where each row is [x, y, z, radius]. Shape: (N, 4)
    """
    _shape_coords = {
        "L": jnp.array([(0, 0, 0), (0, 1, 0), (0, -1, 0), (1, -1, 0)]),
        "O": jnp.array([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]),
    }
    coords = _shape_coords[shape]
    num_coords = coords.shape[0]
    
    spheres = jnp.zeros((num_coords + 2, 4))
    
    # Set coordinates for the main body
    spheres = spheres.at[:num_coords, :3].set(coords * sph_radius * 2)
    spheres = spheres.at[:num_coords, 3].set(sph_radius)
    
    # Add stick to grasp onto
    stick_spheres = jnp.array([
        [0.0, 0.0, -sph_radius * 1.25, sph_radius / 2],
        [0.0, 0.0, -sph_radius * 2, sph_radius / 2]
    ])
    spheres = spheres.at[num_coords:, :].set(stick_spheres)
    
    z_offset = -spheres[-1, 2]
    spheres = spheres.at[:, 2].add(z_offset)
    return spheres

def create_walls(
    goal_pose: List[float], goal_dims: List[float], wall_height: float, wall_thickness: float
) -> jnp.ndarray:
    """Creates four walls around a goal region.

    Args:
        goal_pose: The pose of the goal region [x, y, z].
        goal_dims: The dimensions of the goal region [dx, dy, dz].
        wall_height: The height of the walls.
        wall_thickness: The thickness of the walls.

    Returns:
        A JAX array of shape (4, 6) representing the walls as AABBs (x1, y1, z1, x2, y2, z2).
    """
    cx, cy, cz = goal_pose
    cdx, cdy, _ = goal_dims

    # Wall 1 (positive y)
    w1_x1 = cx - cdx / 2
    w1_y1 = cy + cdy / 2
    w1_z1 = cz
    w1_x2 = cx + cdx / 2
    w1_y2 = cy + cdy / 2 + wall_thickness
    w1_z2 = cz + wall_height

    # Wall 2 (negative y)
    w2_x1 = cx - cdx / 2
    w2_y1 = cy - cdy / 2 - wall_thickness
    w2_z1 = cz
    w2_x2 = cx + cdx / 2
    w2_y2 = cy - cdy / 2
    w2_z2 = cz + wall_height

    # Wall 3 (negative x)
    w3_x1 = cx - cdx / 2 - wall_thickness
    w3_y1 = cy - cdy / 2
    w3_z1 = cz
    w3_x2 = cx - cdx / 2
    w3_y2 = cy + cdy / 2
    w3_z2 = cz + wall_height

    # Wall 4 (positive x)
    w4_x1 = cx + cdx / 2
    w4_y1 = cy - cdy / 2
    w4_z1 = cz
    w4_x2 = cx + cdx / 2 + wall_thickness
    w4_y2 = cy + cdy / 2
    w4_z2 = cz + wall_height

    walls = jnp.array([
        [w1_x1, w1_y1, w1_z1, w1_x2, w1_y2, w1_z2],
        [w2_x1, w2_y1, w2_z1, w2_x2, w2_y2, w2_z2],
        [w3_x1, w3_y1, w3_z1, w3_x2, w3_y2, w3_z2],
        [w4_x1, w4_y1, w4_z1, w4_x2, w4_y2, w4_z2],
    ])
    return walls

def _block_pose_to_spheres(spheres, pose):
    '''
    spheres:  (6, 4) (x y z radius)
    pose:     (7,) [x y z yaw]
    Returns:
        transformed spheres: (6, 4)
    '''
    assert spheres.shape == (6, 4), f"spheres should be of shape (6, 4), got {spheres.shape}"
    assert pose.shape == (4,), f"pose should be of shape (4,), got {pose.shape}"
    
    pose = yaw_to_quat_xyz(pose) 
    
    sphere_pos = spheres[:, :3] # (6, 3)
    sphere_r = spheres[:, 3, None] # (6, 1)
    
    pos = pose[:3] # (3,)
    rot = jaxlie.SO3.from_quaternion_xyzw(pose[3:])
    
    # Apply: (6, 3)
    trans_position = rot.apply(sphere_pos) + pos
    return jnp.concatenate([trans_position, sphere_r], axis=-1)

def block_pose_to_spheres(sim, block_poses):
    '''
    Return the [x y z yaw] of each sphere of each block
    block_poses: (num_blocks, 6, 4)
    '''
    assert block_poses.shape == (sim.num_blocks, 4), f"block_poses should be of shape ({sim.num_blocks}, 4), got {block_poses.shape}"
    return jax.vmap(_block_pose_to_spheres, in_axes=(0, 0)) \
                    (sim.block_spheres, block_poses)

class Simulation:
    def __init__(self, num_blocks: Literal[1, 3, 5, 10, 20]):
        """
        Initializes the simulation environment with Tetris blocks.

        Args:
            num_blocks: The number of blocks to include in the environment.
        """
        self.num_blocks = num_blocks
        
        sph_radius: float = 0.03
        wall_height: float = 0.045
        wall_thickness: float = 0.015

        self.vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        self.vis.delete()

        self.table_dims = [0.8, 1.5, 0.02]
        self.table_pose = [0.30, 0.0, -0.011, *unit_quat]
        self.table_color = [255, 255, 255]

        L_sphs = create_tetris_spheres("L", sph_radius)
        O_sphs = create_tetris_spheres("O", sph_radius)
        L_block_z = (L_sphs[:, 2] + L_sphs[:, 3]).max() - (L_sphs[:, 2] - L_sphs[:, 3]).min() - 1e-2
        O_block_z = (O_sphs[:, 2] + O_sphs[:, 3]).max() - (O_sphs[:, 2] - O_sphs[:, 3]).min() - 1e-2
        
        x_offset = -0.1
        block_poses = [
            [0.50,  0.35, O_block_z, 0],
            [0.15, -0.6, L_block_z, 0],
            [0.00,  0.6, L_block_z, 0],
            [0.15,  0.6, L_block_z, 0],
            [0.00, -0.6, L_block_z, 0],
            # Second row
            [0.50, -0.3, O_block_z, 0],
            [0.50, -0.1, O_block_z, 0],
            [0.50,  0.1, O_block_z, 0],
        ]
        
        assert jnp.isclose(L_block_z, O_block_z).all(), "Block z offsets should be the same."
        self.block_z = L_block_z # Global Z to grasp blocks

        all_block_spheres = [O_sphs, O_sphs, O_sphs, L_sphs, L_sphs,
                             L_sphs, L_sphs, O_sphs] #, L_sphs, L_sphs]
        # self.block_colors = [[255, 255, 186], [255, 186, 201], [186, 201, 255], [186, 255, 201], [102, 186, 232],
        #                      [255, 255, 186], [255, 186, 201], [186, 201, 255],]
        self.block_colors = [0xe81416, 0xffa500, 0xfaeb36, 0x79c314, 0x487de7, 0x87369d, 0x5eb40d, 0xffa500]

        def hex_to_rgb(h):
            return ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)

        self.block_colors = [hex_to_rgb(c) if isinstance(c, int) else c for c in self.block_colors]
            
        # Make bolock colors a bit pastel
        pastel_factor = 0.6
        self.block_colors = [
            [int(v * pastel_factor + 255 * (1 - pastel_factor)) for v in c]
            for c in self.block_colors
        ]
        
        indices = list(range(self.num_blocks))
        if self.num_blocks == 2:
            indices = [1, 2]
        
        # (num_blocks, 6, 4)
        self.block_spheres = jnp.array([all_block_spheres[i] for i in indices])
        self.num_blocks = self.block_spheres.shape[0]
        self.num_spheres = self.block_spheres.shape[1]
        self.block_poses = [jnp.array(block_poses[i]) for i in indices]
        self.block_poses_original = self.block_poses.copy()
        
        self.block_poses_matrix = {}

        diameter = sph_radius * 2
        
        match self.num_blocks:
            case 1:
                goal_wideness = 2
                goal_tallness = 2
            case 3:
                goal_wideness = 6
                goal_tallness = 2
            case 5:
                goal_wideness = 10
                goal_tallness = 2
            case 8:
                goal_wideness = 16
                goal_tallness = 2
            case _: # 10 or 20
                raise ValueError("num_blocks must be one of 1, 3, 5, 9.")

        buffer = sph_radius * 1.0
        goal_wideness = goal_wideness * diameter + buffer
        goal_tallness = goal_tallness * diameter + buffer

        self.goal_dims = jnp.array([goal_tallness, goal_wideness, 0.01])
        self.goal_position = jnp.array([0.3, 0.0, -0.005])
        self.goal_color = [255, 255, 255]

        self.goal_walls = create_walls(self.goal_position, self.goal_dims, wall_height, wall_thickness)
        self.wall_color = [255, 255, 255]

        self.q = Simulation.get_neutral_pose()
        self.qmins, self.qmaxes = get_joint_limits()

    def set_robot_pose(self, q: jnp.ndarray):
        """Sets the robot's joint configuration.
        
        Args:
            q: Joint configuration for the robot. Shape (9,).
        """
        if q.shape[0] == 7:
            q = jnp.pad(q, (0, 2))
        assert q.shape == (9,), f"q should be of shape (9,), got {q.shape}"
        self.q = q
    
    @staticmethod
    def get_neutral_pose():
        # return jnp.array([0., -jnp.pi/4, 0., -3*jnp.pi/4, 0., jnp.pi/2, jnp.pi/4, 0., 0.])
        return jnp.array([0., -jnp.pi/4, 0., -2*jnp.pi/4, 0., jnp.pi/2, jnp.pi/4, 0., 0.])

    def set_state(self, block_poses: jnp.ndarray):
        """Sets the initial state of the robot and blocks.
        
        Args:
            block_poses: An array of poses for each block. Shape (num_blocks, 7).
        """
        assert block_poses.shape == (self.num_blocks, 4), f"block_poses should be of shape ({self.num_blocks}, 4), "\
                                                          f"got {block_poses.shape}"
        self.block_poses = block_poses
    
    def set_one_state(self, block_idx: int, block_pose: jnp.ndarray):
        """Sets the state of a single block.
        
        Args:
            block_idx: Index of the block to set.
            block_pose: Pose for the block. Shape (4,).
        """
        assert 0 <= block_idx < self.num_blocks, f"block_idx should be in [0, {self.num_blocks}), got {block_idx}"
        assert block_pose.shape == (4,), f"block_pose should be of shape (4,), got {block_pose.shape}"
        self.block_poses = self.block_poses.at[block_idx].set(block_pose.copy())
        
    def reset_state(self):
        """Resets the state of the robot and blocks to the original configuration."""
        self.block_poses = self.block_poses_original
        self.q = Simulation.get_neutral_pose()
        
    def step(self):
        """An empty step function."""
        pass

    def render(self):
        """Renders the environment in meshcat."""
        # Render table
        self.vis["table"].set_object(g.Box(self.table_dims), g.MeshPhongMaterial(color=rgb_to_hex(*self.table_color)))
        self.vis["table"].set_transform(tf.translation_matrix(self.table_pose[:3]))

        # Render goal region
        goal_dims = [float(v) for v in self.goal_dims]
        goal_position = [float(v) for v in self.goal_position]
        self.vis["goal"].set_object(g.Box(goal_dims), g.MeshLambertMaterial(color=rgb_to_hex(*self.goal_color)))
        self.vis["goal"].set_transform(tf.translation_matrix(goal_position))

        # Render walls
        for i, wall_aabb in enumerate(self.goal_walls):
            x1, y1, z1, x2, y2, z2 = [float(v) for v in wall_aabb]
            dims = [x2 - x1, y2 - y1, z2 - z1]
            pose = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]
            self.vis[f"wall_{i}"].set_object(g.Box(dims), g.MeshLambertMaterial(color=rgb_to_hex(*self.wall_color)))
            self.vis[f"wall_{i}"].set_transform(tf.translation_matrix(pose))
        
        assert np.isfinite(self.block_poses).all(), f"Invalid block_poses: {self.block_poses}"
        
        # Shape (num_blocks, 6, 4)
        # transformed_spheres = block_pose_to_spheres(self, jnp.array(self.block_poses))
        
        # for i, (spheres, color) in enumerate(zip(transformed_spheres, self.block_colors)):
        #     spheres_np = np.asarray(spheres) # (6, 4)
                        
        #     for j, sphere_data in enumerate(spheres_np):
        #         sphere_data = [float(v) for v in sphere_data]
        #         sphere_pos, sphere_radius = sphere_data[:3], sphere_data[3]
                
        #         self.vis[f"block_{i}_{j}"].set_object(g.Sphere(sphere_radius), g.MeshPhongMaterial(color=rgb_to_hex(*color)))
        #         self.vis[f"block_{i}_{j}"].set_transform(tf.translation_matrix(sphere_pos))
        
        for i in range(self.num_blocks):
            
            if i in self.block_poses_matrix:
                transform_matrix = self.block_poses_matrix.pop(i)
                assert transform_matrix.shape == (4, 4), f"block_poses_matrix[{i}] should be of shape (4, 4), got {transform_matrix.shape}"
                
                spheres_h = jnp.pad(self.block_spheres[i, :, :3], ((0, 0), (0, 1)), constant_values=1.0)
                transformed_spheres_h = spheres_h @ transform_matrix.T
                transformed_pos = transformed_spheres_h[:, :3]
                
                transformed_spheres = jnp.concatenate([transformed_pos, self.block_spheres[i, :, 3, None]], axis=-1)
            else:
                transformed_spheres = _block_pose_to_spheres(self.block_spheres[i], self.block_poses[i])
                
            spheres_np = np.asarray(transformed_spheres)
                        
            for j, sphere_data in enumerate(spheres_np):
                sphere_data = [float(v) for v in sphere_data]
                sphere_pos, sphere_radius = sphere_data[:3], sphere_data[3]
                
                self.vis[f"block_{i}_{j}"].set_object(g.Sphere(sphere_radius), g.MeshPhongMaterial(color=rgb_to_hex(*self.block_colors[i])))
                self.vis[f"block_{i}_{j}"].set_transform(tf.translation_matrix(sphere_pos))

        # Render robot
        robot_pos, robot_radii = fk(self.q)
        draw_spheres(robot_pos, robot_radii, color=gray_hex(), prefix='robot_', viss=self.vis)

    def animate_trajectory(self, traj):
        '''
        traj = [traj, "pick 0", traj, "place 0", ..., "place {N}"]
        '''
        goto_trajs, goto_traj_qs, place_trajs, place_traj_qs = traj
        N = self.num_blocks
        
        assert goto_traj_qs.shape[0] == N, f"Expected goto_traj_qs to have shape ({N}, interps, 9), got {goto_traj_qs.shape}"
        assert place_traj_qs.shape[0] == N, f"Expected place_traj_qs to have shape ({N}, interps, 9), got {place_traj_qs.shape}"

        def animate_segment(block_poses, q_poses, holding_idx=None):
            for block_pose, q in zip(block_poses, q_poses):
                if holding_idx is not None:
                    q = q.at[-2:].set(0.06) # Closed
                else:
                    q = q.at[-2:].set(0.0)  # Open
                self.set_robot_pose(q)
                if holding_idx is not None:
                    self.block_poses[holding_idx] = block_pose
                self.render()
        
        self.reset_state()
        
        for block_idx in range(N):
            animate_segment(goto_trajs[block_idx], goto_traj_qs[block_idx], None)
            animate_segment(place_trajs[block_idx], place_traj_qs[block_idx], block_idx)
    
    
    def draw_trajs(self, q_trajs: jnp.ndarray):
        '''
        Draws the trajectories of the blocks as lines and spheres at the waypoints.
        trajs: (num_blocks, T, 7)
        '''
        assert q_trajs.shape[0] == self.num_blocks * 2 - 1, f"Expected q_trajs to have shape ({self.num_blocks * 2 + 1}, T, 9), got {q_trajs.shape}"
        for i in range(self.num_blocks * 2 - 1):
            points = []
            for t in range(q_trajs.shape[1]):
                pose = q_trajs[i, t]
                pose = jnp.pad(pose, (0, 2))
                ee_pose = get_ee_pose(pose)[0:3, 3]
                points.append(ee_pose)
            
            pointsa = np.array(points).T # Shape (3, T)
            self.vis[f"traj/line_{i}"].set_object(g.Line(g.PointsGeometry(pointsa), 
                                                         g.MeshBasicMaterial(color=rgb_to_hex(*self.block_colors[i//2]))))
            
            for t, position in enumerate(points):
                self.vis[f"traj/sphere_{i}_{t}"].set_object(g.Sphere(0.005), g.MeshPhongMaterial(color=rgb_to_hex(*self.block_colors[i//2])))
                self.vis[f"traj/sphere_{i}_{t}"].set_transform(tf.translation_matrix(position))
        
if __name__ == '__main__':
    from kinematics.util import jax_cache_on
    jax_cache_on()
    
    sim = Simulation(num_blocks=5)
    
    solutions = jnp.load('saved/tetris.npy')
    sim.set_state(solutions)
    
    sim.render()
    
    grasp_to_qf = jax.jit(grasp_to_q)
    # Cycle through block poses
    # while True:
    #     for i, pose in enumerate(sim.block_poses):
    #         print(i)
    #         q, err = grasp_to_qf(yaw_to_quat_xyz(pose))
    #         sim.set_robot_pose(q)
    #         sim.render()
    #         time.sleep(0.5)

