import time
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from typing import List, Literal
import random

import pinocchio as pin
import pinocchio.visualize as pviz
from pinocchio.robot_wrapper import RobotWrapper

import jax
import jax.numpy as jnp
import jaxlie

from kinematics.kinematics import fk, get_ee_pose, get_joint_limits
from kinematics.viz import draw_spheres
from spasm.conversions import grasp_to_q, gray_hex, rgb_to_hex, yaw_to_quat_xyz, interp

def gray_hexr():
    return [210, 210, 210]

class TowerSimulation:
    def __init__(self, num_blocks: Literal[1, 3, 5, 8], num_obs: int = 10, use_pinocchio: bool = False):
        """
        Initializes the simulation environment with blocks for stacking.

        Args:
            num_blocks: The number of blocks to include in the environment.
            num_obs: The number of floating red spheres to include as obstacles.
            key: JAX random key for initial placement.
            use_pinocchio: If True, use pinocchio to render the robot.
        """
        self.num_blocks = num_blocks
        self.num_obs = num_obs
        assert num_obs in [0, 1, 10]
        
        # Custom costs
        self.z_error_mul = 1.0
        
        self.block_dims = jnp.array([0.06, 0.06, 0.06])
        self.block_height = self.block_dims[2]

        self.vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        self.vis.delete()

        self.use_pinocchio = use_pinocchio
        if self.use_pinocchio:
            urdf_path = "kinematics/urdf/panda.urdf"
            self.robot = RobotWrapper.BuildFromURDF(urdf_path, "kinematics/urdf/meshes")
            self.visual_model = self.robot.visual_model
            self.viz_pin = pviz.MeshcatVisualizer(self.robot.model, self.robot.collision_model, self.visual_model)
            self.viz_pin.initViewer(self.vis)
            self.viz_pin.loadViewerModel()

        self.table_dims = [1.1, 1.5, 0.02]
        self.table_pose = [0.15, 0.0, -0.011]
        self.table_color = [255, 255, 255] # gray_hexr() #[200, 200, 255]
        
        # Deepness, width, height
        self.goal_dims = jnp.array([0.6, 1.0, 1.0])
        self.goal_position = jnp.array([0.3, 0.0, 0.5])
        self.goal_color = [186, 255, 201, 0.2]
        
        # Randomly place blocks on the table
        
        key = jax.random.PRNGKey(21)
        # random_xy = jax.random.uniform(key, (num_blocks, 2), minval=self.goal_position[:2] - self.goal_dims[:2] / 2,
        #                                                    maxval=self.goal_position[:2] + self.goal_dims[:2] / 2)
        # random_yaw = jax.random.uniform(key, (num_blocks,), minval=-jnp.pi, maxval=jnp.pi)

        # block_poses = [
        #     jnp.array([random_xy[i, 0], random_xy[i, 1], self.block_height / 2.0, random_yaw[i]])
        #     for i in range(num_blocks)
        # ]
        
        # Spawn all blocks in the left side of the table
        block_poses = [[0.4 - (i - 5) * 0.12, 0.30, self.block_height / 2.0, 0.0] for i in range(num_blocks//2, num_blocks)] + \
                      [[0.4 - i * 0.12, 0.5, self.block_height / 2.0, 0.0] for i in range(num_blocks//2)] 
        block_poses = jnp.array(block_poses)

        self.block_poses = block_poses
        self.block_poses_original = self.block_poses.copy()
        self.block_colors = [0xfd3f52, 0xff6b6b, 0xfd7e03, 0xffbc16, 0xa9e507, 0x65d73d, 0x38c188, 0x0cd4ae, 0x02ccd0, 0x31b5e7]
        # self.block_colors = [0xcdf2c0, 0xcdf2c0, 0xcdf2c0, 0xcdf2c0, 0xcdf2c0, 0xcdf2c0, 0xcdf2c0, 0x65d73d, 0xcdf2c0, 0xcdf2c0]
        random.seed(42)
        random.shuffle(self.block_colors)
        self.block_colors = self.block_colors[:num_blocks]
        
        self.block_poses_matrix = {}

        self.q = TowerSimulation.get_neutral_pose()
        self.qmins, self.qmaxes = get_joint_limits()

        # Add obstacles
        
        self.obstacle_color = [255, 255, 255] # [220, 220, 255]
        obs_key, _ = jax.random.split(key)
        min_bounds = self.goal_position - self.goal_dims / 2
        max_bounds = self.goal_position + self.goal_dims / 2
        # self.obstacle_poses = jax.random.uniform(obs_key, (self.num_obs, 3), minval=min_bounds, maxval=max_bounds)
        self.obstacle_poses = jnp.array([
            [0.20, 0.5, 0.6],
            [0.15, 0.05, 0.48], # over the hill
            [0.5, 0.55, 0.4], # high
            [0.0, -0.4, 0.3], # low
            [0.0, 0.2, 0.2],
            [0.0, -0.1, 0.9],
            [0.1, -0.2, 0.2],
            [0.2, -0.5, 0.1],
            [0.3, -0.5 - 100, 0.5],
            [0.25, -0.4, 0.0]]) # [0.25,-0.2, 0.2]
        
        self.obstacle_radii = jnp.array([
            0.1,
            0.1,
            0.1,
            0.2,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.3,
            ])
            
        if self.num_obs == 0:
            self.obstacle_poses = jnp.zeros((0, 3))
            self.obstacle_radii = jnp.zeros((0,))
        elif self.num_obs == 1:
            self.obstacle_poses = self.obstacle_poses[-1:]
            self.obstacle_radii = self.obstacle_radii[-1:]

    def set_robot_pose(self, q: jnp.ndarray):
        """Sets the robot's joint configuration.
        
        Args:
            q: Joint configuration for the robot. Shape (9,).
        """
        if q.shape == (7,):
            q = jnp.pad(q, (0, 2))
        assert q.shape == (9,), f"q should be of shape (9,), got {q.shape}"
        self.q = q
    
    @staticmethod
    def get_neutral_pose():
        # return jnp.array([0., -jnp.pi/4, 0., -3*jnp.pi/4, 0., jnp.pi/2, jnp.pi/4, 0., 0.])
        return jnp.array([0., -jnp.pi/4, 0., -2*jnp.pi/4, 0., jnp.pi/2, jnp.pi/4, 0., 0.])

    def set_state(self, block_poses: jnp.ndarray):
        """Sets the state of the blocks.
        
        Args:
            block_poses: An array of poses for each block. Shape (num_blocks, 4).
        """
        assert block_poses.shape == (self.num_blocks, 4), f"block_poses should be of shape ({self.num_blocks}, 4), "\
                                                          f"got {block_poses.shape}"
        self.block_poses = block_poses.copy()
        
    def set_one_state(self, block_idx: int, block_pose: jnp.ndarray):
        """Sets the state of a single block.
        
        Args:
            block_idx: Index of the block to set.
            block_pose: Pose for the block. Shape (4,).
        """
        assert 0 <= block_idx < self.num_blocks, f"block_idx should be in [0, {self.num_blocks}), got {block_idx}"
        assert block_pose.shape == (4,), f"block_pose should be of shape (4,), got {block_pose.shape}"
        self.block_poses[block_idx] = block_pose.copy()
        
    def reset_state(self):
        """Resets the state of the robot and blocks to the original configuration."""
        self.block_poses = self.block_poses_original.copy()
        self.q = TowerSimulation.get_neutral_pose()
        
    def get_initial_state(self):
        """Returns the initial state of the blocks. (num_blocks, 4)"""
        return self.block_poses_original.copy()
        
    def step(self):
        """An empty step function."""
        pass
    
    def draw_trajs(self, q_trajs: jnp.ndarray):
        '''
        Draws the trajectories of the blocks as lines and spheres at the waypoints.
        trajs: (num_blocks, T, 7)
        '''
        for i in range(self.num_blocks * 2 - 1):
            # Original points for spheres
            points = []
            for t in range(q_trajs.shape[1]):
                pose = q_trajs[i, t]
                pose = jnp.pad(pose, (0, 2))
                ee_pose = get_ee_pose(pose)[0:3, 3]
                points.append(ee_pose)

            # Interpolated points for line
            q_traj_interp = interp(q_trajs[i], dist_per_step=0.05)
            interp_points = []
            for t in range(q_traj_interp.shape[0]):
                pose = q_traj_interp[t]
                pose = jnp.pad(pose, (0, 2))
                ee_pose = get_ee_pose(pose)[0:3, 3]
                interp_points.append(ee_pose)
            
            pointsa = np.array(interp_points).T # Shape (3, T_interp)
            self.vis[f"traj/line_{i}"].set_object(g.Line(g.PointsGeometry(pointsa), 
                                                         g.MeshBasicMaterial(color=self.block_colors[i//2])))
            
            for t, position in enumerate(points):
                self.vis[f"traj/sphere_{i}_{t}"].set_object(g.Sphere(0.005), g.MeshPhongMaterial(color=self.block_colors[i//2]))
                self.vis[f"traj/sphere_{i}_{t}"].set_transform(tf.translation_matrix(position))
        
        
    def render(self):
        """Renders the environment in meshcat."""
        self.vis["/Grid"].set_property("visible", False)
        self.vis["/Background"].set_property("top_color", [1, 1, 1])
        self.vis["/Background"].set_property("bottom_color", [1, 1, 1])
        self.vis["/Axes"].set_property("visible", False)
        
        # Add point light at x = 1, y = 1 z = 1
        self.vis.set_property("/Lights/PointLight", {
            "type": "PointLight",
            "color": 0xffffff,  # White light
            "intensity": 20.0,
            "distance": 100,
        })
        self.vis["/Lights/PointLight"].set_transform(tf.translation_matrix([1, 1, 2]))

        # Render table
        self.vis["table"].set_object(g.Box(self.table_dims), g.MeshPhongMaterial(color=rgb_to_hex(*self.table_color)))
        self.vis["table"].set_transform(tf.translation_matrix(self.table_pose))

        # Render goal region
        # goal_dims = [float(v) for v in self.goal_dims]
        # goal_position = [float(v) for v in self.goal_position]
        # material = g.MeshPhongMaterial(color=rgb_to_hex(*self.goal_color[:3]), opacity=self.goal_color[3])
        # self.vis["goal"].set_object(g.Box(goal_dims), material)
        # self.vis["goal"].set_transform(tf.translation_matrix(goal_position))
        
        assert np.isfinite(np.array(self.block_poses)).all(), f"Invalid block_poses: {self.block_poses}"

        for i, (pose, color) in enumerate(zip(self.block_poses, self.block_colors)):
            pose_7d = yaw_to_quat_xyz(pose)
            position = pose_7d[:3]
            orientation = jaxlie.SO3.from_quaternion_xyzw(pose_7d[3:]).as_matrix()
            
            transform_matrix = tf.identity_matrix()
            transform_matrix[:3, :3] = orientation
            transform_matrix[:3, 3] = position

            if i in self.block_poses_matrix:
                transform_matrix = tf.identity_matrix()
                assert self.block_poses_matrix[i].shape == (4, 4), f"block_poses_matrix[{i}] should be of shape (4, 4), got {self.block_poses_matrix[i].shape}"
                transform_matrix[:, :] = self.block_poses_matrix[i]
                del self.block_poses_matrix[i]

            self.vis[f"block_{i}"].set_object(g.Box([float(d) for d in self.block_dims]), g.MeshLambertMaterial(color=color))
            self.vis[f"block_{i}"].set_transform(transform_matrix)

        # Render obstacles
        for i, (pos, r) in enumerate(zip(self.obstacle_poses, self.obstacle_radii)):
            opacity = 0.5 if i == 9 else 1.0
            self.vis[f"obstacle_{i}"].set_object(g.Sphere(r.item()), g.MeshLambertMaterial(color=rgb_to_hex(*self.obstacle_color), opacity=opacity))
            self.vis[f"obstacle_{i}"].set_transform(tf.translation_matrix(pos))

        # Render robot
        if self.use_pinocchio:
            self.viz_pin.display(np.asarray(self.q[:7]))
        else:
            robot_pos, robot_radii = fk(self.q)
            # rgb_to_hex(255, 102, 102)
            draw_spheres(robot_pos, robot_radii, color=rgb_to_hex(255, 255, 255), prefix='spheres/', viss=self.vis)

if __name__ == '__main__':
    from kinematics.util import jax_cache_on
    jax_cache_on()
    
    sim_key = jax.random.PRNGKey(42)
    sim = TowerSimulation(num_blocks=10, num_obs=10, use_pinocchio=True)
    
    # data = jnp.load('saved/tower.npz')
    # traj = jnp.load('saved/tower_traj.npy')
    # solutions = data['opt_particles'] # (num_solutions, num_blocks, 4)
    
    # solutions[:, :, 0] += 0.25
    # solutions[:, :, 1] += 0.07
    # sim.set_state(solutions[0])
    # sim.render()
    # print(solutions)
    # jnp.savez('saved/tower2.npz', opt_particles=solutions, opt_errors=0, init_state=sim.get_initial_state())
    traj = jnp.load('saved/tower_traj.npy')
    initial_state = sim.get_initial_state()
    final_state = jnp.load('saved/tower2.npz')['opt_particles'][0]
    
    at_idx = 5
    # Change the initial state to final state for all blocks after at_idx
    for i in range(0, at_idx):
        initial_state = initial_state.at[i].set(final_state[i])
    sim.set_state(initial_state)
    sim.render()
    
    # sim.render()
    # time.sleep(10)
    
    # final_q_state, err = grasp_to_q(yaw_to_quat_xyz(final_block_state))
    
    # Shape (T, 9)
    # final_traj = np.concatenate([traj[5, :], final_q_state[None, :7]], axis=0)
    final_traj = traj[5, :]
    # Exclude ids 5, 6, 7
    # final_traj = np.delete(final_traj, [5, 6, 7], axis=0)
    # Interpolate final_traj
    num_steps = 5
    interpolated_traj = []
    for i in range(len(final_traj) - 1):
        start_q = final_traj[i]
        end_q = final_traj[i+1]
        for t in range(num_steps):
            interpolated_q = start_q + (end_q - start_q) * t / num_steps
            interpolated_traj.append(interpolated_q)
    interpolated_traj.append(final_traj[-1])
    final_traj = np.array(interpolated_traj)
    
    # final_joint_0 = final_q_state[0]
    # # Multiply all q[0] in final_traj by their offset from final_joint_0
    # for i in range(len(final_traj)):
    #     final_traj[i, 0] += (final_traj[i, 0] - final_joint_0) * 0.5
    
    while True:

        for i, q in enumerate(final_traj):
            # print('i', i)
            ee = get_ee_pose(q)
            sim.block_poses_matrix[at_idx] = np.asarray(ee)
            sim.set_robot_pose(q)
            sim.render()        
            # time.sleep(0.8)
        
        # ee = get_ee_pose(final_q_state)
        # sim.block_poses_matrix[9] = np.asarray(ee)
        # sim.set_robot_pose(final_q_state)
        # sim.render()
        # time.sleep(0.5)

