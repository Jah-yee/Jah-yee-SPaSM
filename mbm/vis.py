import time
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp
import matplotlib
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import yaml
from flax.struct import dataclass

from kinematics.kinematics import fk, get_joint_limits
from kinematics.viz import draw_spheres
from spasm.conversions import gray_hex, rgb_to_hex

def flip_xz(pos):
    """Flips the x and z coordinates of a position."""
    # return [pos[2], pos[1], pos[0]]
    return [pos[0], pos[1], pos[2]]

@dataclass
class ProblemData:
    """Holds all the data for a motion planning problem."""
    qmins: jnp.ndarray
    qmaxes: jnp.ndarray
    box_poses: jnp.ndarray
    box_quats: jnp.ndarray
    box_dims: jnp.ndarray
    cylinder_poses: jnp.ndarray
    cylinder_quats: jnp.ndarray
    cylinder_dims: jnp.ndarray
    q_start: jnp.ndarray
    q_goal: jnp.ndarray

class MBMVisualizer:
    """
    Visualizes Motion Bench Maker scenes and requests.
    """
    
    def __init__(self):
        self.vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        self.vis.delete()
        
        self.qmins, self.qmaxes = get_joint_limits()

    def set_scene(self, problem_dir: str, idx):
        """
        Initializes the visualizer by loading scene and request data.

        Args:
            problem_dir: The directory containing the problem's scene/request files.
            scene_id: The ID of the scene to load.
            request_id: The ID of the request to load.
        """
        # If problem_dir not end in panda add it
        if not problem_dir.endswith("_panda"):
            problem_dir += "_panda"
        
        base_path = Path("./problems") / problem_dir
        scene_file = base_path / f"scene{idx:04d}.yaml"
        request_file = base_path / f"request{idx:04d}.yaml"

        with open(scene_file, "r") as f:
            self.scene_data: Dict[str, Any] = yaml.safe_load(f)
        with open(request_file, "r") as f:
            self.request_data: Dict[str, Any] = yaml.safe_load(f)

        self._parse_scene()
        self._parse_request()

    def _parse_scene(self):
        """Parses collision objects from the scene data."""
        boxes, cylinders = [], []
        if "collision_objects" in self.scene_data["world"]:
            for obj in self.scene_data["world"]["collision_objects"]:
                # Assuming one primitive per object as in examples
                primitive = obj["primitives"][0]
                pose = obj["primitive_poses"][0]

                pos = flip_xz(pose["position"])
                rot = pose["orientation"]
                # rot = [0, 0, 0, 1]  # No rotation for simplicity
                dims = primitive["dimensions"]

                if primitive["type"] == "box":
                    boxes.append((pos, rot, dims))
                elif primitive["type"] == "cylinder":
                    # dimensions are [height, radius]
                    cylinders.append((pos, rot, dims))
                    
        # (N, 3) [x, y, z]
        self.box_poses = jnp.array([p[0] for p in boxes]) if boxes else jnp.empty((0, 3))
        # (N, 4) [x, y, z, w]
        self.box_quats = jnp.array([p[1] for p in boxes]) if boxes else jnp.empty((0, 4))
        # (N, 3) [dx, dy, dz]
        self.box_dims = jnp.array([p[2] for p in boxes]) if boxes else jnp.empty((0, 3))
        
        # (M, 3) [x, y, z]
        self.cylinder_poses = jnp.array([c[0] for c in cylinders]) if cylinders else jnp.empty((0, 3))
        # (M, 4) [x, y, z, w]
        self.cylinder_quats = jnp.array([c[1] for c in cylinders]) if cylinders else jnp.empty((0, 4))
        # (M, 2) [height, radius]
        self.cylinder_dims = jnp.array([c[2] for c in cylinders]) if cylinders else jnp.empty((0, 2))

    def _parse_request(self):
        """Parses start and goal joint configurations from the request data."""
        start_state = self.request_data["start_state"]["joint_state"]
        goal_constraints = self.request_data["goal_constraints"][0]["joint_constraints"]
        
        supposed_joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
        # Assert that joint names are consistent
        goal_joint_names = [c["joint_name"] for c in goal_constraints]
        assert goal_joint_names == supposed_joint_names[:7], "Unexpected goal joint names." + str(goal_joint_names)
        assert start_state["name"] == supposed_joint_names, "Unexpected start joint names." + str(start_state["name"])
        
        self.joint_names = start_state["name"]

        self.q_start = jnp.array(start_state["position"])
        self.q_goal = jnp.array([c["position"] for c in goal_constraints])
        
        assert self.q_goal.shape == (7,)
        # Pad q_goal
        self.q_goal = jnp.concatenate([self.q_goal, jnp.array([0.00, 0.00])])

        # Assert that other parameters are consistent (example)
        assert self.request_data["group_name"] == "panda_arm"
        assert self.request_data["planner_id"] == "BKPIECEGood"

    def render_env(self):
        """Renders the static environment (boxes and cylinders)."""
        self.vis["/Grid"].set_property("visible", False)
        self.vis["/Background"].set_property("top_color", [1, 1, 1])
        self.vis["/Background"].set_property("bottom_color", [1, 1, 1])
        self.vis["/Axes"].set_property("visible", False)
        
        green = rgb_to_hex(144, 238, 144)  # Light green
        material = g.MeshPhongMaterial(color=green)
        # Render boxes
        for i, (pos, quat, dims) in enumerate(zip(self.box_poses, self.box_quats, self.box_dims)):
            if i == 6:
                continue
            # Reorder quat from [x, y, z, w] to [w, x, y, z] for transformations library
            quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
            rot_matrix = tf.quaternion_matrix(quat_wxyz)
            trans_matrix = tf.translation_matrix(np.array(pos))
            self.vis[f"env/box_{i}"].set_object(g.Box([float(d) for d in dims]), material)
            self.vis[f"env/box_{i}"].set_transform(trans_matrix @ rot_matrix)

        # Render cylinders
        for i, (pos, quat, dims) in enumerate(zip(self.cylinder_poses, self.cylinder_quats, self.cylinder_dims)):
            height, radius = dims
            # Reorder quat from [x, y, z, w] to [w, x, y, z] for transformations library
            quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
            rot_matrix = tf.quaternion_matrix(quat_wxyz)
            trans_matrix = tf.translation_matrix(np.array(pos))
            # In meshcat, cylinder is along Z, so we rotate it to be along Y
            rot_x = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
            self.vis[f"env/cylinder_{i}"].set_object(g.Cylinder(float(height), float(radius)), material)
            self.vis[f"env/cylinder_{i}"].set_transform(trans_matrix @ rot_matrix @ rot_x)

    def render_robot(self, q: jnp.ndarray):
        """
        Renders the robot at a given joint configuration.

        Args:
            q: The joint configuration of the robot. Shape (9,).
        """
        assert q.shape == (9,), f"Expected q to have shape (9,), got {q.shape}"
        robot_pos, robot_radii = fk(q)
        draw_spheres(robot_pos, robot_radii, color=gray_hex(), prefix='robot/', viss=self.vis)
    
    def render_traj(self, qs, skip=1, hand_only=False, opacity=1.0):
        '''
        Draw all qs transparently
        '''
        # Clear all traj/
        self.vis['traj'].delete()
                
        fk_vmap = jax.jit(jax.vmap(fk, in_axes=(0,)))
        robot_pos, robot_radii = fk_vmap(qs)        
        for i in range(len(qs)):
            is_last = i == len(qs) - 1
            is_skip = i % skip == 0
            if is_last or is_skip:
                ii = len(qs) - i - 1  # Reverse index for color
                # Generate random HSV color with fixed S=0.8, V=0.8, random H
                h = (ii * 0.03) % 1  # Golden ratio conjugate for good distribution
                s, v = 0.4, 1.0
                rgb = matplotlib.colors.hsv_to_rgb([h, s, v]) * 255
                rgb = rgb.astype(int).tolist()
                if hand_only and i > 0:
                    draw_spheres(robot_pos[i, -30:], robot_radii[i, -30:], 
                                 color=rgb_to_hex(*rgb), prefix=f'traj/n{i}', 
                                 viss=self.vis, opacity=opacity)
                else:
                    draw_spheres(robot_pos[i], robot_radii[i], 
                                 color=rgb_to_hex(*rgb), prefix=f'traj/n{i}', 
                                 viss=self.vis, opacity=opacity)
                time.sleep(0.05)
        
    def get_problem_data(self) -> ProblemData:
        """Returns a dataclass with all the problem data."""
        return ProblemData(
            qmins=self.qmins,
            qmaxes=self.qmaxes,
            box_poses=self.box_poses,
            box_quats=self.box_quats,
            box_dims=self.box_dims,
            cylinder_poses=self.cylinder_poses,
            cylinder_quats=self.cylinder_quats,
            cylinder_dims=self.cylinder_dims,
            q_start=self.q_start,
            q_goal=self.q_goal,
        )


s1 = jnp.array([0.0, -1.9, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
s2 = jnp.array([-0.7, -1.9, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
s3 = jnp.array([-1.5, -1.9, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
s4 = jnp.array([-2.35, -1.9, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
s5 = jnp.array([-3.14, -1.9, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
s6 = jnp.array([ 0.7, -1.9, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
s7 = jnp.array([ 1.5, -1.9, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
s8 = jnp.array([ 2.35, -1.9, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
s9 = jnp.array([ 3.14, -1.9, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])

e1 = jnp.array([0.0, -2.9671, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
e2 = jnp.array([0.0,  2.9671, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])

e3 = jnp.array([1.5, -2.9671, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
e4 = jnp.array([1.5,  2.9671, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])

e5 = jnp.array([-1.5, -2.9671, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
e6 = jnp.array([-1.5,  2.9671, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])

e7 = jnp.array([3.14, -2.9671, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
e8 = jnp.array([3.14,  2.9671, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])

e9 = jnp.array([-3.14, -2.9671, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])
e10 = jnp.array([-3.14,  2.9671, 0.0, 0.0, 3.0, 3.0, 1.0, 0.04, 0.04])

all_retractions = jnp.stack([e1, e2, e3, e4, e5, e6, e7, e8, e9, e10])

if __name__ == "__main__":
    # Example: Load scene 1 and request 1 from the box_panda problem
    visualizer = MBMVisualizer()

    # Draw the static environment once

    # Oscillate between start and goal robot poses
    # for idx in range(1, 101):
    #     visualizer.set_scene("table_under_pick", idx=idx)
    #     visualizer.render_env()
    
    #     print("Rendering start pose...")
    #     visualizer.render_robot(visualizer.q_start)
    #     time.sleep(0.2)

    #     print("Rendering goal pose...")
    #     visualizer.render_robot(visualizer.q_goal)
    #     time.sleep(1)
    
    
    # [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, 0.    ,  0.    ]
    # [2.9671, 1.8326, 2.9671, 0.0873, 2.9671, 3.8223, 2.9671, 0.1  0.1   ]
    # Show each pose for 2 seconds
    for s in [s1, s2, s3, s4, s5, s6, s7, s8, s9]:
        print("Rendering pose:", s)
        visualizer.render_robot(s)
        time.sleep(2)