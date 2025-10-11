import jax
import jax.numpy as jnp
import numpy as np
import xmltodict

from kinematics.util import axis_angle_to_matrix, xyzrpy_to_matrix

# Helper functions for transformations with manual scalar operations

def manual_matrix_multiply(A, B):
    """
    Performs a 4x4 matrix multiplication using explicit scalar operations.
    This is a non-optimized, manual implementation for clarity or specific backend needs.

    Args:
        A (jnp.ndarray): A 4x4 matrix.
        B (jnp.ndarray): A 4x4 matrix.

    Returns:
        jnp.ndarray: The resulting 4x4 matrix from A @ B.
    """
    # C = jnp.zeros((4, 4))
    # for i in range(4):
    #     for j in range(4):
    #         sum_val = 0
    #         for k in range(4):
    #             sum_val += A[i, k] * B[k, j]
    #         C = C.at[i, j].set(sum_val)
    # return C
    return jnp.dot(A, B)

def _parse_urdf(urdf_path):
    """
    Parses a URDF file to extract the kinematic structure of the robot.

    Args:
        urdf_path (str): The file path to the URDF file.

    Returns:
        tuple: A tuple containing:
            - links (dict): A dictionary of link data.
            - parent_joint_map (dict): Maps child links to their parent joint.
            - joint_to_q_idx (dict): Maps movable joint names to their index in the `q` vector.
            - link_order (list): A topologically sorted list of link names.
            - base_link (str): The name of the base link.
    """
    with open(urdf_path, 'r') as file:
        xml_content = file.read()
    
    robot_data = xmltodict.parse(xml_content)['robot']
    
    joints = {j['@name']: j for j in robot_data['joint']}
    links = {l['@name']: l for l in robot_data['link']}
    
    # All movable joints (revolute or prismatic)
    movable_joints = []
    for j_name, j_data in joints.items():
        if j_data.get('@type') in ['revolute', 'prismatic']:
            movable_joints.append(j_data)

    # Follow order of appearance in the URDF
    joint_to_q_idx = {j['@name']: i for i, j in enumerate(movable_joints)}

    # Build chain structure
    parent_joint_map = {j['child']['@link']: j for j in joints.values()}

    # Base link is the one without an entry in parent_joint_map but in links
    links_without_parent = [l for l in links if l not in parent_joint_map]
    assert len(links_without_parent) == 1, f"There should be exactly one base link without a parent, got {links_without_parent}"
    base_link = links_without_parent[0]

    # Find all children of a link
    child_map = {}
    for j in joints.values():
        parent = j['parent']['@link']
        child = j['child']['@link']
        if parent not in child_map:
            child_map[parent] = []
        child_map[parent].append(child)

    # Traverse the kinematic tree from the world link to get topological order
    nodes_to_visit = [base_link]
    link_order = []
    while nodes_to_visit:
        link_name = nodes_to_visit.pop(0)
        if link_name in link_order:
            raise ValueError(f"Circular dependency detected in URDF involving {link_name}")
        link_order.append(link_name)
        nodes_to_visit.extend(child_map.get(link_name, []))

    return links, parent_joint_map, joint_to_q_idx, link_order, base_link

def _calculate_link_transforms(q, parent_joint_map, joint_to_q_idx, link_order, base_link):
    """
    Calculates the world transformation for each link in the kinematic chain.

    Args:
        q (jnp.ndarray): The vector of joint values.
        parent_joint_map (dict): A map from child link name to its parent joint data.
        joint_to_q_idx (dict): A map from movable joint name to its index in `q`.
        link_order (list): A topologically sorted list of link names.
        base_link (str): The name of the base link (e.g., 'world').

    Returns:
        dict: A dictionary mapping each link name to its 4x4 world transformation matrix.
    """
    
    assert q.ndim == 1
    
    if q.shape[0] == 7:
        q = jnp.pad(q, (0, 2))
        
    link_transforms = {}
    
    for link_name in link_order:
        # In each iteration, find the global transform for the link
        if link_name == base_link:
            link_transforms[link_name] = jnp.identity(4)
            continue
            
        joint = parent_joint_map[link_name]
        parent_name = joint['parent']['@link']

        # Compute the static transform from the joint origin
        origin_xyz = [float(i) for i in joint['origin']['@xyz'].split()]
        origin_rpy = [float(i) for i in joint['origin']['@rpy'].split()]
        static_transform = xyzrpy_to_matrix(origin_xyz, origin_rpy)

        # Compute the motion transform based on the joint type
        motion_transform = jnp.identity(4)
        joint_type = joint.get('@type')

        if joint_type in ['revolute', 'prismatic']:
            axis_xyz = jnp.array([float(i) for i in joint['axis']['@xyz'].split()])
            q_val = q[joint_to_q_idx[joint['@name']]]

            if joint_type == 'revolute':
                motion_transform = axis_angle_to_matrix(axis_xyz, q_val)
            elif joint_type == 'prismatic':
                translation_vector = axis_xyz * q_val
                motion_transform = motion_transform.at[0:3, 3].set(translation_vector)
        elif joint_type == 'fixed':
            pass  # No motion for fixed joints
        else:
            raise ValueError(f"Unsupported joint type: {joint_type}")

        # Compute the joint transform
        joint_transform = manual_matrix_multiply(static_transform, motion_transform)
        parent_transform = link_transforms[parent_name]
        link_transforms[link_name] = manual_matrix_multiply(parent_transform, joint_transform)
        
    return link_transforms

def _calculate_sphere_positions(links, link_transforms):
    """
    Calculates the world coordinates of all collision spheres defined in the URDF.

    Args:
        links (dict): A dictionary of link data from the parsed URDF.
        link_transforms (dict): A dictionary of world transformations for each link.

    Returns:
        tuple: A tuple containing:
            - jnp.ndarray: An (N, 3) array of sphere positions.
            - jnp.ndarray: An (N,) array of sphere radii.
    """
    all_spheres_pos = []
    all_spheres_radii = []

    for link_name, link_data in links.items():
        # If a link has no collision data, skip it.
        if 'collision' not in link_data or not link_data['collision']:
            continue

        # Assert that the link has a transform
        link_transform = link_transforms.get(link_name)
        if link_transform is None:
            raise ValueError(f"Transform for link '{link_name}' not found.")

        colliders = link_data['collision'] if isinstance(link_data['collision'], list) else [link_data['collision']]
        
        for collider in colliders:
            if (not collider) or ('geometry' not in collider) or (not collider['geometry']):
                raise ValueError(f"Collider in link '{link_name}' is missing geometry data.")

            geometry = collider['geometry']
            
            if 'sphere' not in geometry or not geometry['sphere']:
                raise ValueError(f"Collider in link '{link_name}' does not define a sphere geometry.")

            sphere_data = geometry['sphere']

            if '@radius' not in sphere_data:
                raise ValueError(f"A <sphere> collider in link '{link_name}' is missing the required 'radius' attribute.")

            sphere_radius = float(sphere_data['@radius'])
            
            if 'origin' not in collider:
                raise ValueError(f"Collider in link '{link_name}' is missing origin data.")
            if (not '@xyz' in collider['origin']) or (not '@rpy' in collider['origin']):
                raise ValueError(f"Collider in link '{link_name}' has incomplete origin data.")
            
            origin_xyz = [float(i) for i in collider['origin']['@xyz'].split()]
            origin_rpy = [float(i) for i in collider['origin']['@rpy'].split()]

            sphere_static_transform = xyzrpy_to_matrix(origin_xyz, origin_rpy)
            sphere_transform = manual_matrix_multiply(link_transform, sphere_static_transform)
            
            sphere_pos = sphere_transform[0:3, 3]
            
            all_spheres_pos.append(sphere_pos)
            all_spheres_radii.append(sphere_radius)
    
    return jnp.array(all_spheres_pos), jnp.array(all_spheres_radii)

def _fk(q, urdf_path='kinematics/urdf/panda_sphere_visuals.urdf'):
    """
    Performs forward kinematics for a robot defined by a URDF file to get collision sphere locations.

    This function parses the URDF, calculates the transformation for each link, and then
    computes the world positions of all defined collision spheres.

    Args:
        urdf_path (str): Path to the URDF file.
        q (jnp.ndarray): The vector of joint values for the robot's movable joints.

    Returns:
        positions (jnp.ndarray): An (N, 3) array of world coordinates for each sphere.
        radii (jnp.ndarray): An (N,) array of radii for each sphere.
    """
    links, parent_joint_map, joint_to_q_idx, link_order, base_link = _parse_urdf(urdf_path)
    
    # pp = pprint.PrettyPrinter(indent=2, width=120, compact=False)
    # print("DEBUG: links =")
    # for key in links.keys():
    #     print(f"  {key}")

    # print("DEBUG: parent_joint_map =")
    # for key, val in parent_joint_map.items():
    #     print(f"  {key}: {val.get('@name')}")
        
    # print("DEBUG: joint_to_q_idx =")
    # pp.pprint(joint_to_q_idx)

    # print("DEBUG: link_order =")
    # for i, link_name in enumerate(link_order):
    #     print(f"  {i}: {link_name}")

    link_transforms = _calculate_link_transforms(q, parent_joint_map, joint_to_q_idx, link_order, base_link)   
    
    positions, radii = _calculate_sphere_positions(links, link_transforms)
    
    return positions, radii

fk_func = None
def fk(q):
    global fk_func
    if fk_func is None:
        fk_func = jax.jit(_fk)
    return fk_func(q)

fk_batched_ = None
def fk_batched(q):
    global fk_batched_
    if fk_batched_ is None:
        fk_batched_ = jax.jit(jax.vmap(fk))
    return fk_batched_(q)

def _get_ee_pose(q, target_link_name='panda_grasptarget', urdf_path='kinematics/urdf/panda_sphere_visuals.urdf', ):
    """
    Computes the pose (translation and rotation) of a specific link.

    Args:
        urdf_path (str): Path to the URDF file.
        q (jnp.ndarray): The vector of joint values.
        target_link_name (str): The name of the link to get the pose of.

    Returns:
        jnp.ndarray: A 4x4 transformation matrix representing the pose of the target link.
    
    Raises:
        ValueError: If the target_link_name is not found in the URDF.
    """
    _, parent_joint_map, joint_to_q_idx, link_order, base_link = _parse_urdf(urdf_path)
    link_transforms = _calculate_link_transforms(q, parent_joint_map, joint_to_q_idx, link_order, base_link)
    
    if target_link_name in link_transforms:
        pose_matrix = link_transforms[target_link_name]
        # translation = pose_matrix[0:3, 3]
        # rotation_matrix = pose_matrix[0:3, 0:3]
        return pose_matrix
    
    raise ValueError(f"Link '{target_link_name}' not found in URDF.")

_get_ee_pose_jit = None
def get_ee_pose(q):
    """
    Computes the pose (translation and rotation) of a specific link.

    Args:
        urdf_path (str): Path to the URDF file.
        q (jnp.ndarray): The vector of joint values.
        target_link_name (str): The name of the link to get the pose of.

    Returns:
        jnp.ndarray: A 4x4 transformation matrix representing the pose of the target link.
    
    Raises:
        ValueError: If the target_link_name is not found in the URDF.
    """
    global _get_ee_pose_jit
    if _get_ee_pose_jit is None:
        _get_ee_pose_jit = jax.jit(_get_ee_pose)
    return _get_ee_pose_jit(q)

def get_hand_pose(q, target_link_name='panda_link8', urdf_path='kinematics/urdf/panda_sphere_visuals.urdf', ):
    return get_ee_pose(q, target_link_name, urdf_path)

# ee_pose_batched_ = None
# def ee_pose_batched(q):
#     global ee_pose_batched_
#     if ee_pose_batched_ is None:
#         ee_pose_batched_ = jax.jit(jax.vmap(get_ee_pose))
#     return ee_pose_batched_(q)

# ee_pose_ = None
# def ee_pose(q):
#     global ee_pose_
#     if ee_pose_ is None:
#         ee_pose_ = jax.jit(get_ee_pose)
#     return ee_pose_(q)

fk_jac_batched_ = None
def fk_jac_batched(q):
    '''jacobian to end-effector, use jaxrev'''
    global fk_jac_batched_
    if fk_jac_batched_ is None:
        fk_jac_batched_ = jax.jit(jax.vmap(jax.jacrev(get_ee_pose)))
            
    return fk_jac_batched_(q)

def transform_to_rotation_matrix(transform):
    """
    Extracts the rotation matrix from a 4x4 transformation matrix.

    Args:
        transform (jnp.ndarray): A ...x4x4 transformation matrix.

    Returns:
        jnp.ndarray: A 3x3 rotation matrix.
    """
    return transform[..., 0:3, 0:3]

def transform_to_position(transform):
    """
    Extracts the translation vector from a 4x4 transformation matrix.

    Args:
        transform (jnp.ndarray): A ...x4x4 transformation matrix.

    Returns:
        jnp.ndarray: A 3D translation vector.
    """
    return transform[..., 0:3, 3]

joint_min = None
joint_max = None

def get_joint_limits():
    """
    Parses a URDF file to get the joint limits for all movable joints.

    Returns:
        mins (jnp.ndarray): The lower joint limits.
        maxes (jnp.ndarray): The upper joint limits.
    """
    
    global joint_min, joint_max
    
    if joint_min is not None and joint_max is not None:
        return joint_min, joint_max
    
    urdf_path = 'kinematics/urdf/panda_sphere_visuals.urdf'
    _, parent_joint_map, joint_to_q_idx, link_order, base_link = _parse_urdf(urdf_path)
    
    with open(urdf_path, 'r') as file:
        xml_content = file.read()
    
    robot_data = xmltodict.parse(xml_content)['robot']
    joints = {j['@name']: j for j in robot_data['joint']}

    mins = jnp.zeros((len(joint_to_q_idx),))
    maxes = jnp.zeros((len(joint_to_q_idx),))
        
    for index in range(len(joint_to_q_idx)):
        # Look for joint that maps to index
        for joint_name, q_index in joint_to_q_idx.items():
            if q_index == index:
                break
        else:
            raise ValueError(f"Joint index {index} not found.")

        joint = joints[joint_name]
        
        if 'limit' in joint and '@lower' in joint['limit'] and '@upper' in joint['limit']:
            lower = float(joint['limit']['@lower'])
            upper = float(joint['limit']['@upper'])
            mins = mins.at[index].set(lower)
            maxes = maxes.at[index].set(upper)
        else:
            raise ValueError(f"Joint '{joint_name}' does not have limits specified.")

    joint_min = mins
    joint_max = maxes
    
    return mins, maxes
