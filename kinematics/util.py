import jax
import jax.numpy as jnp
import numpy as np

def jax_cache_on():
    import jax
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    # jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

def rgb_to_hex(r, g, b):
    """Converts RGB color values [0 - 255] to a hexadecimal value 0x."""
    return (r << 16) | (g << 8) | b

def white_hex():
    """Returns the hexadecimal color code for white."""
    return rgb_to_hex(255, 255, 255)

def purple_hex():
    """Returns the hexadecimal color code for purple."""
    return rgb_to_hex(128, 0, 128)

def gray_hex():
    """Returns the hexadecimal color code for gray."""
    return rgb_to_hex(200, 200, 200)

def dark_blue_grey():
    return 0x2c3e50
    
def rpy_to_matrix(roll, pitch, yaw):
    """Converts RPY angles to a 3x3 rotation matrix."""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def xyzrpy_to_matrix(xyz, rpy):
    """Converts XYZ translation and RPY rotation to a 4x4 homogeneous transformation matrix."""
    rotation_matrix = rpy_to_matrix(rpy[0], rpy[1], rpy[2])
    translation_vector = np.array(xyz).reshape(3, 1)
    matrix = np.eye(4)
    matrix[0:3, 0:3] = rotation_matrix
    matrix[0:3, 3] = translation_vector.flatten()
    return matrix

def axis_angle_to_matrix(axis, angle):
    """Creates a 4x4 rotation matrix from an axis and angle."""
    norm = jnp.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    axis = jnp.where(norm > 0, axis / norm, axis)
    
    x, y, z = axis[0], axis[1], axis[2]
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    t = 1.0 - c

    # Rodrigues' rotation formula
    rotation_matrix_3x3 = jnp.array([
        [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ])
    
    matrix = jnp.identity(4)
    matrix = matrix.at[0:3, 0:3].set(rotation_matrix_3x3)
    return matrix

def distance(a, b):
    """
    Computes the distance between two 4x4 transformation matrices.
    The distance is a weighted sum of translational and rotational distances.
    """
    
    # Rotational distance
    a = matrix_to_zyx(a)
    b = matrix_to_zyx(b)
    rot_dist = jnp.linalg.norm(a - b)

    return rot_dist

def matrix_to_angle_axis(R):
    """
    Converts a rotation matrix to an angle-axis representation.
    The angle is in radians.
    """
    trace = jnp.trace(R)
    angle = jnp.arccos((trace - 1) / 2)
    return angle

def matrix_to_zyx(R):
    """Converts a rotation matrix to ZYX Euler angles."""
    sy = jnp.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    x = jnp.arctan2(R[2, 1], R[2, 2])
    y = jnp.arctan2(-R[2, 0], sy)
    z = jnp.arctan2(R[1, 0], R[0, 0])
    
    # Handle singularity
    x_singular = jnp.arctan2(-R[1, 2], R[1, 1])
    y_singular = jnp.arctan2(-R[2, 0], sy)
    z_singular = 0.0
    
    return jnp.where(singular, jnp.array([x_singular, y_singular, z_singular]), jnp.array([x, y, z]))

def zyx_to_matrix(zyx):
    """Converts ZYX Euler angles to a rotation matrix."""
    z, y, x = zyx[0], zyx[1], zyx[2]
    cz, sz = jnp.cos(z), jnp.sin(z)
    cy, sy = jnp.cos(y), jnp.sin(y)
    cx, sx = jnp.cos(x), jnp.sin(x)
    
    return jnp.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx]
    ])