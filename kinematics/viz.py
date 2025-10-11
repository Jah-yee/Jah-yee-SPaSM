import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np

vis = None

def init_vis(port=None): 
    global vis

    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    
    # Clear all
    vis.delete()

def draw_spheres(pos, radii, color=0xffffff, prefix='sphere/', viss=None, opacity=1.0):
    """
    Draw spheres in meshcat, where pos: (n, 3) and radii: (n,)
    """
    global vis
    
    if viss is None:
        viss = vis
        if viss is None:
            raise ValueError("Visualizer not initialized. Call init_vis() first.")

    # Convert to numpy arrays
    if not isinstance(pos, np.ndarray):
        pos = np.array(pos)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    assert pos.ndim == 2 and pos.shape[1] == 3, f"pos must be of shape (n, 3), got {pos.shape}"
    
    if pos.shape[0] == 0:
        return

    if pos.shape[0] != radii.shape[0]:
        raise ValueError(f"Number of positions {pos.shape[0]} and radii {radii.shape[0]} must match.")

    for i in range(pos.shape[0]):
        sphere_name = prefix + str(i)
        
        if isinstance(color, np.ndarray):
            c = color[i].item()
        else:
            c = color
        
        viss[sphere_name].set_object(g.Sphere(radii[i].item()), 
                                    g.MeshLambertMaterial(color=c, opacity=opacity))
        viss[sphere_name].set_transform(tf.translation_matrix(pos[i]))
