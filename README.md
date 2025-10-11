# SPaSM: Millisecond Sequential Manipulation

### [Project Page](http://commalab.org/papers/spasm) | [Paper](http://arxiv.org/abs/2510.07674) | [Video](https://www.youtube.com/watch?v=VK8PYsdXNBk)

Jax code for solving sequential constraint satisfaction and manipulation problems in milliseconds<br><br>
[SPaSM: Differentiable Particle Optimization for Fast Sequential Manipulation](http://commalab.org/papers/spasm)  
 [Lucas Chen](https://commalab.org/members/lucas)<sup>1</sup>,
 [Shrutheesh R. Iyer](https://shrutheeshir.github.io)<sup>1</sup>,
 [Zachary Kingston](https://commalab.org/members/kingston)<sup>1</sup> <br>
<sup>1</sup>Purdue University <br>
submitted to ICRA 2026

[!Video](https://github.com/user-attachments/assets/2081bfb8-7161-4d8b-9afc-8b4aefe5cf05)

SPaSM solution to the obstructed block stacking task. The block placements and arm trajectories are jointly optimized for feasiblity and shortness.


## Install
Set up a conda environment, taking care to ensure you have the cuda version of JAX installed.
```bash
conda create -n spasm 'numpy<2' jax "jaxlib=*=*cuda*" flax -c conda-forge 
conda activate spasm
pip install meshcat jaxlie xmltodict seaborn
```

## Run
First start meshcat server in a separate terminal:
```bash
meshcat-server --zmq-url tcp://127.0.0.1:6000
```

You should be able to go to `localhost:7000/static` in your browser and see a blank environment.

You can proceed to run any of the problems discussed in our [paper](http://arxiv.org/abs/2510.07674).
Here we will demonstrate solving the tetris problem with 8 tetrominoes, visualizing intermediate steps.
```bash
python spasm/solve.py --num_blocks 8 --viewopt
```

You should see a solved configuration like this:

<img width="500" alt="Screenshot 2025-10-10 053537" src="./media/tetris.png" />
<br>

Try running the solver again with timing (Be sure to disable visualization (`--viewopt`) for accurate timings)
```bash
python spasm/solve.py --num_blocks 8 --bench
```
A reported timing of ~30ms is common for laptop GPUs, and even faster runs are expected for more powerful CUDA-supported graphics cards. Please refer to our paper for timings on our RTX 4090 machine. 

## Problem Listing

<b> [spasm/solve.py](spasm/solve.py): </b> Run tetromino placement CSP for the tetris problem. 

<b> [spasm/tetris_traj.py](spasm/tetris_traj.py): </b> Run trajopt for the tetris problem. `spasm/solve.py` must be run first so its solution file can be read.

<b> [spasm/tower_solve.py](spasm/tower_solve.py): </b> Run block placements CSP for the tower problem. 

<b> [spasm/tower_traj.py](spasm/tower_traj.py): </b> Run trajopt for the tower problem. `spasm/tower_solve.py` must be run first so its solution file can be read.

<b> [mbm/solve.py](spasm/tower_traj.py): </b> Run solver for the [MotionBenchMaker](https://github.com/KavrakiLab/motion_bench_maker) problems (Franka Research 3 only). You must first download the problem files into `problems/`
