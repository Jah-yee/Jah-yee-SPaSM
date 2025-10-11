import jax
import jax.numpy as jnp
import time
import os
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from mbm.solve import optimize_traj, TrajOptParams
from mbm.vis import MBMVisualizer
from kinematics.util import jax_cache_on

def run_benchmarks(problems, max_idx, trials):
    """
    Runs the trajectory optimization benchmark.
    """
    jax_cache_on()
    
    viz = MBMVisualizer()
    viz.set_scene('cage', idx=1)
    
    params = TrajOptParams()
    
    
    for problem in problems:
        problem_path = os.path.join("solutions", problem)
        
        # Warmup
        key = jax.random.key(time.time_ns())
        optimize_traj(key, params, viz.get_problem_data())
    
        os.makedirs(problem_path, exist_ok=True)
        
        print('Running problem:', problem)

        all_fail_idx = []
        
        for idx in range(1, max_idx + 1):            
            viz.set_scene(problem, idx=idx)
            
            all_pene = []
            
            for trial in range(trials):
                solution_file = os.path.join(problem_path, f"solution_{idx}_{trial}.npy")
                timing_file = os.path.join(problem_path, f"timing_{idx}_{trial}.npz")
                os.makedirs(os.path.dirname(solution_file), exist_ok=True)
                
                # if os.path.exists(solution_file):
                #     continue

                key = jax.random.key(trial + 10 * idx)

                start_time = time.perf_counter()
                q_traj_optimized, best_pene, steps = optimize_traj(key, params, viz.get_problem_data())
                q_traj_optimized.block_until_ready()
                end_time = time.perf_counter()
                
                assert best_pene < 1e-6
                
                elapsed_ms = (end_time - start_time) * 1000
                
                np.save(solution_file, np.asarray(q_traj_optimized))
                np.savez(timing_file, elapsed=np.array(elapsed_ms), best_penetration=best_pene, steps=steps)
                all_pene.append(best_pene)
        
            # print('    Success:', np.sum(np.array(all_pene) < 1e-5), '/', len(all_pene))
            if np.any(np.array(all_pene) >= 1e-6):
                all_fail_idx.append(idx)
        print(f"Problem {problem} - Failed indices: {all_fail_idx}")
        
def analyze_results(problems, max_idx, trials):
    """
    Analyzes the benchmark results.
    """
    total_trials = max_idx * trials

    for problem in problems:
        problem_path = os.path.join("solutions", problem)
        
        successful_timings = []
        successful_traj_lengths = []
        
        timing_files = glob.glob(os.path.join(problem_path, "timing_*.npz"))
        
        for timing_file in timing_files:
            data = np.load(timing_file)
            best_pene = data['best_penetration']
            
            if best_pene < 1e-3:
                successful_timings.append(data['elapsed'])
                
                sol_file = timing_file.replace("timing_", "solution_").replace(".npz", ".npy")
                if os.path.exists(sol_file):
                    traj = np.load(sol_file)
                    traj_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
                    successful_traj_lengths.append(traj_len)

        print(f"--- Problem: {problem} ---")
        if not successful_timings:
            print("  No successful trials found.")
            continue

        # Trajectory Length Analysis
        if successful_traj_lengths:
            mean_len = np.mean(successful_traj_lengths)
            std_len = np.std(successful_traj_lengths)
            ci_len = 1.96 * std_len / np.sqrt(len(successful_traj_lengths))
            print(f"  Trajectory Length: {mean_len:.3f} +/- {ci_len:.3f} (95% CI) from {len(successful_traj_lengths)} successful trials")
        
        # Success vs. Time Plot
        sorted_timings = np.sort(successful_timings)
        success_rate = np.arange(1, len(sorted_timings) + 1) / total_trials * 100
        
        plt.figure()
        sns.set_theme(style="whitegrid")
        ax = sns.lineplot(x=sorted_timings, y=success_rate)
        ax.set(title=f"Success vs. Time for {problem}", xlabel="Time (ms)", ylabel="Success Rate (%)")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(f"{problem}_success_vs_time.png")
        plt.close()
        print(f"  Generated success vs. time plot: {problem}_success_vs_time.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MBM Benchmark")
    args = parser.parse_args()

    # problems = [d for d in os.listdir("problems") if os.path.isdir(os.path.join("problems", d))]
    problems = ['bookshelf_tall', 'table_pick', 'bookshelf_thin', 'cage', 'box', 'bookshelf_small', 'table_under_pick']
    # problems = ['cage']
    print("Running benchmarks on problems:", problems)
    
    trials = 1
    run_benchmarks(problems, 100, trials)

    analyze_results(problems, 100, trials)
