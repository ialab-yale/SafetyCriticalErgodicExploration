# SafetyCriticalErgodicExploration

## Description
Ergodic trajectory planning with safety critical constraints

## Directory Descriptions
All utilities and supplemental scripts are contained within [erg_traj_opt_lib](erg_traj_opt_lib/)

Main scripts for generating trajectories are contained in [collision_stats](experiments/collision_stats/) while the main script for running ablation experiment is contained in [alpha_ablation](experiments/alpha_ablation/)

## File Descriptions

Within [collision_stats](experiments/collision_stats/), [run_safety_critical.py](experiments/collision_stats/run_safety_critical.py) uses the CBF safety critical constraints to generate 50 trajectories using random initial and final state pairs. It saves the trajectories into a directory that needs to be specified and the 50 inital and final state pairs are saved into a numpy data file in that directory as well.

Within [collision_stats](experiments/collision_stats/), [run.py](experiments/collision_stats/run.py) uses normal inequality constraints to generate 50 trajectories using the same initial and final state pairs from run_safety_critical.py. It saves the trajectories into a directory that needs to be specified.

Within [alpha_ablation](experiments/alpha_ablation/), [run.py](experiments/alpha_ablation/run.py) generates trajectory graphs for each of the values of gamma for the same exploration space and initial and final conditions. Then, it generates a graph of the ergodic metric versus the values of gamma.