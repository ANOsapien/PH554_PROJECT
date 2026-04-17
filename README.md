# PH554 Project

Simulation and analysis code for non-reciprocal XY (NRXY) model experiments, including:
- `selfish` dynamics
- `auxiliary` dynamics
- a mixed/two-cone variant in `different_vision_cones`

## Project Structure

- `src/selfish.py`: Selfish NRXY simulation backend.
- `src/auxiliary.py`: Auxiliary NRXY simulation backend.
- `src/runner.py`: Parameter sweep driver for `selfish` or `auxiliary`.
- `src/analyze.py`: Post-processing and Tc estimation from JSON outputs.
- `src/selfish_results/`: Default outputs for selfish runs.
- `src/auxiliary_results/`: Default outputs for auxiliary runs.
- `src/different_vision_cones/runner.py`: Driver for two-cone experiments.
- `src/different_vision_cones/calculate_Tc.py`: Tc estimate utility for two-cone data.
- `src/different_vision_cones/data.json`: Example data for Tc calculation.
- `website/nrxy_spin_visualizer.html`: Visualization page.
- `recreate_results.py`: One-command script to run all runner files.

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install numpy matplotlib numba scipy
```

## Recreate Presented Results (One Command)

From the project root:

```bash
python recreate_results.py
```

This script runs all runner files and both modes (`selfish`, `auxiliary`) in sequence:
1. `src/runner.py --mode selfish`
2. `src/runner.py --mode auxiliary`
3. `src/different_vision_cones/runner.py --mode selfish`
4. `src/different_vision_cones/runner.py --mode auxiliary`

Dry run (print commands only):

```bash
python recreate_results.py --dry-run
```

## Run Individual Experiments

### Main runner (single cone)

From `src/` directory:

```bash
python runner.py --mode selfish
python runner.py --mode auxiliary
```

Outputs are written under `src/selfish_results/` and `src/auxiliary_results/` when run this way.

### Different vision cones runner

From `src/different_vision_cones/` directory:

```bash
python runner.py --mode selfish
python runner.py --mode auxiliary
```

Optional cone assignment controls (auxiliary mode):

```bash
python runner.py --mode auxiliary --cone_assign random --theta_deg2 220 --seed 42
python runner.py --mode auxiliary --cone_assign rows --theta_deg2 220
```

## Analysis

### Analyze main outputs

From project root:

```bash
python src/analyze.py --aux-json src/auxiliary_results/results_auxiliary.json --selfish-json src/selfish_results/results_selfish.json --output-dir src/analysis_outputs
```

This generates:
- `*_m_vs_T.png`
- `*_epr_vs_T.png`
- `*_tc_vs_theta.png`
- `*_tc_summary.csv`
- combined plot `combined_m_comparison.png`

### Estimate Tc for two-cone dataset

From `src/different_vision_cones/`:

```bash
python calculate_Tc.py
```

By default it reads `data.json` in the same folder.

## Notes

- Simulations are computationally heavy (large Monte Carlo steps). Runtime can be significant.
- Numba JIT compilation may make the first run slower; subsequent runs are faster.
- Runner scripts write outputs relative to the current working directory.
- The `recreate_results.py` script automatically sets each runner's working directory so outputs land in the expected `src/...` folders.
