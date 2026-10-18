# SVJD filters on option-implied cumulants: KF-DNF, KF (Feunou–Okou) and Particle Filter

Python + C++ implementation of the KF–DNF filter of Bégin and Zafari (2025) [BZ], the Kalman filter of Feunou and Okou (2018) [FO] and a particle filter, for the three-factor affine stochastic-volatility jump-diffusion model of Andersen, Fusari and Todorov [AFT] observed through model-implied cumulants (orders 2, 3 and 4 at 1, 3, 6 and 12 months, i.e. 12 observables).

The scripts implement the simulation study of **Section 4 of BZ** (Figure 1 and Tables 1–4). The empirical application of Section 5 (S&P 500 option data) is not included. The settings used here are summarised in [Implementation notes](#implementation-notes).

| Filter | C++ class (module `kfdnf_fast`) | Description |
|---|---|---|
| **KF-DNF** | `DiscreteNonlinearFilter` | Kalman filter conditioned on a grid of `K` jump-size nodes for each of the two negative-jump variables (`(K+1)²` combinations including "no jump"); the node estimates are collapsed with the tower property and the law of total variance (BZ, Algorithm 1) |
| **KF** (Feunou–Okou) | `FeunouOkouKF` | Modified linear Kalman filter of FO: AFT transition of FO Eq. 27–28 with the conditional covariance evaluated at the filtered state; filtered values floored at 1e-8 |
| **PF** | `ParticleFilter` | Auxiliary particle filter (the jump occurrence is drawn from a proposal that uses the current observation) with systematic resampling |

## Project structure

```text
.
├── CMakeLists.txt              # build of the C++ module
├── requirements.txt            # Python dependencies
├── cpp/
│   ├── bindings.cpp            # pybind11 bindings (module kfdnf_fast)
│   └── kfdnf_core.cpp          # KF-DNF, KF (Feunou–Okou) and particle filter (Eigen)
├── src/
│   ├── cumulant_engine.py      # model-implied cumulants: Numba RK4 on the ODE system of BZ Supplementary Material, Section SM.B
│   ├── main_calibration.py     # observation / transition matrices and DNF grid; running it executes a smoke test on dummy data
│   └── data_generator.py       # synthetic latent paths (Euler–Maruyama with jumps)
└── scripts/
    ├── sim_tables_1_2_3.py     # Monte-Carlo study (500 paths) -> Simulation_results.csv
    ├── report_tables_1_2_3.py  # Figure 1 and Tables 1-3 (console + LaTeX)
    ├── mle_table_4.py          # maximum likelihood estimation of 20 parameters -> MLE_<N>traj_<C>cycles.csv
    └── report_table_4.py       # Table 4: bias / RMSE and log-likelihood comparison
```

The compiled module `kfdnf_fast` is built into `src/`, next to the Python modules.

## What is implemented from BZ

| BZ element | Scripts | Output |
|---|---|---|
| Figure 1: relative RMSE of the log-likelihood vs grid size K | `sim_tables_1_2_3.py` → `report_tables_1_2_3.py` | `Figure_1_RRMSE_Simulation_results.pdf/.png` |
| Table 1: filtered vs simulated factors (RMSE ratios to the PF) | same | console and `tables_output_Simulation_results.tex` |
| Table 2: summary statistics of true and filtered states (K = 5) | same | same |
| Table 3: computation times (seconds) | same | same |
| Table 4: MLE bias and RMSE, KF-DNF vs KF | `mle_table_4.py` → `report_table_4.py` | console |

The "true" parameters of the simulation study are the values of BZ Table 4.

## Requirements

* Linux or macOS with GCC or Clang (on Windows use WSL: the compiler flags in `CMakeLists.txt` are GCC/Clang flags)
* Python 3, a C++ compiler, CMake ≥ 3.12, [Eigen](https://eigen.tuxfamily.org) ≥ 3.3
* The Python packages in `requirements.txt` (pybind11 is only needed to build the module)

Tested on Ubuntu 24.04 with Python 3.12.3, GCC 13.3, CMake 3.28.3, Eigen 3.4.0, numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, numba 0.68.0, joblib 1.6.0, matplotlib 3.11.2 and pybind11 3.1.0.

## Installation

```bash
# 1. System packages
sudo apt install build-essential cmake libeigen3-dev python3-venv   # Debian / Ubuntu
brew install cmake eigen                                            # macOS (Homebrew)

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Build the C++ module (creates src/kfdnf_fast*.so)
cmake -S . -B build -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build
```

CMake does not find a pip-installed pybind11 on its own, hence the `-Dpybind11_DIR` hint. If Eigen lives in a non-standard location add `-DEigen3_DIR=<path to the folder containing Eigen3Config.cmake>`.
Build on the machine where you run the code (see `-march=native` in the build notes).

## Usage

Run the commands from the repository root; the scripts read and write their csv / figure / LaTeX files in the current directory.

```bash
python src/main_calibration.py                         # smoke test (dummy data): checks that the module builds and runs
python scripts/sim_tables_1_2_3.py                     # Monte-Carlo study -> Simulation_results.csv
python scripts/report_tables_1_2_3.py                  # Figure 1 + Tables 1-3 (console and tables_output_*.tex)
python scripts/mle_table_4.py                          # MLE -> MLE_<N>traj_<C>cycles.csv (re-run to resume)
python scripts/report_table_4.py MLE_250traj_50cycles.csv   # bias / RMSE table; the argument is optional
```

Settings you may want to change (constants at the top / in the `__main__` block of each script):

* `sim_tables_1_2_3.py`: `N_TRAJECTORIES` = 500 and 2520 days per path (as in BZ), `n_cores` = 6, 10 000 PF particles.
* `mle_table_4.py`: `N_TRAJECTORIES` = 250 and `N_CYCLES` = 50 by default, (as in BZ), `N_DAYS` = 2520; the worker pool uses `cpu_count() - 2` processes.

## Implementation notes

### Settings relative to BZ

BZ refers to the in-press version of the paper.

| | This repository | BZ |
|---|---|---|
| PF benchmark | auxiliary PF, 10 000 particles, systematic resampling | PF following Johannes et al. (2009), 100 000 particles, stratified resampling (Section 4.1) |
| DNF grid width δ_K | 3 + log K for K > 1 (δ_1 = 0) | log(K)/2 (Section 3.2) |
| MLE optimiser | SciPy Nelder–Mead, then `trust-constr` (box bounds, normalised parameters) | MATLAB `fminsearch`, then `fmincon` (BFGS) |
| Time step of the simulated paths | `steps_per_day` = 1 (Monte-Carlo study), 10 (MLE) | daily step, h = 1/252 (Section 2.3) |
| Initial state of the simulated factors | `v0` argument of `generate_high_fidelity_data`, default (ν̄₁, ν̄₂, 10⁻⁴) | not stated in Section 4 |

PF log-likelihoods, RMSEs and run times (Figure 1, Tables 1–3) are therefore not directly comparable with the paper's.

### Build notes

* `CMakeLists.txt` defaults to a `Release` build (`-O3 -DNDEBUG`), which switches off Eigen's run-time assertions.
* `-ffast-math` is combined with `-fno-finite-math-only`, so that `std::isnan` and `Eigen::allFinite()` checks stay active (plain `-ffast-math` lets GCC 13 drop them). Disable fast-math with `-DKFDNF_FAST_MATH=OFF`.
* `-march=native` ties the module to the CPU family of the machine that built it; disable with `-DKFDNF_NATIVE=OFF`.
* Run times (Table 3) depend on these flags and on the hardware: compare filters within one build and one machine.

### Reproducibility

* `np.random.seed()` called from Python does not seed Numba's generator, which produces the latent paths in `data_generator.py`; use `data_generator.seed_all(seed)` to seed both.
* `ParticleFilter` is seeded from `std::random_device` and the Monte-Carlo / MLE scripts do not fix a seed, so those runs are reproducible in distribution rather than bit by bit. KF-DNF and KF are deterministic given the data.

## References

The papers are not redistributed here; see the DOIs.

* **[BZ]** J.-F. Bégin and G. Zafari, "Estimation of multifactor stochastic volatility jump-diffusion models: A marginalized filter approach", *Econometrics and Statistics* (2025), doi:10.1016/j.ecosta.2025.03.002.
* **[FO]** B. Feunou and C. Okou, "Risk-neutral moment-based estimation of affine option pricing models", *Journal of Applied Econometrics* 33(7), 2018, pp. 1007–1025, doi:10.1002/jae.2630.
* **[AFT]** T. G. Andersen, N. Fusari and V. Todorov, "The risk premia embedded in index options", *Journal of Financial Economics* 117(3), 2015, pp. 558–584.
