Learning the parameters of opinion-dynamics models from noisy observations, using a hidden-state model whose likelihood is estimated with a particle filter.

```
opinion_pf/
├── README.md                          # Project goal and file overview.
├── requirements.txt                   # Python dependencies.
├── .gitignore                         # Files and folders excluded from git.
├── configs/
│   └── degroot_base.yaml              # Parameters for the baseline DeGroot simulation.
├── src/
│   ├── __init__.py                    # Marks src as a Python package.
│   ├── config.py                      # Loads YAML config files into parameter dataclasses.
│   ├── model.py                       # Network, opinion dynamics and observation (noise) layers.
│   ├── simulate.py                    # Generates ground-truth hidden opinions and noisy observations.
│   └── particle_filter.py             # Particle-filter estimate of the log-likelihood.
├── checks/
│   └── check_kalman.py                # Compares the particle-filter likelihood with the exact Kalman likelihood (linear-Gaussian DeGroot).
└── notebooks/
    └── 01_simulate_and_filter.ipynb   # Simulate data and run the particle filter.
```
