Learning the parameters of opinion-dynamics models from noisy observations, using a hidden-state model whose likelihood is estimated with a particle filter.

```
learning_od_from_data/
├── README.md                          # Project goal and file overview.
├── requirements.txt                   # Python dependencies.
├── .gitignore                         # Files and folders excluded from git.
├── docs/
│   └── PROJECT_PLAN.md                # Project plan: phases, conventions, decisions and open questions.
├── configs/
│   └── degroot_base.yaml              # Parameters for the baseline DeGroot simulation.
├── src/
│   ├── __init__.py                    # Marks src as a Python package.
│   ├── config.py                      # Loads YAML config files into parameter dataclasses.
│   ├── model.py                       # Network, opinion dynamics and observation (noise) layers.
│   ├── simulate.py                    # Generates ground-truth hidden opinions and noisy observations.
│   ├── kalman.py                      # Exact Kalman-filter likelihood (joint and per-agent), the reference for the particle filter.
│   └── particle_filter.py             # Particle-filter estimate of the log-likelihood (joint and per-agent).
├── checks/
│   ├── check_joint_filter.py          # Joint particle filter vs. the exact joint Kalman likelihood.
│   ├── check_per_agent_filter.py      # Per-agent particle filter vs. the per-agent Kalman filter.
│   └── check_plugin_bias.py           # Bias from plug-in neighbour inputs (per-agent vs. joint Kalman).
├── notebooks/
│   ├── 01_sanity_checks.ipynb         # Sanity checks of the model, simulator and Kalman filter.
│   └── 02_simulate_and_filter.ipynb   # Simulation and filtering exploration (empty for now).
└── results/                           # Generated figures and configs (git-ignored); checks write to results/checks/<name>/.
```
