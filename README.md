# RL Gen
Title: _Tool and Reproducibility Package for: Model-driven Design and Generation of Training Simulators for Reinforcement Learning_



## Overview
This repository contains the scripts and examples that accompany the conference submission "Model-driven Design and Generation of Domain Simulators for Reinforcement Learning". 

The python scripts implementing GMEnv and and Query Interface can be found in `/scripts`
The example goal models and various tests and experiments against them can be found in `/examples`


## Prerequisites
- Python 3.13 (latest) - This project has been tested with Python 3.13 and may not work with other versions
- SWI-Prolog (for running the DT-Golog code)
- Git

## Installation and Setup Instructions

To ensure a reproducible environment, it is recommended to use a Python virtual environment. Follow these steps:

1. **Clone the repository locally.**
   ```bash
   git clone <repository-url>
   cd RLGen
   ```

2. **Verify Python version and create virtual environment:**
   ```bash
   # Check Python version
   python --version  # Should show Python 3.13.x
   ```
   ```bash
   # Create virtual environment with Python 3.13
   python -m venv venv
    ```
   
   ```bash
   
   source venv/bin/activate  # On Linux/Mac
   # OR
   venv\Scripts\activate     # On Windows
   ```

3. **Install the required dependencies:**
   ```bash
   # Install project dependencies
   python -m pip install -r requirements.txt
   ```

   This will install all necessary packages with the correct versions for this project.

4. **Acquire the DT-Golog code** from [its creator's page](https://www.cs.ryerson.ca/~mes/publications/appendix/appendixC/dtgolog), and place it in a file called `DT-Golog.pl` under `/scripts/QE/`

5. **Make the following changes so that it runs on SWI-Prolog**
  - Comment out the following:
    ```
    /* :- pragma(debug).  */
    ```
  - Add:
    ```
    :- op(900, fy, [not]).
    (not X) := (\+ X).
    cputime(5).
    ```
6. **Run the test and trial scripts in the `/examples` folder from your IDE.**

## The Models

* Several models have been developed for tests and experiments. They can be reviewed in GoalModels.drawio which can be opened using https://app.diagrams.net/
* The corresponding specifications can be found in the `.pl` file in the `/examples` folder. The same model may have both a discrete and a continuous implementation (which are different only in one line of code whereby the state space is specified).
* Files named `[XXX]_Tests.py` contain simple python `unittest` tests.
* Files named `[XXX]_Trials.py` contain simulation and learning experiments. 
  * For running simulations or learning be sure to give meaningful iteration numbers to `simRandomIter`, `simOptimalRandomIter`, `trainingIter` (number of training steps) `testingIter` (number of testing episodes). `10,000` is a good number to start with.
  * `learningAlgorithm` can be one of `A2C`, `PPO`, or `DQN` implemented as part of [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html)

## Running Simulations and Training

The project provides a unified command-line interface through `scripts/main.py` for running both simulations and training. This replaces the need to run individual trial scripts directly.

### Command Line Interface

```bash
  python scripts/main.py <pl_file> --mode {simulate,train} --config <config_file> [--sim-params <params>]
```

Required arguments:
- `pl_file`: Path to the Prolog file containing the domain specification (e.g., `examples/discrete/3Build.pl`)
- `--mode`: Operation mode, either `simulate` or `train`
- `--config`: Path to the JSON configuration file (e.g., `scripts/config.json`)

Optional arguments:
- `--sim-params`: Simulation parameters for semi-random simulation (default: `[1]`)

### Configuration File

The configuration file (`config.json`) controls various parameters for both simulation and training modes. Here's an example configuration:

```json
{
    "debug": false,
    "seed": 123,
    "simRandomIter": 100,
    "simCustomIter": 100,
    "simOptimalIter": 100,
    "testingIter": 1000,
    "trainingIter": 1000,
    "learningAlgorithm": "PPO",
    "learningLoggingInterval": 500,
    "optimalSimParams": [0, 2],
    "dtGologOptimal": 0.476
}
```

Configuration parameters:
- `debug`: Enable/disable debug output
- `seed`: Random seed for reproducibility
- `simRandomIter`: Number of iterations for random simulation
- `simCustomIter`: Number of iterations for custom simulation
- `simOptimalIter`: Number of iterations for optimal simulation
- `testingIter`: Number of testing episodes for training
- `trainingIter`: Number of training steps
- `learningAlgorithm`: Learning algorithm to use (`A2C`, `PPO`, or `DQN`)
- `learningLoggingInterval`: Interval for logging during training
- `optimalSimParams`: Parameters for optimal simulation
- `dtGologOptimal`: Expected optimal reward value

### Example Usage

1. Running simulations:
```bash
python scripts/main.py examples/discrete/3Build.pl --mode simulate --config scripts/config.json
```

2. Running training:
```bash
python scripts/main.py examples/discrete/3Build.pl --mode train --config scripts/config.json
```

The script will output results in a format consistent with the original trial scripts, including:
- For simulation mode:
  - DT-Golog simulated policy reward
  - Random simulated policy reward (with and without penalty forgiveness)
- For training mode:
  - Learned policy reward
  - Learning parameters

# Contact

Please send questions, issues, bugs and recommendations to [liaskos@yorku.ca](mailto:liaskos@yorku.ca?Subject=RLGen).

  
    

    

