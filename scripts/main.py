#!/usr/bin/env python3
import argparse
import json
import os
import sys

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from scripts import GMEnv
from scripts import Tester

def parse_args():
    parser = argparse.ArgumentParser(description='Run RL trials with configurable paths')
    parser.add_argument('pl_file', type=str, help='Path to the Prolog file')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the config file')
    parser.add_argument('--sim-params', type=str, default='[1]',
                      help='Simulation parameters for semi-random simulation (default: [1])')
    parser.add_argument('--mode', type=str, choices=['simulate', 'train'], required=True,
                      help='Mode to run: simulate (run simulations only) or train (run training only)')
    return parser.parse_args()

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return json.load(f)

def run_simulation(pl_file, config, sim_params):
    """Run simulation mode with the given configuration."""
    env = GMEnv.GMEnv(pl_file)
    env.setDebug(config['debug'])
    env.setSeed(config['seed'])
    tester = Tester.TestIt(env)
    tester.debug = config['debug']
    
    results = {}
    
    # Run optimal simulation first (matching 3SBuild_Trials.py order)
    if config.get('simOptimalIter', 0) > 0:
        print("\nRunning optimal simulation...")
        results['optimal'] = tester.simulate(config['simOptimalIter'], config['optimalSimParams'])
    
    # Run random simulation with penalty forgiveness
    if config.get('simRandomIter', 0) > 0:
        print("\nRunning random simulation with penalty forgiveness...")
        results['random_forgive'] = tester.simulate(config['simRandomIter'])
    
    # Run random simulation without penalty forgiveness
    if config.get('simRandomIter', 0) > 0:
        print("\nRunning random simulation without penalty forgiveness...")
        results['random'] = tester.simulate(config['simRandomIter'], forgivePenalty=False)
    
    # Print results in the same format as 3SBuild_Trials.py
    print("\nSimulation Results:")
    if 'optimal' in results:
        print('DT-Golog - simulated policy reward.: {}'.format(results['optimal']))
    if 'random_forgive' in results:
        print('Random simulated policy reward (fg): {}'.format(results['random_forgive']))
    if 'random' in results:
        print('Random simulated policy reward.....: {}'.format(results['random']))
    
    env.closeQE()
    return results

def run_training(pl_file, config):
    """Run training mode with the given configuration."""
    env = GMEnv.GMEnv(pl_file)
    env.setDebug(config['debug'])
    env.setSeed(config['seed'])
    tester = Tester.TestIt(env)
    tester.debug = config['debug']
    
    print("\nStarting training...")
    result, params = tester.test_learning(
        config['trainingIter'],
        config['testingIter'],
        logging=config['learningLoggingInterval'],
        algo=config['learningAlgorithm']
    )
    
    # Print results in the same format as 3SBuild_Trials.py
    print("\nTraining Results:")
    print('Learned policy reward..............: {}'.format(result))
    print('--> Learning Parameters: \n {}'.format(params))
    
    env.closeQE()
    return result, params

def main():
    args = parse_args()
    
    # Validate paths
    if not os.path.exists(args.pl_file):
        print(f"Error: Prolog file not found: {args.pl_file}")
        sys.exit(1)
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.sim_params:
        config['simParams'] = eval(args.sim_params)
    
    if args.mode == 'simulate':
        run_simulation(args.pl_file, config, config.get('simParams', [1]))
    
    elif args.mode == 'train':
        run_training(args.pl_file, config)

if __name__ == '__main__':
    main()