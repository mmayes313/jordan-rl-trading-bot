import argparse
import subprocess
from src.models.ppo_model import create_ppo_model
import streamlit as st  # For dashboard

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--dashboard', action='store_true')
parser.add_argument('--tune', type=int, default=50)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

if args.train:
    model = create_ppo_model('data/raw/eurusd_1y.csv')
    model.learn(total_timesteps=1000000)  # 100 days local
    model.save('data/models/ppo_model.zip')
    print("Training done!")
elif args.tune:
    from src.models.hyperparam_optimizer import run_optimization
    run_optimization('data/raw/eurusd_1y.csv', args.tune)
elif args.dashboard:
    exec(open('streamlit/app.py').read())  # Run dashboard
elif args.test:
    subprocess.run(['python', 'tests/run_all_tests.py'])
