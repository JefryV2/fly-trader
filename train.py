import argparse
import os
import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import FlyTradeEnv
from fly_brain import FlyBrainPolicy

def main():
    parser = argparse.ArgumentParser(description="Train FlyTrader with PPO")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker, e.g., AAPL")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--window", type=int, default=30, help="Observation window size")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Training timesteps")
    parser.add_argument("--use-connectome", action="store_true", help="Use real FlyWire v783 connectome weights")
    parser.add_argument("--connectome-dir", type=str, default="flywire_data", help="FlyWire data directory")
    parser.add_argument("--weight-cache", type=str, default="flywire_weights", help="Cached weight directory")
    parser.add_argument("--N-neurons", type=int, default=1024, help="Number of neurons (default 1024 for backward compatibility)")
    args = parser.parse_args()

    # Ensure data exists
    data_path = os.path.join("data", f"{args.ticker}_daily.csv")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}. Run data_fetch.py first.")

    # Create environment
    env = DummyVecEnv([lambda: FlyTradeEnv(Path(data_path), window_size=args.window)])

    # Policy that wraps the fly brain simulation
    policy_kwargs = dict(
        features_extractor_class=FlyBrainPolicy,
        features_extractor_kwargs={
            "use_connectome": args.use_connectome,
            "connectome_dir": args.connectome_dir,
            "weight_cache": args.weight_cache,
            "N_NEURONS": args.N_neurons,
        },
    )

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=args.total_timesteps)
    model.save(f"flytrader_{args.ticker}.zip")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()