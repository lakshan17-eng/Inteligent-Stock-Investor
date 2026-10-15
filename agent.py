# MODULE 4 - DQN AGENT: train, load, and pick a stock
import os, numpy as np, torch
from stable_baselines3 import DQN
from config import *
from data import download
from features import make_features
from env import TradingEnv, build_obs

def train(steps=300_000):
    prices = download()
    feats = make_features(prices).loc[:TRAIN_END]
    env = TradingEnv(prices, feats)
    model = DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=100_000, learning_starts=5000,
                batch_size=64, gamma=0.99, exploration_fraction=0.3, target_update_interval=1000,
                policy_kwargs=dict(net_arch=[128, 128]), verbose=1)
    model.learn(steps)
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print("Saved", MODEL_PATH)

def pick_stock(model, frow, period, allowed=None):
    """Return (index of best allowed stock to buy now, Q-values). `allowed` = list of stock indices."""
    n = len(TICKERS)
    p = min(period, MAX_DAYS)          # the model was trained on periods up to MAX_DAYS
    obs = torch.as_tensor(build_obs(frow, -1, 0.0, p, p, n)).unsqueeze(0)
    with torch.no_grad():
        q = model.policy.q_net(obs.to(model.device)).cpu().numpy()[0]
    buy = q[1:n + 1].copy()
    if allowed is not None:
        mask = np.full(n, -np.inf); mask[list(allowed)] = 0; buy = buy + mask
    return int(np.argmax(buy)), q

if __name__ == "__main__":
    train()