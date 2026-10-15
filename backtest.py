# MODULE 5 - BACKTEST: DQN vs equal-weight buy-and-hold on dates the model never saw
import json, os, numpy as np
from stable_baselines3 import DQN
from config import *
from data import download
from features import make_features
from agent import pick_stock

def run():
    prices = download(); feats = make_features(prices); px = prices.loc[feats.index].values
    model = DQN.load(MODEL_PATH)
    test = np.where(feats.index >= TEST_START)[0]
    out = {}
    for L in (10, 20, 30, 60):
        ai, base = [], []
        for t in test:
            if t + L >= len(px): break
            s, _ = pick_stock(model, feats.values[t], L)
            ret = px[t + L] / px[t] - 1
            ai.append(ret[s] - 2 * COST); base.append(ret.mean() - 2 * COST)
        ai, base = np.array(ai), np.array(base)
        out[str(L)] = dict(trials=len(ai), dqn_avg_pct=round(ai.mean() * 100, 2), basket_avg_pct=round(base.mean() * 100, 2),
                           dqn_win_rate_pct=round((ai > 0).mean() * 100, 1), beats_basket_pct=round((ai > base).mean() * 100, 1),
                           dqn_worst_pct=round(ai.min() * 100, 2))
    os.makedirs("reports", exist_ok=True)
    json.dump(out, open("reports/backtest.json", "w"), indent=2)
    for L, v in out.items(): print(f"{L} days:", v)

if __name__ == "__main__":
    run()
