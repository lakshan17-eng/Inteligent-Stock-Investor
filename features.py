# MODULE 2 - FEATURES: indicators that become the DQN's state
import pandas as pd

def rsi(p, n=14):
    d = p.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def make_features(prices):
    cols = []
    for t in prices.columns:
        p = prices[t]; r = p.pct_change()
        cols.append(pd.DataFrame({
            f"{t}_r1": r, f"{t}_r5": p.pct_change(5), f"{t}_r20": p.pct_change(20),
            f"{t}_vol": r.rolling(20).std(), f"{t}_ma": p / p.rolling(20).mean() - 1,
            f"{t}_rsi": rsi(p) / 100 - 0.5}))
    return pd.concat(cols, axis=1).dropna().clip(-1, 1)
