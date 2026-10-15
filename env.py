# MODULE 3 - TRADING ENVIRONMENT (Gymnasium)
# Actions: 0 Hold | 1..N Buy stock i | N+1 Sell. Episode = one random investment period.
# Reward = (return of held stock, or 0 if flat) - equal-weight basket return - trading costs.
import numpy as np, gymnasium as gym
from gymnasium import spaces
from config import COST, MIN_DAYS, MAX_DAYS

def build_obs(frow, held, pnl, days_left, period, n):
    pos = np.zeros(n, dtype=np.float32)
    if held >= 0: pos[held] = 1
    return np.concatenate([frow, pos, [pnl, days_left / MAX_DAYS, period / MAX_DAYS]]).astype(np.float32)

class TradingEnv(gym.Env):
    def __init__(self, prices, feats):
        super().__init__()
        self.p = prices.loc[feats.index].values
        self.f = feats.values.astype(np.float32)
        self.T, self.n = self.p.shape
        self.action_space = spaces.Discrete(self.n + 2)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.f.shape[1] + self.n + 3,), np.float32)

    def _obs(self):
        pnl = self.p[self.t, self.held] / self.entry - 1 if self.held >= 0 else 0.0
        return build_obs(self.f[self.t], self.held, pnl, self.end - self.t, self.L, self.n)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.L = int(self.np_random.integers(MIN_DAYS, MAX_DAYS + 1))
        self.t = int(self.np_random.integers(0, self.T - self.L - 1))
        self.end, self.held, self.entry = self.t + self.L, -1, 0.0
        return self._obs(), {}

    def step(self, a):
        r = 0.0
        if self.held < 0 and 1 <= a <= self.n:
            self.held, self.entry = a - 1, self.p[self.t, a - 1]; r -= COST
        elif self.held >= 0 and a == self.n + 1:
            self.held = -1; r -= COST
        nxt = self.t + 1
        basket = (self.p[nxt] / self.p[self.t] - 1).mean()
        if self.held >= 0:
            r += self.p[nxt, self.held] / self.p[self.t, self.held] - 1
        r -= basket
        self.t = nxt
        done = self.t >= self.end
        if done and self.held >= 0:
            r -= COST; self.held = -1      # forced sell at end of period
        return self._obs(), float(r * 100), done, False, {}
