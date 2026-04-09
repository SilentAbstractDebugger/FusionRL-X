import numpy as np
from typing import Optional


class LogReturnReward:
   
    def __init__(self):
        self.prev_portfolio_value: Optional[float] = None

    def reset(self, initial_value: float):
        self.prev_portfolio_value = initial_value

    def compute(self, current_value: float) -> float:
        if self.prev_portfolio_value is None or self.prev_portfolio_value <= 0:
            reward = 0.0
        else:
            # Clamp to avoid log(0) or negative portfolio values
            ratio = max(current_value / self.prev_portfolio_value, 1e-8)
            reward = np.log(ratio)

        self.prev_portfolio_value = current_value
        return float(reward)


class DSRReward:

    def __init__(self, eta: float = 1.0 / 252):
        self.eta = eta
        self.X   = 0.0   # EMA of returns (first moment)
        self.Y   = 0.0   # EMA of squared returns (second moment)

    def reset(self):
        self.X = 0.0
        self.Y = 0.0

    def compute(self, r_t: float) -> float:
        X_prev = self.X
        Y_prev = self.Y

        # Update EMAs
        delta_X = r_t - X_prev
        delta_Y = r_t**2 - Y_prev

        self.X = X_prev + self.eta * delta_X
        self.Y = Y_prev + self.eta * delta_Y

        # Compute DSR
        numerator   = Y_prev * delta_X - 0.5 * X_prev * delta_Y
        denominator = (Y_prev - X_prev**2)

        if denominator <= 1e-10:
            return 0.0

        dsr = numerator / (denominator ** 1.5)

        # Clip to prevent extreme outliers destabilizing training
        dsr = float(np.clip(dsr, -10.0, 10.0))
        return dsr


class MDDReward:

    def __init__(self):
        self.peak_value: float = 0.0

    def reset(self, initial_value: float):
        self.peak_value = initial_value

    def compute(self, current_value: float) -> float:
        # Update peak if new high
        if current_value > self.peak_value:
            self.peak_value = current_value

        if self.peak_value <= 0:
            return 0.0

        # Current drawdown from peak
        drawdown = (self.peak_value - current_value) / self.peak_value
        drawdown = float(np.clip(drawdown, 0.0, 1.0))

        # Negative drawdown = reward (agent is punished for drawdowns)
        return -drawdown

def get_reward_fn(reward_type: str):
    mapping = {
        "log_return": LogReturnReward,
        "dsr":        DSRReward,
        "mdd":        MDDReward,
    }
    if reward_type not in mapping:
        raise ValueError(f"Unknown reward type: {reward_type}. Choose from {list(mapping.keys())}")
    return mapping[reward_type]()

if __name__ == "__main__":
    print("=== Testing Reward Functions ===\n")

    # Simulate a portfolio that grows then crashes
    portfolio_values = [1_000_000, 1_010_000, 1_025_000, 1_020_000,
                        1_050_000, 1_040_000, 1_030_000, 1_060_000]

    # Log Return
    lr = LogReturnReward()
    lr.reset(portfolio_values[0])
    print("Log Return Rewards:")
    for v in portfolio_values[1:]:
        print(f"  Portfolio ${v:>10,} → reward = {lr.compute(v):+.6f}")

    print()

    # DSR
    dsr = DSRReward()
    dsr.reset()
    print("DSR Rewards:")
    prev = portfolio_values[0]
    for v in portfolio_values[1:]:
        r_t = np.log(v / prev)
        reward = dsr.compute(r_t)
        print(f"  r_t = {r_t:+.5f} → DSR = {reward:+.6f}")
        prev = v

    print()

    # MDD
    mdd = MDDReward()
    mdd.reset(portfolio_values[0])
    print("MDD Rewards (negative drawdown):")
    for v in portfolio_values[1:]:
        print(f"  Portfolio ${v:>10,} → reward = {mdd.compute(v):+.6f}")
