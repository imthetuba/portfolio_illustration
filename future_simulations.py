import numpy as np
import pandas as pd

def monte_carlo_simulation(returns, start_value=100000, n_years=10, n_simulations=100):
    n_months = int(n_years * 12)
    mean_return = returns.mean()
    std_return = returns.std()
    simulations = np.zeros((n_months, n_simulations))
    for i in range(n_simulations):
        monthly_returns = np.random.normal(mean_return, std_return, n_months)
        simulations[:, i] = start_value * np.cumprod(1 + monthly_returns)
    return simulations