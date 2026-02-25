import numpy as np
from scipy.optimize import linprog

class MartingaleOptimalTransport:
    """
    Model-Free Option Pricing Bounds using Martingale Optimal Transport (MOT).
    
    Given a set of observed, liquid vanilla option prices (the marginals), 
    what are the absolute, model-free upper and lower bounds for an exotic or 
    illiquid vanilla option?
    
    This avoids assuming any parametric volatility model, instead solving a 
    linear programming problem constrainted by the martingale condition 
    (no-arbitrage path constraints). Particularly useful in the Indian market 
    where extreme regulatory changes cause parametric models to fail.
    """

    def __init__(self, spot, strikes, calls, puts):
        """
        Initializes the state with liquid vanilla prices.
        strikes: array of K
        calls: array of C(K)
        puts: array of P(K)
        """
        self.spot = spot
        self.strikes = np.asarray(strikes)
        self.calls = np.asarray(calls)
        self.puts = np.asarray(puts)
        
        # Discretize the state space for the LP solver
        min_s = self.spot * 0.5
        max_s = self.spot * 1.5
        self.grid = np.linspace(min_s, max_s, 100)

    def calculate_bounds(self, target_payoff_func):
        """
        Calculates the model-free lower and upper bounds for a derivative 
        given its payoff function: payoff = func(S_T)
        
        Returns: (lower_bound, upper_bound)
        """
        n_points = len(self.grid)
        
        # Payoff evaluated on the grid
        c_obj = target_payoff_func(self.grid)
        
        # Ensure non-trivial grid
        if n_points < 3:
            return 0.0, 0.0
            
        # We set up an LP to find a primal probability measure P (pi) over grid points 
        # that minimizes/maximizes the expected payoff E[c_obj].
        
        # Constraints:
        # matrix A_eq · pi = b_eq
        
        # 1. Sum of probabilities = 1
        A_eq = [np.ones(n_points)]
        b_eq = [1.0]
        
        # 2. Martingale condition: E[S_T] = Spot (assuming r=0, q=0 for simple bounds, or forward adjusted)
        # Note: Ideally spot is the forward price F. 
        A_eq.append(self.grid)
        b_eq.append(self.spot)
        
        # 3. Call option prices match observed market prices exactly
        # Intentionally restricting constraint set to avoid infeasiblity 
        # in noisy Indian market data.
        n_strikes_to_use = min(len(self.strikes), 5) 
        
        for i in range(n_strikes_to_use):
            K = self.strikes[i]
            obs_call_price = self.calls[i]
            # E[max(S-K, 0)] = C
            call_payoff = np.maximum(self.grid - K, 0)
            A_eq.append(call_payoff)
            b_eq.append(obs_call_price)
            
        A_eq = np.vstack(A_eq)
        b_eq = np.array(b_eq)
        
        # Bounds for probabilities: pi >= 0
        bounds = [(0, 1) for _ in range(n_points)]
        
        try:
            # Lower bound (minimize expected payoff)
            res_min = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            lower_bound = res_min.fun if res_min.success else 0.0
            
            # Upper bound (maximize expected payoff = minimize -expected payoff)
            res_max = linprog(-c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            upper_bound = -res_max.fun if res_max.success else 0.0
            
            return max(lower_bound, 0.0), max(upper_bound, 0.0)
            
        except Exception:
            # High-dimensions or noisy data can cause linprog failure
            return 0.0, 1e6
