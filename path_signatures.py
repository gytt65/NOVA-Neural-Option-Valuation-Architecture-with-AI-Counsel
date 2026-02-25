import numpy as np

try:
    # Requires iisignature or signatory for true path signatures
    import iisignature
    IISIGNATURE_AVAILABLE = True
except ImportError:
    IISIGNATURE_AVAILABLE = False


class PathSignatureExtractor:
    """
    Computes lead-lag rough path signatures from sequential market data.
    
    Path signatures (Terry Lyons' framework) capture the joint, non-commutative 
    trajectory of variables without parametric assumptions. 
    In the Indian market, this naturally encodes patterns like the "gamma blast" 
    or the trajectory of the spot price relative to the India VIX.
    """
    
    def __init__(self, truncation_level=2):
        self.truncation_level = truncation_level

    def _lead_lag_transform(self, X):
        """
        Transforms a multivariate path X of shape (T, D) into its 
        lead-lag enhanced version of shape (2T-1, 2D).
        This makes the signature capture quadratic variation and jump timings explicitly.
        """
        T_len, D = X.shape
        # Interleave elements to create the lead-lag step path
        ll_path = np.zeros((2 * T_len - 1, 2 * D))
        
        # Lead path (forward looking step)
        ll_path[0::2, :D] = X
        ll_path[1::2, :D] = X[1:]
        
        # Lag path (backward looking step)
        ll_path[0::2, D:] = X
        ll_path[1::2, D:] = X[:-1]
        
        return ll_path

    def compute_signature(self, spot_history, vix_history, pcr_history=None):
        """
        Computes the truncated signature of the 3D path: (Spot, VIX, PCR).
        
        Args:
            spot_history: Array of historical spot prices (length T)
            vix_history: Array of historical India VIX values (length T)
            pcr_history: Array of Put-Call Ratio values (length T). Optional.
            
        Returns:
            The truncated signature vector. 
        """
        spot = np.asarray(spot_history).reshape(-1, 1)
        vix = np.asarray(vix_history).reshape(-1, 1)
        
        if pcr_history is not None:
            pcr = np.asarray(pcr_history).reshape(-1, 1)
            path = np.hstack([spot, vix, pcr])
        else:
            path = np.hstack([spot, vix])
            
        # Normalize the path to start at origin
        path = path - path[0, :]
        
        # Apply lead-lag transform to capture quadratic variation
        ll_path = self._lead_lag_transform(path)
        
        if IISIGNATURE_AVAILABLE:
            # iisignature.sig returns the signature terms flattened
            sig = iisignature.sig(ll_path, self.truncation_level)
            return sig
        else:
            # Fallback for systems without iisignature
            # Compute a manual naive 1st order signature (just the integrals/increments)
            # and synthetic 2nd order cross-area terms
            increments = np.diff(ll_path, axis=0)
            level_1 = np.sum(increments, axis=0)
            
            # Simple manual calculation for level 2 (Levy area + half quadratic variation)
            level_2 = np.empty((ll_path.shape[1], ll_path.shape[1]))
            for i in range(ll_path.shape[1]):
                for j in range(ll_path.shape[1]):
                    # Riemann sum approx of integral X_i dX_j
                    integral_val = np.sum(ll_path[:-1, i] * increments[:, j])
                    level_2[i, j] = integral_val
                    
            # Flatten exactly like iisignature
            return np.concatenate([level_1, level_2.flatten()])
