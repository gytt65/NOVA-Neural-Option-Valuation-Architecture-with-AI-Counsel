import numpy as np

class StructuralFrictions:
    """
    Encodes the deterministic bounds and price distortions caused by SEBI's regulatory 
    framework and the structural mechanics of the Indian option market.
    """
    
    # Constants based on NSE/SEBI rules (as of late 2024/2025)
    STT_RATE_EXERCISED = 0.00125  # 0.125% on STT for exercised options (historically 0.125%, moving to 0.15% in some proposals, keeping 0.125% as base)
    
    def __init__(self, spot, strike, T, r, lot_size=75):
        self.spot = spot
        self.strike = strike
        self.T = T
        self.r = r
        self.lot_size = lot_size
        
        # Determine if option is Call or Put
        self.is_call = self.strike >= self.spot  # Rough check, but normally we pass an explicit flag. We will provide methods below where option_type is explicit.

    def apply_stt_exercise_distortion(self, option_type, intrinsic_value):
        """
        STT on exercised options is charged on the INTRINSIC value, not the premium.
        For slightly ITM options on expiry day, the STT cost can exceed the intrinsic profit.
        Returns the effective exercise value after STT friction.
        """
        if intrinsic_value <= 0:
            return 0.0
            
        # STT is only applicable on the sell side, but for options exercised, 
        # it is levied on the settlement price (intrinsic value) for the buyer.
        stt_cost = intrinsic_value * self.STT_RATE_EXERCISED
        
        effective_value = intrinsic_value - stt_cost
        
        # If effective value is negative, a rational actor will let it expire worthless
        return max(0.0, effective_value)

    def calculate_seller_margin_premium(self, iv, days_to_expiry, vix):
        """
        Calculates a volatility markup (IV spread) inversely proportional to the 
        required margin capital, capturing the 25-30x capital asymmetry between buyers and sellers.
        Sellers demand a higher premium to lock up ₹1.5L+ in SPAN margin.
        """
        # Baseline assumption: Sellers demand a ~5% higher IV equivalent when VIX is high and DTE is short
        base_markup = 0.005 # 50 bps base markup
        
        # Margin opportunity cost factor
        vix_factor = max(1.0, vix / 15.0)
        time_factor = 1.0 / np.sqrt(max(days_to_expiry, 1.0) / 365.0)
        
        margin_premium_iv = base_markup * vix_factor * min(time_factor, 10.0)
        return margin_premium_iv

    def overnight_gap_risk_adjustment(self, iv, is_overnight):
        """
        Separates the Variance Risk Premium (VRP) into intraday and overnight.
        If an option is held overnight, it commands an overnight risk premium due to the 
        17.75 hour non-trading gap.
        """
        if not is_overnight:
            # Intraday options do not strictly carry the overnight gap VRP
            # We decay the IV slightly to represent purely continuous trading hours
            return iv * 0.92 
        else:
            # Overnight options carry the full gap risk
            return iv * 1.05
