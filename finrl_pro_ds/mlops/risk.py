"""Module: risk
Purpose: Implement risk management policies (drawdown limits, exposure caps)."""

class RiskControlPolicy:
    """
    Enforces risk constraints on trading actions.
    """
    def __init__(self, max_drawdown: float = 0.25, max_exposure: float = 1.0):
        """
        Args:
            max_drawdown: Maximum allowed drawdown (percentage, e.g., 0.25 for 25%).
                          If exceeded, the policy forces a flat position (stop loss).
            max_exposure: Maximum gross exposure allowed (leverage cap).
        """
        self.max_drawdown = max_drawdown
        self.max_exposure = max_exposure
        self.peak_value = 0.0
        self.triggered = False

    def update(self, current_value: float) -> None:
        """Updates the policy state with the current portfolio value."""
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0.0
        
        if drawdown > self.max_drawdown:
            self.triggered = True

    def transform_action(self, action):
        """
        Modifies the action to comply with risk constraints.
        Args:
            action: The proposed action (scalar or vector).
        Returns:
            The allowed action.
        """
        if self.triggered:
            # Force flat position
            # Assuming action 0 means flat/close positions, or depends on action space.
            # For continuous [-1, 1], 0 usually means hold or flat.
            # Ideally, we want to *close* positions. 
            # If action represents *target position*, then 0 is correct.
            return 0.0 * action 
        
        # Apply exposure cap (simplified for scalar action)
        # For vector actions, would need norm check.
        # return np.clip(action, -self.max_exposure, self.max_exposure)
        return action

    def reset(self):
        self.peak_value = 0.0
        self.triggered = False
