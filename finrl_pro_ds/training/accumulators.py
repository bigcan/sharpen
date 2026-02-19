from typing import Callable, Optional
import torch.optim as optim
import logging

class GradientAccumulator:
    """
    Manages fractional gradient accumulation to support large global batch sizes 
    with limited environment throughput.
    
    Formula:
        Accumulation Steps = Global Batch Size / (Num Envs * Steps Per Loop)
    """
    def __init__(self, batch_size: int, num_envs: int, log_interval: int = 100):
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.current_samples = 0
        self.accumulation_steps = 0
        self.logger = logging.getLogger("GradientAccumulator")
        
        # Calculate optimal steps
        # If batch_size=4096, num_envs=24, we need ~170.6 steps.
        # We track 'samples' continuously.
        
    def should_step(self, current_step_samples: int = 0) -> bool:
        """
        Check if we should perform an optimization step.
        Adds new samples to counter.
        """
        self.current_samples += current_step_samples
        if self.current_samples >= self.batch_size:
            return True
        return False
        
    def reset(self):
        """Reset counter after optimization, carrying over excess."""
        self.current_samples = self.current_samples % self.batch_size

    def scale_loss(self, loss):
        """
        Scale loss by (1 / accumulation_steps) or similar factor if needed.
        In PPO, we usually normalize advantages, so the mean over the minibatch is stable.
        However, if we accumulate gradients over N forward passes, we sum gradients.
        So we should divide loss by (Total Steps / Single Step).
        
        Actually, simpler: 
        If we process mini-batches and Step(), we don't need this if we do standard PPO buffering.
        BUT, DeepScalper paper implies we might update generic networks differently?
        
        If we use a standard buffer (e.g. 4096 capacity), we just wait until buffer full.
        This class is useful if we are doing "Online" updates or DQN updates that are frequency based.
        
        For DQN: Update every C steps.
        
        Let's assume this is primarily for the DQN path or keeping generic logic.
        """
        # If we accumulate gradients, we typically divide by N steps.
        # But we are likely just flagging 'Time To Train'.
        pass
