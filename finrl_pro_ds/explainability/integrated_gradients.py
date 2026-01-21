"""Module: integrated_gradients
Purpose: Provide Integrated Gradients-based explainability for RL agents."""

from __future__ import annotations
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from typing import Callable

class IntegratedGradientsExplainer:
    """
    A class to explain predictions of a PyTorch model using Integrated Gradients.
    Assumes the model is a PyTorch `nn.Module` and its `forward` method
    takes a tensor and returns a tensor.
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        """
        Initializes the IntegratedGradientsExplainer.

        Args:
            model: The PyTorch model (nn.Module) to explain.
            device: The device to run the attribution on ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.model.eval()  # Set model to evaluation mode
        self.ig = IntegratedGradients(self.model)
        self.device = device

    def attribute(self, input_tensor: torch.Tensor, target: int | None = None,
                  baselines: torch.Tensor | None = None,
                  n_steps: int = 50, internal_batch_size: int | None = None) -> torch.Tensor:
        """
        Computes Integrated Gradients attributions for the given input.

        Args:
            input_tensor: The input tensor for which to compute attributions.
                          Shape should be (batch_size, features).
            target: The target output index for which to compute attributions.
                    Required for multi-output models.
            baselines: A baseline input, or a tensor of baselines to use.
                       If None, a zero tensor of the same shape as input_tensor is used.
            n_steps: The number of steps used in the approximation of the integral.
            internal_batch_size: Divides the `n_steps` into batches for processing.
                                 If None, uses `n_steps` as a single batch.

        Returns:
            A tensor containing the attributions, with the same shape as input_tensor.
        """
        input_tensor = input_tensor.to(self.device)
        if baselines is not None:
            baselines = baselines.to(self.device)

        attributions = self.ig.attribute(
            inputs=input_tensor,
            baselines=baselines,
            target=target,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size
        )
        return attributions
