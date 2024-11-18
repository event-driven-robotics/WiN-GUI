"""
surrogate_gradient.py

This module implements a custom autograd function for spiking nonlinearity with a surrogate gradient, 
which is essential for training spiking neural networks (SNNs) using backpropagation. The surrogate 
gradient is based on the normalized negative part of a fast sigmoid, as described in Zenke & Ganguli (2018).

Classes and Functions:
- SurrGradSpike: A custom autograd function that implements the spiking nonlinearity and surrogate gradient.
  - scale: A class attribute that defines the scaling factor for the surrogate gradient.
  - forward(ctx, input): Computes a step function of the input tensor and returns it. The input tensor is saved 
    for backward computation.
  - backward(ctx, grad_output): Computes the surrogate gradient of the loss with respect to the input using the 
    saved input tensor. The gradient is calculated using the normalized negative part of a fast sigmoid.
- activation: A function that applies the SurrGradSpike autograd function.

Usage:
- The SurrGradSpike class provides static methods for the forward and backward passes.
  - forward(ctx, input): Computes a step function of the input tensor and returns it. The input tensor is saved for 
    backward computation.
  - backward(ctx, grad_output): Computes the surrogate gradient of the loss with respect to the input using the saved 
    input tensor. The gradient is calculated using the normalized negative part of a fast sigmoid.

Example:
    import torch
    from surrogate_gradient import activation

    input_tensor = torch.tensor([0.5, -0.5, 0.0, 1.0], requires_grad=True)
    output_tensor = activation(input_tensor)
    output_tensor.backward(torch.ones_like(input_tensor))

Dependencies:
- torch: PyTorch library for tensor computations and neural network operations.

License:
This project is licensed under the GPL-3.0 License. See the LICENSE file for more details.

"""

import torch


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


activation = SurrGradSpike.apply
