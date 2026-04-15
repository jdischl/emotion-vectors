"""
Forward-hook utilities for activation capture and representation steering.

These provide clean context-manager APIs that handle hook registration and
cleanup automatically.  They work with any HuggingFace transformers model
whose decoder layers live at ``model.model.layers``.

Methodology note (Anthropic, 2026):
    The residual stream at a given layer is the sum of the original embedding
    plus all attention and MLP outputs up to that point.  Capturing the layer
    output gives us exactly this residual stream value.  For steering, we add
    a direction vector to the residual stream, which causally shifts all
    downstream computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config as _config


class ActivationCapture:
    """Capture residual-stream activations at specified layers during a forward pass.

    Usage::

        with ActivationCapture(model, layer_indices=[17, 29, 47]) as cap:
            model(**inputs)
            acts = cap.get()  # {17: tensor(batch, seq, d_model), ...}
    """

    def __init__(self, model: nn.Module, layer_indices: list[int]):
        self.model = model
        self.layer_indices = layer_indices
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._activations: dict[int, torch.Tensor] = {}

    # -- context manager --------------------------------------------------

    def __enter__(self) -> "ActivationCapture":
        self._activations.clear()
        layers = _config.get_decoder_layers(self.model)

        for idx in self.layer_indices:
            # Closure over idx so each hook writes to the right key
            def make_hook(layer_idx: int):
                def hook_fn(module, input, output):
                    # Decoder layers return a tuple; element 0 is the hidden state
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._activations[layer_idx] = hidden.detach()
                return hook_fn

            handle = layers[idx].register_forward_hook(make_hook(idx))
            self._hooks.append(handle)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        return False  # don't suppress exceptions

    # -- public API -------------------------------------------------------

    def get(self) -> dict[int, torch.Tensor]:
        """Return captured activations as ``{layer_idx: tensor}``."""
        return dict(self._activations)

    def clear(self):
        """Discard captured activations (useful between batches)."""
        self._activations.clear()


class SteeringHook:
    """Add a scaled direction vector to the residual stream at a single layer.

    Usage::

        with SteeringHook(model, layer_idx=29, vector=vec, alpha=0.1):
            output = model.generate(**inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        vector: torch.Tensor,
        alpha: float = 0.1,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.vector = vector.detach().clone()
        self.alpha = alpha
        self._hook = None

    def __enter__(self) -> "SteeringHook":
        layer = _config.get_decoder_layers(self.model)[self.layer_idx]
        vec = self.vector.to(
            device=next(layer.parameters()).device,
            dtype=next(layer.parameters()).dtype,
        )
        alpha = self.alpha

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden + alpha * vec
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        self._hook = layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        return False


class MultiLayerSteeringHook:
    """Add a scaled direction vector to the residual stream at ALL layers.

    Jeong (2026) found that distributing the steering perturbation across all
    layers (via ``hook_resid_post`` at every layer) produces more stable and
    coherent outputs than concentrating it at a single layer.  Each layer gets
    the same ``alpha * vector`` addition, so the total perturbation is
    ``alpha * num_layers`` but spread through the network.

    Usage::

        with MultiLayerSteeringHook(model, vector=vec, alpha=0.01):
            output = model.generate(**inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        vector: torch.Tensor,
        alpha: float = 0.01,
    ):
        self.model = model
        self.vector = vector.detach().clone()
        self.alpha = alpha
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def __enter__(self) -> "MultiLayerSteeringHook":
        layers = _config.get_decoder_layers(self.model)
        alpha = self.alpha

        for layer in layers:
            vec = self.vector.to(
                device=next(layer.parameters()).device,
                dtype=next(layer.parameters()).dtype,
            )

            def make_hook(v):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    hidden = hidden + alpha * v
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook_fn

            handle = layer.register_forward_hook(make_hook(vec))
            self._hooks.append(handle)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        return False
