from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import flax.nnx as nnx
import jax
from jax.sharding import Sharding
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        self.action_dim = model.action_dim
        self.action_horizon = model.action_horizon

        if hasattr(model, "get_prefix_rep") and callable(model.get_prefix_rep):
            self._get_prefix_rep = nnx_utils.module_jit(model.get_prefix_rep)
        else:

            def get_prefix_rep(*args, **kwargs):
                return None

            self._get_prefix_rep = get_prefix_rep

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._sample_actions_with_model = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions,
                static_argnames=(
                    "noise_level",
                    "return_info_dict",
                    "num_steps",
                    "return_prefix_rep",
                ),
            )

            def _sample_actions_with_model_fn(
                m: _model.BaseModel,
                *,
                observation: _model.Observation,
                noise: jnp.ndarray,
                num_steps: int | jnp.ndarray = 10,
                rng: at.KeyArrayLike | None = None,
                noise_level: float = 0.0,
                return_info_dict: bool = False,
                return_prefix_rep: bool = False,
            ):
                return m.sample_actions(
                    observation=observation,
                    noise=noise,
                    num_steps=num_steps,
                    rng=rng,
                    noise_level=noise_level,
                    return_info_dict=return_info_dict,
                    return_prefix_rep=return_prefix_rep,
                )

            self._sample_actions_with_model = nnx.jit(
                _sample_actions_with_model_fn,
                static_argnames=(
                    "noise_level",
                    "return_info_dict",
                    "num_steps",
                    "return_prefix_rep",
                ),
            )

            def _get_prefix_rep_with_model_fn(
                m: _model.BaseModel,
                *,
                observation: _model.Observation,
            ):
                prefix_rep = m.get_prefix_rep(observation)
                return prefix_rep[0]

            self._get_prefix_rep_with_model = nnx.jit(_get_prefix_rep_with_model_fn)

            self._rng = rng or jax.random.key(0)

    def _jax_inference(
        self,
        inputs: dict,
        noise: jnp.ndarray | np.ndarray | None = None,
        model: _model.BaseModel | None = None,
        noise_level: float = 0.0,
        return_info_dict: bool = False,
        return_prefix_rep: bool = False,
        sharding_spec: Sharding | None = None,
    ) -> dict:  # type: ignore[misc]
        # Make a batch and convert to jax.Array.
        batch_keys_skip = ["tokenized_prompt", "tokenized_prompt_mask", 
                           "token_ar_mask", "token_loss_mask"]

        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]

            def _add_batch_dim(x):
                return jnp.broadcast_to(x[jnp.newaxis, ...], (batch_size,) + x.shape)

            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            for key in inputs:
                if key not in ["image", "state"]:
                    # Skip if already batched for tokenized prompt keys
                    if key in batch_keys_skip and inputs[key].ndim == 2:
                        continue
                    inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
        else:
            batch_size = 1
            for key in inputs:
                # Skip if already batched for tokenized prompt keys
                if key in batch_keys_skip and inputs[key].ndim == 2:
                    continue
                inputs[key] = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs[key])
        # self._rng, sample_rng = jax.random.split(self._rng)
        if noise is None:
            self._rng, sample_rng = jax.random.split(self._rng)
            noise = jax.random.normal(
                sample_rng, (batch_size, self.action_horizon, self.action_dim)
            )
        if sharding_spec is not None:
            inputs = jax.device_put(inputs, sharding_spec)
        observation = _model.Observation.from_dict(inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        start_time = time.monotonic()
        if model is None:
            actions = self._sample_actions(
                observation=observation,
                rng=sample_rng,
                noise=noise,
                noise_level=noise_level,
                return_info_dict=return_info_dict,
                return_prefix_rep=return_prefix_rep,
                **self._sample_kwargs,
            )
        else:
            actions = self._sample_actions_with_model(
                m=model,
                observation=observation,
                rng=sample_rng,
                noise=noise,
                noise_level=noise_level,
                return_info_dict=return_info_dict,
                return_prefix_rep=return_prefix_rep,
                **self._sample_kwargs,
            )
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time

        # Unbatch and convert to np.ndarray.
        if batch_size == 1:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _torch_inference(
        self, inputs: dict, noise: jnp.ndarray | np.ndarray | None = None
    ) -> dict:  # type: ignore[misc]
        # Make a batch and convert to jax.Array.
        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]

            def _add_batch_dim(x):
                return np.broadcast_to(x[np.newaxis, ...], (batch_size,) + x.shape)

            inputs = jax.tree.map(lambda x: np.asarray(x), inputs)
            for key in inputs:
                if key not in ["image", "state"]:
                    inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
        else:
            batch_size = 1
            inputs = jax.tree.map(lambda x: np.asarray(x)[np.newaxis, ...], inputs)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs
        )
        sample_rng_or_pytorch_device = self._pytorch_device
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device)

            if (
                noise.ndim == 2
            ):  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(
                sample_rng_or_pytorch_device, observation, **sample_kwargs
            ),
        }
        model_time = time.monotonic() - start_time
        if batch_size == 1:
            # Unbatch and convert to np.ndarray.
            outputs = jax.tree.map(
                lambda x: np.asarray(x[0, ...].detach().cpu()), outputs
            )
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: jnp.ndarray | np.ndarray | None = None,
        sharding_spec: Sharding | None,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        if self._is_pytorch_model:
            return self._torch_inference(inputs, noise)
        else:
            return self._jax_inference(
                inputs=inputs, noise=noise, sharding_spec=sharding_spec
            )

    def infer_with_model(
        self,
        model: _model.BaseModel,
        obs: dict,
        *,
        noise: jnp.ndarray | None = None,
        noise_level: float = 0.0,
        return_info_dict: bool = False,
        return_prefix_rep: bool = False,
        sharding_spec: Sharding | None,
    ) -> tuple[dict, jnp.ndarray] | dict:
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if self._is_pytorch_model:
            return self._torch_inference(inputs, noise)
        else:
            return self._jax_inference(
                inputs=inputs,
                noise=noise,
                model=model,
                noise_level=noise_level,
                return_info_dict=return_info_dict,
                return_prefix_rep=return_prefix_rep,
                sharding_spec=sharding_spec,
            )

    @override
    def get_prefix_rep_with_model(
        self, model: _model.BaseModel, obs: dict
    ) -> at.Float[at.Array, "batch prefix_seq embed"]:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        # add batch dim and broadcast for keys that are not "images" and "state"
        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]

            def _add_batch_dim(x):
                return jnp.broadcast_to(x[jnp.newaxis, ...], (batch_size,) + x.shape)

            for key in inputs:
                if key not in ["image", "state"]:
                    inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
        else:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return self._get_prefix_rep_with_model(
            model, _model.Observation.from_dict(inputs)
        )

    @override
    def get_prefix_rep(self, obs: dict) -> at.Float[at.Array, "batch prefix_seq embed"]:
        return self.get_prefix_rep_with_model(
            self._model, _model.Observation.from_dict(obs)
        )[0]

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
