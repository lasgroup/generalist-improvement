import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)

def test_pi0_sample_actions_shapes():
    import jax.numpy as jnp
    config = _pi0_config.Pi0Config(
        action_dim=4,
        action_horizon=6,
        max_token_len=8,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )

    model = config.create(jax.random.key(0))
    obs_spec, action_spec = config.inputs_spec(batch_size=2)
    obs = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), obs_spec)
    noise = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), action_spec)

    x_0, info = jax.eval_shape(
        lambda: model.sample_actions(
            observation=obs,
            noise=noise,
            num_steps=3,
            rng=None,
            noise_level=0.0,
            return_info_dict=True,
        )
    )

    assert x_0.shape == (2, config.action_horizon, config.action_dim)
    assert info["x_next"].shape == (3, 2, config.action_horizon, config.action_dim)
    assert info["x"].shape == (3, 2, config.action_horizon, config.action_dim)
    assert info["log_prob"].shape == (3, 2, config.action_horizon)
    assert info["time"].shape == (3,)
    assert info["time_next"].shape == (3,)