from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey


def update(key: PRNGKey, actor: Model, critic: Model, value: Model,
           batch: Batch, temperature: float, double: bool) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)

    q1, q2 = critic(batch.observations, batch.actions)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1
    exp_a = jnp.exp((q - v) * temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

def update_evaluation_policy(key: PRNGKey, actor: Model, critic: Model, value: Model,
           batch: Batch, temperature: float, double: bool) -> Tuple[Model, InfoDict]:
    """
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict[str, float]]:
        pi_dist = actor.apply_fn({"params": actor_params}, batch["observations"])
        log_probs = pi_dist.log_prob(batch["actions"])

        actions, _ = pi_dist.sample_and_log_prob(seed=key)
        qs = critic.apply_fn({"params": critic.params}, batch["observations"], actions)
        q = qs.mean(axis = 0)
        q = jnp.clip(q, a_max=100.0)

        actor_loss = (-q - temperature * log_probs).mean()
        return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)
    """

    #"""
    #v = value(batch.observations)

    #q = jnp.clip(q, a_max=1000.0)
    #exp_a = jnp.exp((q - v) * temperature)
    #exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:

        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        actions, _ = dist.sample_and_log_prob(seed=key)
        log_probs = dist.log_prob(batch.actions)

        q1, q2 = critic(batch.observations, actions)
        #if double:
        q = (q1 + q2) / 2.0
        #else:
        #    q = q1
        #actor_loss = -(exp_a * log_probs).mean()
        actor_loss = (-q - temperature * log_probs).mean()

        return actor_loss, {'evaluation_actor_loss': actor_loss, 'evaluation_adv': q - v}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    #"""
    return new_actor, info