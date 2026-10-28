

import jax
import jax.numpy as jnp
import numpy as np

from particle_gibbs.util import *




def exact_filter_and_smooth(initial_logp,           # x_0 -> log p(x_0)
                            transition_logp,        # t, x_{t}, x_{t+1} -> log f_t(x_{t+1} | x_t)
                            observation_logp,       # t, x_t -> log g_t(y_t | x_t)  
                            states,
                            T                       # Number of transitions, observations for t=0..T
                            ):
    # alpha_t = p(y_1,y_2,..,y_t,x_t)

    states = jnp.asarray(states, dtype=np.int32)

    # log_x_t^i = log p(x_t==i , y_{0:t})
    log_x = jax.vmap(lambda p : initial_logp(p) + observation_logp(0, p))(states) 

    init_lp = jax.scipy.special.logsumexp(log_x)

    init_filtered = log_x - init_lp

    def step_fn(carry, t):
        log_x = carry

        #pred_forward_log_x = jax.scipy.special.logsumexp(log_P + forward_log_x[:,None], axis=0)
        def new_log_p(new_state):

            log_P_to_j = jax.vmap(lambda s: transition_logp(t, s, new_state))(states)

            pred_log_x = jax.scipy.special.logsumexp(
                log_x + log_P_to_j
            ) # P(x_{t+1}==j | y_{0:t})= log sum_i x_t^{i} P_{i,j}

            # Could re-use pred_log_x in the smoother
            lp = observation_logp(t+1, new_state) # log P(y_{t+1} | x_{t+1}==j)
            return pred_log_x + lp

        log_x = jax.vmap(new_log_p)(states)

        log_filtered = log_x - jax.scipy.special.logsumexp(log_x)

        return log_x, log_filtered

    final_log_x, filtered_dist = jax.lax.scan(step_fn, log_x, jnp.arange(T))
    filtered_dist = jnp.concatenate((init_filtered[None], filtered_dist))
    lp = jax.scipy.special.logsumexp(final_log_x)



    # Smoothed distribution is P(x_t==i | y_{0:T})

    def smooth_backwards_step(carry, u):
        t, x = u
        forward_log_x = x  #  log P(x_t=i , y_{0:t})
        next_smoothed_log_x = carry

        log_P = jax.vmap(jax.vmap(lambda xt, xtp1: transition_logp(t, xt, xtp1), in_axes=(None,0)), in_axes=(0,None))(states, states)

        # P(x_{t+1}==i, y) = \sum_j P(x_{t+1}==i | x_{t}==j) P(x_{t}==j | y_{0:t})
        pred_forward_log_x = jax.scipy.special.logsumexp(log_P + forward_log_x[:,None], axis=0)
        # Note that this is calculated in the forward pass

        delta_smoothed_log_x = jnp.where(
            jnp.isneginf(pred_forward_log_x),
            -jnp.inf,
            next_smoothed_log_x - pred_forward_log_x,
        )

        smoothed_log_x = forward_log_x + jax.scipy.special.logsumexp(log_P + delta_smoothed_log_x[None,:], axis=1)
        smoothed_log_x = smoothed_log_x - jax.scipy.special.logsumexp(smoothed_log_x)    # Should be unnecessary
        return smoothed_log_x, smoothed_log_x

    _, smoothed_dist = jax.lax.scan(smooth_backwards_step, filtered_dist[-1], (jnp.arange(T), filtered_dist[:-1]), reverse=True)
    smoothed_dist = jnp.concatenate((smoothed_dist, filtered_dist[-1:]))

    return filtered_dist, smoothed_dist, lp

def exact_filter_and_smooth_mm(initial_logp,           # x_0 -> log p(x_0)
                               transition_logp,
                            observations,              # t, x_t -> log g(y_t | x_t)
                            states,                    # list of states
                            T,                         # Number of transitions - t=0...T
                         ):

    states = jnp.asarray(states, dtype=np.int32)

    log_x = jax.vmap(lambda p : initial_logp(p) + observations(0, p))(states) # x_t^i = p(x_t==i | y_{0:t})

    shift = jnp.max(log_x)
    x = jnp.exp(log_x - shift)
    s = jnp.sum(x)
    x = x/s
    lp = jnp.log(s) + shift


    def step_fn(carry, t):
        alpha = carry


        pred = P.T @ alpha
        obs_log_likelihood = jax.vmap(lambda s: observations(t+1, s))(states)

        shift = jnp.max(obs_log_likelihood)
        obs_lik = jnp.exp(obs_log_likelihood - shift)

        new_p = pred*obs_lik

        s = jnp.sum(new_p)

        new_p = new_p / s
        return new_p, (new_p, jnp.log(s) + shift)

    final_alpha, (filtered_dist, ls_hist)= jax.lax.scan(step_fn, x, jnp.arange(T))
    filtered_dist = jnp.concatenate((x[None], filtered_dist))

    lp = lp + jnp.sum(ls_hist)

    # Smoothed distribution is P(x_t==i | y_{0:T})

    def smooth_backwards_step(carry, x):
        forward_alpha = x #  log P(x_t=i | y_{0:t})
        next_smoothed_x = carry

        # P(x_{t+1}==i, y) = \sum_j P(x_{t+1}==i | x_{t}==j) P(x_{t}==j | y_{0:t})

        pred = P.T @ forward_alpha

        pred = jnp.where(pred == 0, 1.0, pred)


        ratio = jnp.divide(next_smoothed_x, pred)



        smoothed_alpha = forward_alpha * (P @ ratio)

        return smoothed_alpha, smoothed_alpha

    _, smoothed_dist = jax.lax.scan(smooth_backwards_step, final_alpha, filtered_dist[:-1], reverse=True)
    smoothed_dist = jnp.concatenate((smoothed_dist, final_alpha[None]))

    return filtered_dist, smoothed_dist, lp


