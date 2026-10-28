                                                                                                                                                                                                                                                                                                                                        
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import jax.scipy.special

from particle_gibbs.exact import *
from particle_gibbs.revised_pg import *
from particle_gibbs.simple_pg import *

from particle_gibbs.util import *
from particle_gibbs.visualize import *

import pytest

jax.config.update('jax_enable_x64', True)


def enumerate_filter_and_smooth(N, T, initial_logp, transition_logp, observation_logp):

    states = jnp.arange(N)

    def score_path(x):
        lp = initial_logp(x[0]) + observation_logp(0, x[0])
        for t in range(T):
            lp += transition_logp(t, x[t], x[t+1]) + observation_logp(t+1, x[t+1])
        return lp
    
    def make_path(i):
        o = [i%N]
        r = i
        for t in range(T):
            r = r//N
            o.append(r%N)
        return jnp.array(o)
    

    all_paths = jax.vmap(make_path)(jnp.arange(N**(T+1)))

    all_scores = jax.vmap(score_path)(all_paths)

    enumerated_lp = jax.scipy.special.logsumexp(all_scores)

    def one_hot_contrib(x, lp):
        return jnp.exp(lp - enumerated_lp)* jax.nn.one_hot(x, N)

    smoothing_dist = jnp.sum(jax.vmap(one_hot_contrib)(all_paths, all_scores), axis=0)


    def filtering_contributions(x):
        lp = initial_logp(x[0]) + observation_logp(0, x[0])
        res = [jnp.where(states == x[0], lp, -jnp.inf)]
        for t in range(T):
            lp += transition_logp(t, x[t], x[t+1]) + observation_logp(t+1, x[t+1])
            res.append(jnp.where(states == x[t+1], lp, -jnp.inf))
        return jnp.stack(res)

    log_filtering_dist = jax.scipy.special.logsumexp(jax.vmap(filtering_contributions)(all_paths), axis=0)

    log_filtering_dist = log_filtering_dist - jax.scipy.special.logsumexp(log_filtering_dist, axis=1, keepdims=True)

    filtering_dist = jnp.exp(log_filtering_dist)

    return filtering_dist, smoothing_dist, enumerated_lp

def compare_exact_and_enumerate(N, T, initial_logp, transition_logp, observation_logp):

    states = jnp.arange(N)

    filtered_dist, smoothed_dist, lp = exact_filter_and_smooth(initial_logp,           # x_0 -> log p(x_0)
                            transition_logp,
                            observation_logp,       # x_t, y_t -> log p(y_t | x_t)
                            states,
                            T
                            )
    
    # Calculate the filtering and scoring distributions by enumerating them all

    enumerate_filtered_dist, enumerate_smoothed_dist, enumerate_lp = enumerate_filter_and_smooth(
        N, T, initial_logp, transition_logp, observation_logp
    )

    np.testing.assert_allclose(enumerate_filtered_dist, jnp.exp(filtered_dist))

    np.testing.assert_allclose(enumerate_smoothed_dist, jnp.exp(smoothed_dist))

    np.testing.assert_allclose(enumerate_lp, lp)    


@pytest.mark.parametrize("seed_idx,N,T,sparse",[[0,2,2,False]])
def test_exact_and_enumerate(seed_idx, N, T, sparse):

    seed = jax.random.key(seed_idx)

    states = jnp.arange(N)

    seed, initial_seed = jax.random.split(seed)

    initial_P = jax.random.uniform(initial_seed, shape=(N,), minval=0.1)
    initial_P = initial_P / jnp.sum(initial_P)
    initial_log_P = jnp.log(initial_P)

    def initial_logp(x):
        return initial_log_P[x]
    
    seed, P_seed = jax.random.split(seed)

    P = jax.random.uniform(P_seed, shape=(T,N,N), minval=0.1)



    if sparse: # Insert some zeros into the transition matrix
        seed, keep_seed = jax.random.split(seed)
        keep = jax.random.bernoulli(keep_seed, p=0.5, shape = P.shape)
        idx = jnp.arange(N)
        keep = keep.at[:, idx, idx].set(True)

        P = jnp.where(keep, P, 0.0)



    P = P / jnp.sum(P, axis=2, keepdims=True)
    log_P = jnp.log(P)

    def transition_logp(t, x, xtp1):
        return log_P[t, x, xtp1]

    seed, obs_seed = jax.random.split(seed)

    obs_P = jax.random.uniform(obs_seed, shape=(T+1,N), minval=0.1)

    if sparse: # Insert some zeros into the transition matrix
        seed, keep_obs_seed = jax.random.split(seed)
        keep = jax.random.bernoulli(keep_obs_seed, p=0.5, shape = obs_P.shape)

        obs_P = jnp.where(keep, obs_P, 0.0)

    obs_log_P = jnp.log(obs_P)




    def observation_logp(t, x):
        return obs_log_P[t, x]


    filtered_dist, smoothed_dist, lp = exact_filter_and_smooth(initial_logp,           # x_0 -> log p(x_0)
                            transition_logp,
                            observation_logp,       # x_t, y_t -> log p(y_t | x_t)
                            states,
                            T
                            )
    
    # Calculate the filtering and scoring distributions by enumerating them all

    enumerate_filtered_dist, enumerate_smoothed_dist, enumerate_lp = enumerate_filter_and_smooth(
        N, T, initial_logp, transition_logp, observation_logp
    )

    np.testing.assert_allclose(enumerate_filtered_dist, jnp.exp(filtered_dist))

    np.testing.assert_allclose(enumerate_smoothed_dist, jnp.exp(smoothed_dist))

    np.testing.assert_allclose(enumerate_lp, lp)    


def test_enumerate_filter():
    N = 2
    T = 2

    init_logP = jnp.log(jnp.array([ 0.5, 0.5 ]))

    def initial_logp(x):
        return init_logP[x]

    log_P = jnp.log(jnp.array([[0, 1], [1, 0]]))

    def transition_logp(t, x, xtp1):
        return log_P[x, xtp1]

    o_array = jnp.log(jnp.array([[0.5, 0.5], [0.5, 0.5], [0.0, 1.0]]))

    def observation_logp(t, x):
        lp = o_array[t, x]
        return lp

    enumerate_filtered_dist, enumerate_smoothed_dist, enumerate_lp = enumerate_filter_and_smooth(
        N, T, initial_logp, transition_logp, observation_logp
    )

    np.testing.assert_allclose(enumerate_filtered_dist, jnp.array([[0.5, 0.5], [0.5, 0.5], [0.0, 1.0]]))

    np.testing.assert_allclose(enumerate_smoothed_dist, jnp.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))


@pytest.mark.parametrize("method", ["enumerate", "exact"])    
def test_filter_time_dependent(method):
    N = 3
    T = 2

    init_logP = jnp.log(jnp.ones(N)/N)

    def initial_logp(x):
        return init_logP[x]

    log_P_same = jnp.log(jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    log_P_swap = jnp.log(jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))

    log_P = jnp.stack((log_P_same, log_P_swap))

    log_P = log_P - jax.scipy.special.logsumexp(log_P, axis=2, keepdims=True)

    def transition_logp(t, x, xtp1):
        return log_P[t, x, xtp1]

    o_array = jnp.log(jnp.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]]))

    def observation_logp(t, x):
        lp = o_array[t, x]
        return lp

    if method == "enumerate":
        filtered_dist, smoothed_dist, lp = enumerate_filter_and_smooth(
            N, T, initial_logp, transition_logp, observation_logp
        )
    else:
        filtered_dist, smoothed_dist, lp = exact_filter_and_smooth(initial_logp,           # x_0 -> log p(x_0)
                            transition_logp,
                            observation_logp,       # x_t, y_t -> log p(y_t | x_t)
                            jnp.arange(N),
                            T
                            )
        filtered_dist = jnp.exp(filtered_dist)
        smoothed_dist = jnp.exp(smoothed_dist)    

    uniform = jnp.array([1/3, 1/3, 1/3])

    np.testing.assert_allclose(filtered_dist, jnp.array([uniform, uniform, [1.0, 0.0, 0.0]]))

    np.testing.assert_allclose(smoothed_dist, jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))

    np.testing.assert_allclose(
        lp, jnp.log(1 / 3),
    )

@pytest.mark.parametrize("method", ["enumerate", "exact"])    
def test_filter_impossible(method):
    N = 2
    T = 1

    init_logP = jnp.log(jnp.array([ 0.5, 0.5 ]))

    def initial_logp(x):
        return init_logP[x]

    log_P = jnp.log(jnp.array([[0, 1], [1, 0]]))

    def transition_logp(t, x, xtp1):
        return log_P[x, xtp1]

    o_array = jnp.log(jnp.array([[0.5, 0.5], [0.0, 0.0]]))

    def observation_logp(t, x):
        lp = o_array[t, x]
        return lp
    init_logP = jnp.log(jnp.ones(N)/N)

    
    if method == "enumerate":
        filtered_dist, smoothed_dist, lp = enumerate_filter_and_smooth(
            N, T, initial_logp, transition_logp, observation_logp
        )
    else:
        filtered_dist, smoothed_dist, lp = exact_filter_and_smooth(initial_logp,           # x_0 -> log p(x_0)
                            transition_logp,
                            observation_logp,       # x_t, y_t -> log p(y_t | x_t)
                            jnp.arange(N),
                            T
                            )
        filtered_dist = jnp.exp(filtered_dist)
        smoothed_dist = jnp.exp(smoothed_dist)    


    np.testing.assert_allclose(filtered_dist[0], jnp.array([0.5, 0.5]))

    assert np.all(np.isnan(filtered_dist[1]))
    assert np.all(np.isnan(smoothed_dist))
    assert np.isneginf(lp)


def pg_smoothing(pg_seed, N, T, initial_sample, transition_sample, observation_logp, starting_reference_path, N_particles=10, N_pg_steps=10000, burn_in=1000):

    def pg_scan_step(carry, x):
        reference_path = carry
        seed = x

        new_path = simple_pg(
            initial_sample,             # seed -> x_0 ~ p(x_{0})
            transition_sample,          # t, x_{t}, seed -> x_{t+1} ~ p(x_{t+1} | x_t)
            observation_logp,               # t, x_{t} -> log g_t(y_{t} | x_{t})
            reference_path,             # Path x'_{0:T}
            T, 
            N_particles,                          # Number of particles
            seed,                       # Base rng seed
        )

        return new_path, new_path

    scan_seeds = jax.random.split(pg_seed, burn_in + N_pg_steps)

    _, all_paths = jax.lax.scan(pg_scan_step, starting_reference_path, scan_seeds)
    all_paths = all_paths[burn_in:]
    empirical_smooth = jnp.sum(jax.vmap(lambda p: jax.nn.one_hot(p, N))(all_paths), axis=0)
    empirical_smooth = empirical_smooth / jnp.sum(empirical_smooth, axis=1, keepdims=True)

    return empirical_smooth


@pytest.mark.parametrize("seed_idx,N,T,sparse",[[0,2,2,False], [1, 3, 3, True]])
def test_pg_vs_exact(seed_idx, N, T, sparse):
    seed = jax.random.key(seed_idx)
 
    states = jnp.arange(N)
 
    seed, initial_seed = jax.random.split(seed)
 
    initial_P = jax.random.uniform(initial_seed, shape=(N,), minval=0.1)
    initial_P = initial_P / jnp.sum(initial_P)
    initial_log_P = jnp.log(initial_P)


 
    def initial_logp(x):
        return initial_log_P[x]

    def initial_sample(s):
        return jax.random.categorical(s, initial_log_P)

    seed, P_seed = jax.random.split(seed)
 
    P = jax.random.uniform(P_seed, shape=(T,N,N), minval=0.1)
 
    
 
    if sparse: # Insert some zeros into the transition matrix
        seed, keep_seed = jax.random.split(seed)
        keep = jax.random.bernoulli(keep_seed, p=0.5, shape = P.shape)
        idx = jnp.arange(N)
        keep = keep.at[:, idx, idx].set(True)
 
        P = jnp.where(keep, P, 0.0)
 
 
 
    P = P / jnp.sum(P, axis=2, keepdims=True)
    log_P = jnp.log(P)

    def transition_sample(t, x, seed):
        return jax.random.categorical(seed, log_P[t, x])
    
    def transition_logp(t, x, xtp1):
        return log_P[t, x, xtp1]
 
    seed, obs_seed = jax.random.split(seed)
 
    obs_P = jax.random.uniform(obs_seed, shape=(T+1,N), minval=0.1)
 
    if sparse: # Insert some zeros into the transition matrix
        seed, keep_obs_seed = jax.random.split(seed)
        keep = jax.random.bernoulli(keep_obs_seed, p=0.5, shape = obs_P.shape)
 
        obs_P = jnp.where(keep, obs_P, 0.0)
 
    obs_log_P = jnp.log(obs_P)
 
 
    def observation_logp(t, x):
        return obs_log_P[t, x]
        

    seed, init_sample_seed = jax.random.split(seed)
    
    initial_pos = initial_sample(init_sample_seed)

    
    def step(carry, x):
        pos = carry
        t, seed = x
        new_pos = transition_sample(t, pos, seed)
        return new_pos, new_pos


    seed, sample_seed = jax.random.split(seed)
    scan_seeds = jax.random.split(sample_seed, T)
    _, all_pos = jax.lax.scan(step, initial_pos, (jnp.arange(T), scan_seeds))

    starting_reference_path = jnp.concatenate((initial_pos[None], all_pos))

    seed, pg_seed = jax.random.split(seed)

    pg_smoothing_dist = pg_smoothing(pg_seed, N, T, initial_sample, transition_sample, observation_logp, starting_reference_path)

    _, exact_smoothing_dist, _ = exact_filter_and_smooth(initial_logp, transition_logp, observation_logp, states, T)

    np.testing.assert_allclose( pg_smoothing_dist, jnp.exp(exact_smoothing_dist), atol=0.02, rtol=0.1)

