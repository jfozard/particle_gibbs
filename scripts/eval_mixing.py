                                                                                                                                                                                                                                                                                                                                        
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

jax.config.update('jax_enable_x64', True)




def eval_mixing(seed_idx, N, T, sparse=False):

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


    N_particles = 10
    burn_in = 1_000
    N_pg_steps = 100

    def pg_scan_step_no_pgas(carry, x):
        reference_path = carry
        seed = x


        new_path = revised_pg_no_pgas(
            initial_sample,             # seed -> x_0 ~ p(x_{0})
            transition_sample,          # t, x_{t}, seed -> x_{t+1} ~ p(x_{t+1} | x_t)
            observation_logp,               # t, x_{t} -> log g_t(y_{t} | x_{t})
            reference_path,             # Path x'_{0:T}
            T, 
            N_particles,                          # Number of particles
            seed,                       # Base rng seed
        )

        return new_path, new_path
    
    def pg_scan_step(carry, x):
        reference_path = carry
        seed = x

        seed, final_particle_seed = jax.random.split(seed)
        seed, cpf_seed = jax.random.split(seed)
        
        cpf_result = conditional_scan(          
            initial_sample,          # seed -> x_0 ~ p_0(x_0)
            transition_sample,       # t, x_{t}, seed -> x_{t+1} ~ p(x_{t+1} | x_t)
            transition_logp,         # t, x_t, x_{t+1} -> log p(x_{t+1}) | x_t) - used only with use_pgas
            observation_logp,            # t, x_{t} -> log g(y_{t} | x_{t})          
            reference_path,          # x'_{0:T}
            T,                       # Number of transitions
            N_particles,                       # Number of particles
            cpf_seed,                    # rng seed
            use_pgas=True,
            )
        

        new_path = sample_particle(final_particle_seed, cpf_result)

        return new_path, new_path

    scan_seeds = jax.random.split(pg_seed, burn_in + N_pg_steps)

    _, all_paths = jax.lax.scan(pg_scan_step, starting_reference_path, scan_seeds)
    all_paths = all_paths[burn_in:]

    _, exact_smoothing_dist, _ = exact_filter_and_smooth(initial_logp, transition_logp, observation_logp, states, T)

    np.set_printoptions(precision=2)

    Nv = 5
    fig, ax = plt.subplots((Nv+4)//5, 5, figsize=(10,20))
    ax = ax.flatten()
    for i in range(Nv):
        ax[i].plot(all_paths[:,i])
        ax[i].set_title(exact_smoothing_dist[i])
    plt.savefig(f"mixing_{seed_idx}_{N}_{T}.png")

def main():
    #eval_mixing(1234, 4, 5, sparse=False)
    eval_mixing(123, 2, 20, sparse=False)


if __name__=="__main__":
    main()



