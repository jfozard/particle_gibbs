



from typing import Any, NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.scipy.special


# Particle state needs to be a pytree -> {'xa':xa[n, :], 'xb':xb[n,:], 'e':e[n] }
# where n is the particle index



def make_conditional_filter_scan_step(       

            transition_sample,          # t, x_{t}, seed -> x_{t+1} ~ p_t(x_{t+1} | x_t)
            transition_logp,            # t, x_t, x_t_1 -> log p_t(x_{t+1}) | x_t) - used only with use_pgas
            observations,               # t, x_{t} -> log g(y_{0:T} | x_{t})
            reference_path,             # x'_{0:T}
            use_pgas = False
        ):

    
    def step_fn(carry, x):
        particles, logw = carry
        t, seed = x 

        M = logw.shape[0]

        logw_norm = logw - jax.scipy.special.logsumexp(logw)
        ess_pre = 1/jnp.sum(jnp.exp(2*logw_norm))

        # Resample particles
        seed, resample_seed = jax.random.split(seed)

        a_t_Mm1 = jax.random.categorical(resample_seed, logw, shape=(M-1,))

        next_ref_particle = jax.tree.map(lambda x: x[t+1], reference_path)

        if use_pgas:
            seed, pgas_seed = jax.random.split(seed)

            ancestor_log_score = jax.vmap(lambda p: transition_logp(t, p, next_ref_particle))(particles)

            pgas_logw = logw + ancestor_log_score
            
            a_t_M = jax.random.categorical(pgas_seed, pgas_logw)
            a_t_M = jnp.array([a_t_M], dtype=a_t_Mm1.dtype)
        else:
            a_t_M = jnp.array([M-1], dtype=a_t_Mm1.dtype)
        a_t = jnp.concatenate((a_t_Mm1, a_t_M))

        resampled_particles = jax.tree.map(lambda x: x[a_t_Mm1, ...], particles)

        seed, transition_seed = jax.random.split(seed)
        transition_seeds = jax.random.split(transition_seed, M-1)


        new_particles_t_Mm1 = jax.vmap(lambda p, s: transition_sample(t, p, s))(resampled_particles, transition_seeds)

        new_particles = jax.tree.map(lambda x, y: jnp.concatenate((x,y[None])) , new_particles_t_Mm1, next_ref_particle)

        # Calculate second-stage weights


        second_stage_weights = jax.vmap(lambda p: observations(t+1, p))(new_particles)

        new_logw = second_stage_weights
        new_logw_norm = new_logw - jax.scipy.special.logsumexp(new_logw)
        ess = 1/jnp.sum(jnp.exp(2*new_logw_norm))

        return (new_particles, new_logw), (new_particles, new_logw, a_t, ess, ess_pre)
    return step_fn


# Simple multiple sample from history.
def sample_particles(N_samples, seed, bpf_result):
    T = bpf_result.ancestor_history.shape[0]

    def sample_path(final_seed):

        I_T = jax.random.categorical(final_seed, bpf_result.logw_history[-1])

        def trace(i, a):
            return (a[i], a[i])

        _, path = jax.lax.scan(trace, I_T, bpf_result.ancestor_history, reverse=True)
        path = jnp.concatenate((path, I_T[None]))
        return path
    
    path_seeds = jax.random.split(seed, N_samples)
    path = jax.vmap(sample_path)(path_seeds)

    # Treat particle_history as a pytree?
    #sample = bpf_result.particle_history[jnp.arange(T+1)[None,:], path]
    sample = jax.tree.map(lambda u: u[jnp.arange(T+1)[None,:], path], bpf_result.particle_history)

    
    # Need to use tree_map as event_history can be a pytree.
    # sample_events = jax.tree_util.tree_map(lambda u: u[jnp.arange(T)[None,:], path[:,1:]], bpf_result.event_history)

    return sample
    
# Simple sample from history.
def sample_particle(final_seed, bpf_result):
    T = bpf_result.ancestor_history.shape[0]


    I_T = jax.random.categorical(final_seed, bpf_result.logw_history[-1])

    def trace(i, a):
        return (a[i], a[i])

    _, path = jax.lax.scan(trace, I_T, bpf_result.ancestor_history, reverse=True)
    path = jnp.concatenate((path, I_T[None]))

    sample = jax.tree.map(lambda u: u[jnp.arange(T+1), path], bpf_result.particle_history)

    
    # Need to use tree_map as event_history can be a pytree.
    # sample_events = jax.tree_util.tree_map(lambda u: u[jnp.arange(T)[None,:], path[:,1:]], bpf_result.event_history)

    return sample



class FilterAnalytics(NamedTuple):
    ess_history: Array
    ess_pre_history: Array


class CPFResult(NamedTuple):
    particle_history: Any
    logw_history: Array
    ancestor_history: Array
    filter_analytics: FilterAnalytics


def conditional_scan(          
        initial_sample,          # seed -> x_0 ~ p_0(x_0)
        transition_sample,       # t, x_{t}, seed -> x_{t+1} ~ p(x_{t+1} | x_t)
        transition_logp,         # t, x_t, x_{t+1} -> log p(x_{t+1}) | x_t) - used only with use_pgas
        observations,            # t, x_{t} -> log g(y_{t} | x_{t})          
        reference_path,          # x'_{0:T}
        T,                       # Number of transitions
        M,                       # Number of particles
        seed,                    # rng seed
        use_pgas=True,
        ):
    ### Initialize particles

    seed, init_seed = jax.random.split(seed)
    particle_seeds = jax.random.split(init_seed, M-1)
    particles = jax.vmap(initial_sample)(particle_seeds)

    particles = jax.tree.map(lambda x, y: jnp.concatenate((x,y[0][None]), axis=0) , particles, reference_path)

    logw = jax.vmap(lambda p: observations(0, p))(particles)

    logw_norm = logw - jax.scipy.special.logsumexp(logw)
    ess = 1/jnp.sum(jnp.exp(2*logw_norm))

    seed, scan_seed = jax.random.split(seed)
    scan_seeds = jax.random.split(scan_seed, T)

    step_fn = make_conditional_filter_scan_step(       
            transition_sample,
            transition_logp,
            observations,
            reference_path,
            use_pgas=use_pgas
        )

    (_final_particles, _final_logw), (particle_history, logw_history, ancestor_history, ess_history, ess_pre_history) = jax.lax.scan(step_fn, (particles, logw), (jnp.arange(T), scan_seeds))


    particle_history = jax.tree.map(
        lambda x, y: jnp.concatenate((x[None], y)),
            particles, particle_history)

    logw_history = jnp.concatenate((logw[None], logw_history))
    ess_history = jnp.concatenate((ess[None], ess_history))/M
    ess_pre_history = ess_pre_history/M

    return CPFResult(particle_history=particle_history,
                     logw_history=logw_history,
                     ancestor_history=ancestor_history,
                     filter_analytics=FilterAnalytics(ess_history=ess_history,
                                     ess_pre_history=ess_pre_history),
                     )

