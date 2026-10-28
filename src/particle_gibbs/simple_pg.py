


import jax
from jax import Array
import jax.numpy as jnp
import jax.scipy.special



"""
Simple implementation - bpf, simply trace particles backwards, explicit for loops,
explicit multinomial resampling, assume particle states are plain Arrays.
"""



def simple_pg(
        initial_sample,             # seed -> x_0 ~ p(x_{0})
        transition_sample,          # t, x_{t}, seed -> x_{t+1} ~ p(x_{t+1} | x_t)
        observations,               # t, x_{t} -> log g_t(y_{t} | x_{t})
        reference_path,             # Path x'_{0:T}
        T, 
        M,                          # Number of particles
        seed,                       # Base rng seed
):

    seed, final_particle_seed = jax.random.split(seed) # Save this seed for later so that we can be consistent with scan version
    seed, init_seed = jax.random.split(seed)
    init_particle_seeds = jax.random.split(init_seed, M-1)
    particles_1_Mm1 = jax.vmap(initial_sample)(init_particle_seeds)
    particles = jnp.concatenate((particles_1_Mm1, reference_path[0][None]))
    logw = jax.vmap(lambda p: observations(0, p))(particles)

    seed, step_seed = jax.random.split(seed)
    step_seeds = jax.random.split(step_seed, T)
    ancestors = []
    particle_hist = [particles]

    for t in range(T):
        
        seed = step_seeds[t]

        seed, resample_seed = jax.random.split(seed)

        a_t_1_Mm1 = jax.random.categorical(resample_seed, logw, shape=(M-1,))
        
        a_t_M = jnp.array([M-1], dtype=a_t_1_Mm1.dtype)
        a_t = jnp.concatenate((a_t_1_Mm1, a_t_M))

        ancestors.append(a_t)
        
        resampled_particles = particles[a_t_1_Mm1] 

        seed, transition_seed = jax.random.split(seed)
        transition_seeds = jax.random.split(transition_seed, M-1)

        new_particles_t_1_Mm1 = jax.vmap(lambda p, s: transition_sample(t, p, s))(resampled_particles, transition_seeds)
        
        next_ref_particle = reference_path[t+1]
        particles = jnp.concatenate((new_particles_t_1_Mm1, next_ref_particle[None]))

        particle_hist.append(particles)
        
        # Reweight according to observations at time t+1

        logw = jax.vmap(lambda p: observations(t+1, p))(particles)


    # sample a single particle from its geneaology.
 
    I_T = jax.random.categorical(final_particle_seed, logw)

    sample = [particle_hist[T][I_T]]
    idx = I_T
    
    for t in range(T-1, -1, -1):
        idx = ancestors[t][idx]
        sample.append(particle_hist[t][idx])

    sample.reverse()

    sample = jnp.stack(sample)
    return sample
        

