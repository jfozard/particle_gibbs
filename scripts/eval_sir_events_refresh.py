                                                                                                                                                                                                                                                                                                                                        

import jax
import jax.numpy as jnp
import jax.scipy.special
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from particle_gibbs.exact import *
from particle_gibbs.refresh_pg import *
from particle_gibbs.util import *

jax.config.update('jax_enable_x64', True)

def make_SIR_model(alpha, gamma, timestep):
    def SI_rate(x):
        return alpha*x[1]/jnp.sum(x)

    def IR_rate(x):
        return gamma

    def transition_sample(_t, x, seed):
        p_SI = -jnp.expm1(-timestep*SI_rate(x))
        p_IR = -jnp.expm1(-timestep*IR_rate(x))
        seed, k = jax.random.split(seed)
        n_SI = jax.random.binomial(k, x[0], p_SI)
        seed, k = jax.random.split(seed)
        n_IR = jax.random.binomial(k, x[1], p_IR)
        S = x[0]
        I = x[1]
        return jnp.array([x[0] - n_SI, x[1] + n_SI - n_IR, x[2] + n_IR], dtype=x.dtype)

    def transition_logp(_t, x_old, x_new):
        
        r_SI = SI_rate(x_old)
        r_IR = IR_rate(x_old)

        p_SI = -jnp.expm1(-r_SI*timestep)
        p_IR = -jnp.expm1(-r_IR*timestep)
        S = x_old[0]
        I = x_old[1]
        R = x_old[2]
        n_SI = x_old[0] - x_new[0]
        n_IR = x_new[2] - x_old[2]

        expected_I_new = I + n_SI - n_IR

        invalid = (
            (n_SI < 0)
            | (n_IR < 0)
            | (n_SI > S)
            | (n_IR > I)
            | (x_new[1] != expected_I_new)
        )
        
        logp = lchoose(S, n_SI) + xlogy(n_SI, p_SI) + xlogy((S - n_SI), (1-p_SI))
        logp += lchoose(I, n_IR) + xlogy(n_IR, p_IR) + xlogy((I - n_IR), (1-p_IR)) 
        return jax.lax.select(invalid, -jnp.inf, logp)

    return transition_sample, transition_logp


def make_SIR_model_events(alpha, gamma, timestep):
    def SI_rate(x):
        return alpha*x[1]/jnp.sum(x)

    def IR_rate(x):
        return gamma

    def transition_sample(_t, p, seed):
        x = p['x']
        p_SI = -jnp.expm1(-timestep*SI_rate(x))
        p_IR = -jnp.expm1(-timestep*IR_rate(x))
        seed, k = jax.random.split(seed)
        n_SI = jax.random.binomial(k, x[0], p_SI)
        seed, k = jax.random.split(seed)
        n_IR = jax.random.binomial(k, x[1], p_IR)
        return {'x': jnp.array([x[0] - n_SI, x[1] + n_SI - n_IR, x[2] + n_IR], dtype=x.dtype),
                'e': jnp.array([n_SI, n_IR], dtype=x.dtype) }

    def transition_logp_and_refresh(_t, p_old, p_new, _s):

        x_old = p_old['x']
        x_new = p_new['x']

        r_SI = SI_rate(x_old)
        r_IR = IR_rate(x_old)

        p_SI = -jnp.expm1(-r_SI*timestep)
        p_IR = -jnp.expm1(-r_IR*timestep)
        S = x_old[0]
        I = x_old[1]
        R = x_old[2]
        n_SI = x_old[0] - x_new[0]
        n_IR = x_new[2] - x_old[2]

        expected_I_new = I + n_SI - n_IR

        invalid = (
            (n_SI < 0)
            | (n_IR < 0)
            | (n_SI > S)
            | (n_IR > I)
            | (x_new[1] != expected_I_new)
        )
        
        logp = lchoose(S, n_SI) + xlogy(n_SI, p_SI) + xlogy((S - n_SI), (1-p_SI))
        logp += lchoose(I, n_IR) + xlogy(n_IR, p_IR) + xlogy((I - n_IR), (1-p_IR)) 
        return jax.lax.select(invalid, -jnp.inf, logp), {'x':x_new, 'e':jnp.array([n_SI, n_IR], dtype=x_old.dtype)}

    return transition_sample, transition_logp_and_refresh

def plot_samples_and_marginals(samples, filtered_states, smoothed_states, states, N_pop):
    plt.figure()

    T = samples.shape[1]

    for trajectory in samples:
        for x, c in zip(trajectory.T, ['r', 'g', 'b']):
            plt.plot(x, color=c, alpha=0.1)

    plt.savefig('samples.png')

    # marginals for S I R

    for i in range(T+1):
        marginals = jax.vmap(lambda x: jnp.bincount(x, length=N_pop+1), in_axes=-1)(samples[:,i,:].astype(jnp.int32))
        marginals = marginals/jnp.sum(marginals, axis=-1, keepdims=True)

        #smoothed_marginals = jax.vmap(lambda x: jnp.bincount(x, length=N_pop+1, weights=jnp.exp(smoothed_states[i])), in_axes=-1)(states)
        #filtered_marginals = jax.vmap(lambda x: jnp.bincount(x, length=N_pop+1, weights=jnp.exp(filtered_states[i])), in_axes=-1)(states)
        smoothed_marginals = jax.vmap(lambda x: jnp.bincount(x, length=N_pop+1, weights=(smoothed_states[i])), in_axes=-1)(states)
        filtered_marginals = jax.vmap(lambda x: jnp.bincount(x, length=N_pop+1, weights=(filtered_states[i])), in_axes=-1)(states)
        filtered_marginals = filtered_marginals/jnp.sum(filtered_marginals, axis=-1, keepdims=True)


        # Calculate smoothed marginals

        
        plt.figure()
        plt.plot(jnp.arange(N_pop+1), marginals[0,:], 'r')
        plt.plot(jnp.arange(N_pop+1), smoothed_marginals[0,:], 'r:')
        plt.plot(jnp.arange(N_pop+1), filtered_marginals[0,:], 'r-.')             
 
        plt.plot(jnp.arange(N_pop+1), marginals[1,:], 'g')
        plt.plot(jnp.arange(N_pop+1), smoothed_marginals[1,:], 'g:')        
        plt.plot(jnp.arange(N_pop+1), filtered_marginals[1,:], 'g-.')             

        plt.plot(jnp.arange(N_pop+1), marginals[2,:], 'b')
        plt.plot(jnp.arange(N_pop+1), smoothed_marginals[2,:], 'b:')              
        plt.plot(jnp.arange(N_pop+1), filtered_marginals[2,:], 'b-.')             


        plt.savefig(f"marginals_{i}.png")
        plt.close()



def eval_sir(seed_idx):

    seed = jax.random.key(seed_idx)

    N_pop = 100

    T = 20

    initial_probs = jnp.array([1-2/N_pop, 2/N_pop, 0.0])

    def initial_sample_events(seed):
        return {'x': jax.random.multinomial(seed, N_pop, initial_probs).astype(jnp.int64),
                'e': jnp.array([0,0], dtype=jnp.int64) }
    
    def initial_logp_events(p):
        x = p['x']
        return multinomial_logprob(x, N_pop, initial_probs)


    def initial_sample(seed):
        return jax.random.multinomial(seed, N_pop, initial_probs).astype(jnp.int64)
    
    def initial_logp(x):
        return multinomial_logprob(x, N_pop, initial_probs)
    
    transition_sample, transition_logp = make_SIR_model(3.0, 0.3, 0.2)
    transition_sample_events, transition_logp_events = make_SIR_model_events(3.0, 0.3, 0.2)


    states = jnp.array(partitions(N_pop, 3))

    initial_pos = {'x': jnp.array([N_pop-2, 2, 0], dtype=jnp.int64), 'e':jnp.array([0,0], dtype=jnp.int64)}

    
    def step(carry, x):
        pos = carry
        t, seed = x
        new_pos = transition_sample_events(t, pos, seed)
        return new_pos, new_pos


    seed, sample_seed = jax.random.split(seed)
    scan_seeds = jax.random.split(sample_seed, T)
    _, all_pos = jax.lax.scan(step, initial_pos, (jnp.arange(T), scan_seeds))

    starting_reference_path = jax.tree.map(lambda x, y: jnp.concatenate((x[None], y)), initial_pos, all_pos)

    print(starting_reference_path)

    # Observation model

    seed, obs_seed = jax.random.split(seed)

    y = jax.random.poisson(obs_seed, 0.3*starting_reference_path['x'][:,1])

    def observation_logp(t, x):
        return poisson_logprob(y[t], 0.3*x[1])
    
    def observation_logp_events(t, p):
        x = p['x']
        return poisson_logprob(y[t], 0.3*x[1])

    seed, pg_seed = jax.random.split(seed)


    N_particles = 10
    burn_in = 1_000
    N_pg_steps = 1_000

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
            initial_sample_events,          # seed -> x_0 ~ p_0(x_0)
            transition_sample_events,       # t, x_{t}, seed -> x_{t+1} ~ p(x_{t+1} | x_t)
            transition_logp_events,         # t, x_t, x_{t+1} -> log p(x_{t+1}) | x_t) - used only with use_pgas
            observation_logp_events,            # t, x_{t} -> log g(y_{t} | x_{t})          
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
    all_paths = all_paths['x'][burn_in:]

    #exact_filtering_dist, exact_smoothing_dist, _ = exact_filter_and_smooth(initial_logp, transition_logp, observation_logp, states, T)

    """
    def idx_to_state(idx):
        B = ((N_pop + 1)*(N_pop+2))//2
        u = B - 1 - idx
        n = (jnp.sqrt(8*u+1)-1)//2
        s = N_pop - n
        L = B - ((N_pop+1-s)*(N_pop+2-s))//2
        i = idx - L
        r = N_pop - i - s
        return jnp.array((s, i, r), dtype=jnp.int64)
    
    def state_to_idx(sir):
        s = sir[...,0]
        i = sir[...,1]
        return i + ((N_pop+1)*(N_pop+2) -(N_pop+1-s)*(N_pop+2-s))//2
    

    def state_to_svec(s):
        return jax.nn.one_hot(state_to_idx(s), states.shape[0])


    empirical_smoothing_dist = jnp.mean(
        state_to_svec(all_paths),
        axis=0,
    ) 

    """

    #np.testing.assert_allclose(empirical_smoothing_dist, jnp.exp(exact_smoothing_dist), atol=0.02, rtol=0.1)

    # Plot the smoothing / filtering marginals vs pg distribution

    #plot_samples_and_marginals(all_paths, jnp.exp(exact_filtering_dist), jnp.exp(exact_smoothing_dist), states, N_pop)

    fig, ax = plt.subplots(2, 1, figsize=(10,16))
    
    ax[0].plot(all_paths[:,0,1])
    ax[1].plot(all_paths[:,-1,1])

    plt.savefig("mixing.png")


def main():
    #eval_mixing(1234, 4, 5, sparse=False)
    eval_sir(123)


if __name__=="__main__":
    main()



