
# Particle Gibbs with Auxillary weights, proposal and PGAS

Algorithm for a single trajectory update.

Note that auxillary weights are in the original PGAS paper, so this is definitely fine


## Notation
T - number of transitions (T+1 observation times including the initial step)
N - number of particles - particle n=N is always the reference particle
x_{0:T} - hidden states
x'_{0:T} - previous/reference trajectory
a_{1:T}^{1:N} - index of ancestor of particle at previous time
y_{0:T} - observations
p_0(x_{0}) - prior for initial state
R_0(x_{0}) - proposal for initial state
f_t(x_{t}|x_{t-1}) - model for transitions
R_t(x_{t}|x_{t-1}, y_{0:T}) - proposal kernel
nu_t(x_{t-1} | y_{0:T}) - adjustment multiplier weights
g_t(y_{t}|x_{t}) - observation model

## State space model

x_0 ~ p_0(x_0)
x_t ~ f_t(x_t|x_{t-1})
y_t ~ g_t(y_t|x_{t})


Sample initial particles
x_0^i ~ R_0(x_0) for i=1..N-1
x_0^N = x_0'

Calculate initial weights
w_0^i = g_0(y_0|x_0^i) p_0(x_0^i) / R_0(x_0^i) for i=1..N

normalize \vct{w}_0

for t = 1..T

    # Calculate auxillary weights

    m_{t}^i = w_{t-1}^{i} nu_t(x_{t-1}^{i}, y_{0:T})

    normalize m


    # Resample
    for i=1..N-1
        sample a_t^i ~ Categorical(\vct{m})

    # Propagate
    for i=1..N-1
        sample x_t^i ~ R_t(x_t | x_{t-1}^{a_t^i}, y_{0:T})

    # Impose reference trajectory
    set x_t^N = x'_t

    # PGAS - resample ancestor of reference particle
    # Simplest version where we try to relink every timestep
    for i=1..N
       set l_i = w_{t-1}^{i} f_t(x'_t | x_{t-1}^i)

    normalize l
    sample a_t^N ~ Categorical(\vct{l})

    # Update weights

    for i=1..N
        set w_t^i = g_t(y_t | x_t^i) f_t(x_t^i | x_{t-1}^{a_t^i}) / nu_t(x_{t-1}^{a_t^i}, y_{0:T}) / R_t(x_t^i | x_{t-1}^{a_t^i}, y_{0:T})

    normalize w_t (if Categorical requires normalized weights)

# Sampling

Sample an ancestral path
I_T ~ Categorical(w_T)
for t=T..1
    I_{t-1} = a_t^{I_{t}}

Updated sample given by x^{new}_{t} = x_{t}^{I_t}

 
 
 



