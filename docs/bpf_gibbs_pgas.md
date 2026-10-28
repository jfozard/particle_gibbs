
# Particle Gibbs with BPF and PGAS

Algorithm for a single trajectory update.



## Notation
T - number of transitions (T+1 observation times including the initial step)
N - number of particles - particle n=N is always the reference particle
x_{0:T} - hidden states
x'_{0:T} - previous/reference trajectory
a_{1:T}^{1:N} - index of ancestor of particle at previous time
y_{0:T} - observations
p(x_{0}) - prior for initial state
f(x_{t}|x_{t-1}) - model for transitions
g(y_{t}|x_{t}) - observation model

## State space model

x_0 ~ p(x_0)
x_t ~ f(x_t|x_{t-1})
y_t ~ g(y_t|x_{t})


Sample initial particles
x_0^i ~ p(x_0) for i=1..N-1
x_0^N = x_0'

Calculate initial weights
w_0^i = g(y_0|x_0^i) for i=1..N

normalize \vct{w}_0

for t = 1..T

    # Resample
    for i=1..N-1
        sample a_t^i ~ Categorical(\vct{w}_{t-1})


    # Propagate
    for i=1..N-1
        sample x_t^i ~ f(x_t | x_{t-1}^{a_t^i})

    # Impose reference trajectory
    set x_t^N = x'_t

    # PGAS - resample ancestor of reference particle
    # Simplest version where we try to relink every timestep
    for i=1..N
       set nu_i = w_{t-1}^{i} f(x'_t | x_{t-1}^i)

    normalize nu
    sample a_t^N ~ Categorical(\vct{nu})

    # Update weights

    for i=1..N
        set w_t^i = g(y_t | x_t^i)

    normalize w_t (if Categorical requires normalized weights)

# Sampling

Sample an ancestral path
I_T ~ Categorical(w_T)
for t=T..1
    I_{t-1} = a_t^{I_{t}}

Updated sample given by x^{new}_{t} = x_{t}^{I_t}

 
 
 



