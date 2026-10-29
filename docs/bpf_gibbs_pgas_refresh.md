
# Particle Gibbs with BPF, PGAS and reference particle refresh for degenerate state-space models


Algorithm for a single trajectory update.

## Notation
T - number of transitions (T+1 observation times including the initial step)
N - number of particles - particle n=N is always the reference particle
x_{0:T} - hidden states
e_{1:T} - hidden events
x'_{0:T} - previous/reference trajectory states
e'_{1:T} - previous/reference trajectory events
a_{1:T}^{1:N} - index of ancestor of particle at previous time
y_{0:T} - observations of states x_{0:T}
z_{1:T} - observations of events e_{1:T}
p(x_{0}) - prior for initial state
r(e_{t}|x_{t-1}) - model for event
x_{t} = F(x_{t-1}, e_{t}) - deterministic update rule for x_{t}
g(y_{t}|x_{t}) - observation model for states
h(z_{t}|e_{t}) - observation model for events

## State space model

x_0 ~ p(x_0)
e_t ~ r(e_t|x_{t-1})
x_{t} = F(x_{t-1}, e_t)
y_t ~ g(y_t|x_t)
z_t ~ h(z_t|e_t)


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
        sample e_t^i ~ r(e_t^i | x_{t-1}^{a_t^i})
	update x_t^i = F(x_{t-1}^{a_t^i}, e_t^i)

    # Impose reference trajectory
    set x_t^N = x'_t e_t^N = e'_t

    # PGAS - resample ancestor of reference particle
    # Simplest version where we try to relink every timestep
    for i=1..N
       calculate E^i = { e | F(x_{t-1}^i, e) = x_t'}
       calculate Z^i = sum_{e in E^i} w_{t-1}^i r(e | x_{t-1}^i) h (z_t | e)
       [ set Z^i=0 if no events in E^i ]
    normalize Z
    sample j ~ Categorical(Z)
    e ~ E^j with weights \propto r(e | x_{t-1}^{j}) h (z_t | e)
    update e_t^N = e
    

    # Update weights

    for i=1..N
        set w_t^i = g(y_t | x_t^i) h(z_t | e_t^i)

    normalize w_t (if Categorical requires normalized weights)

# Sampling

Sample an ancestral path
I_T ~ Categorical(w_T)
for t=T..1
    I_{t-1} = a_t^{I_{t}}

Updated sample given by x^{new}_t = x_t^{I_t} t=0..T, e^{new}_t=e_t^{I_t} t=1..T