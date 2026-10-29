
# Exact filtering and backwards smoothing for finite state-space models

Case where we have a struture where the event e is sampled (and potential observed)
before a deterministic update of the state x.

## Notation

T - number of transitions (T+1 observation times including the initial step)
M - number of states 

x_{0:T} - hidden states
p(x_{0}) - prior for initial state
e_{1:T} - hidden events

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

## Forward filtering

P(x_0 = i, y_0) = p(x^i) g(y_0 | x^i)

for t = 1..T
    for j = 1..M
        for i = 1..M
	   Calculate the set events E_{i,j} =  { e | F(x^i, e) = x^j }
	   Calculate R_{i,j}^{t} = sum_{e \in E_{i,j}} r(e | x^i) h(z_t | e)
    
    P(x_t = j, y_{0:t-1}, z_{1:t}) = sum_{i} R_{i,j}^{t} P(x_{t-1} = i, y_{0:t-1}, z_{1:t-1})
    P(x_t = j, y_{0:t}) = g(y_t | x^j) P(x_t = j, y_{0:t-1}, z_{1:t})

This generates P(x_t=j, y_{0:t}, z_{1:t}) for t=0..T

## Backwards smoothing with event observations

Have P(x_T=j, y_{0:T}, z_{1:T}) from forward filter.
Normalize to give P(x_T=j | y_{0:T}, z_{1:T})

For t = T-1..0

    Have P(x_t=j | y_{0:t}, z_{1:t}) and P(x_{t+1}=i | y_{0:T}, z_{1:T})

    Can calculate (or cache from forward pass) P(x_{t+1}=j, z_{t+1} | y_{0:t}, z_{1:t}) = \sum_i R_{i,j}^{t+1} P(x_t=i | y_{0:t}, z_{1:t})
    where R_{i,j}^{t+1} is as defined in the forward pass

    # Note that the problem is equivalent to the version without events but with modified transition probabilities?!
    # So the formulas should be the same!

    P(x_t=i | y_{0:T}, z_{1:T}) = sum_j P(x_{t+1}=j | y_{0:T}, z_{1:T}) P(x_{t}=i | x_{t+1}=j, y_{0:T}, z_{1:T})

    P(x_{t}=i | x_{t+1}=j, y_{0:T}, z_{1:T}) = P(x_{t}=i | x_{t+1}=j, y_{0:t}, z_{1:t+1})
                                             = P(x_{t}=i | y_{0:t}, z_{1:t}) P(x_{t+1}=j, z_{t+1} | x_{t}=i, z_{1:t}) / P(x_{t+1}=j, z_{t+1} | y_{0:t}, z_{1:t})





    
    
