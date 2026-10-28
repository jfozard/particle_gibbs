
# Exact filtering and backwards smoothing for finite state-space models

## Notation
T - number of transitions (T+1 observation times including the initial step)
M - number of states 

x_{0:T} - hidden states
y_{0:T} - observations of states x_{0:T}
p(x_{0}) - prior for initial state
f(x_{t}|x_{t-1}) - model for transitions
g(y_{t}|x_{t}) - observation model for states

## State space model

x_0 ~ p(x_0)
x_t ~ r(x_t|x_{t-1})
y_t ~ g(y_t|x_t)

## Forward filtering

P(x_0 = i, y_0) = p(x^i) g(y_0 | x^i)

for t = 1..T
    P(x_t = j, y_{0:t-1}) = sum_{i} f(x^j | x^i) P(x_{t-1} = i, y_{0:t-1})
    P(x_t = j, y_{0:t}) = g(y_t | x^j) P(x_t = j, y_{0:t-1})

This generates P(x_t=j, y_{0:t}) for t=0..T

## Backwards smoothing

Have P(x_T=j, y_{0:T}) from forward filter.
Normalize to give P(x_T=j | y_{0:T})

For t = T-1..0

    Have P(x_t=j | y_{0:t}) and P(x_{t+1}=i | y_{0:T})

    Can calculate (or cache from forward pass) P(x_{t+1}=i | y_{0:t}) = \sum_j f(x^i | x^j) P(x_t=j | y_{0:t})

    P(x_t=i | y_{0:T}) = sum_j P(x_{t+1}=j | y_{0:T}) P(x_{t}=i | x_{t+1}=j, y_{0:T})

    P(x_{t}=i | x_{t+1}=j, y_{0:T}) = P(x_{t}=i | x_{t+1}=j, y_{0:t})
                                    = P(x_{t}=i | y_{0:t}) P(x_{t+1}=j | x_{t}=i) / P(x_{t+1}=j | y_{0:t})


    
    
