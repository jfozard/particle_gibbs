

# Design considerations for a Particle Gibbs in gemlib

## Introduction

We wish to incorporate Particle Gibbs (PG) into the gemlib Metropolis-within-Gibbs (MwG) framework as
an alternative method to sample from the latent infection state, rather than move / add / delete an event, or iFFBS.

## Algorithm overview

## Notes on the validity of this method

Particle Gibbs uses a conditional SMC update to sample from the latent parameters. The standard proof
of its validity (Andrieu 2010) considers it as a two-step Gibbs sample from a target distribution on a
space (U, K), where U is the complete state of the cSMC filter sweep and K is the index of the selected
particle at the final time.

## Testing the particle gibbs method

### Small discrete hidden markov model

Consider a m=2 state model with T=4 steps. This has 2 states at each timesteps and 2^4 total possible paths.
Observe the state at each step. Enumerate all the latent paths x_{1:4} and calculate p(x).

### Slightly larger discrete markov model



