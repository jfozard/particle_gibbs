
import jax
from jax import Array
import jax.numpy as jnp
from jax.scipy.special import xlogy, gammaln


# Multinomial coefficients
def lmchoose(c):
    N = jnp.sum(c)
    s = jax.lax.lgamma(N+1.0) - jnp.sum(jax.vmap(lambda v: jax.lax.lgamma(v+1.0))(c))
    return s

# Binomial coefficients
def lchoose(N,c):
    s = jax.lax.lgamma(N+1.0) - jax.lax.lgamma(N-c+1.0) - jax.lax.lgamma(c+1.0)
    return s

def partitions(s, N):
    res = []
    if s==0:
        return [(0,)*N]
    if N==1:
        return [(s,)]
    for i in range(s+1):
        pp = partitions(s-i, N-1)
        res += [(i,) + r for r in pp]
    return res

def list_product(p):
    if len(p)==1:
        return [(r,) for r in p[0]]
    lp = list_product(p[1:])
    return [(q,) + r for q in p[0] for r in lp]


# verify integer inputs?
def binomial_logprob(k, n, p):
    """log P(K=k), K ~ Binomial(n, p)."""
    logp = (
        gammaln(n + 1)
        - gammaln(k + 1)
        - gammaln(n - k + 1)
        + xlogy(k, p)
        + xlogy(n - k, 1.0 - p)
    )

    valid = (k >= 0) & (k <= n) & (p >= 0) & (p <= 1)
    return jnp.where(valid, logp, -jnp.inf)


def poisson_logprob(k, rate):
    """log P(K=k), K ~ Poisson(rate)."""
    logp = xlogy(k, rate) - rate - gammaln(k + 1)

    valid = (k >= 0) & (rate >= 0)
    return jnp.where(valid, logp, -jnp.inf)

def multinomial_logprob(k, n, p):
    """
    k: (..., C) counts
    p: (..., C) probabilities
    """
    total = jnp.sum(k, axis=-1)

    logp = (
        gammaln(n + 1)
        - jnp.sum(gammaln(k + 1), axis=-1)
        + jnp.sum(xlogy(k, p), axis=-1)
    )

    valid = (
        jnp.all(k >= 0, axis=-1)
        & jnp.all(p >= 0, axis=-1)
        & jnp.isclose(jnp.sum(p, axis=-1), 1.0)
        & (total == n)
    )

    return jnp.where(valid, logp, -jnp.inf)

def stable_normalize(x):
    return x - jax.scipy.special.logsumexp(x)


