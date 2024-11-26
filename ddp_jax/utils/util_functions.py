import jax
import jax.numpy as jnp



@jax.jit
def regularize_matrix(hermite_matrix: jnp.ndarray, min_lambda: float = 1e-3) -> jnp.ndarray:
    min_value = jnp.min(jnp.linalg.eigh(hermite_matrix)[0])
    hermite_matrix = jax.lax.cond(
        jnp.where(min_value < min_lambda, True, False).any(),
        lambda x: x + jnp.eye(x.shape[0]) * (min_lambda - min_value.min()),
        lambda x: x,
        operand=hermite_matrix
    )
    return hermite_matrix
