import jax
import jax.numpy as jnp


@jax.jit
def add(x, y):
    return x + y


def main():
    x = jnp.array([1, 2, 3])
    y = jnp.array([4, 5, 6])
    z = add(x, y)
    print(z)


if __name__ == '__main__':
    main()
