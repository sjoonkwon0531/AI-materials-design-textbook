import jax
import jax.numpy as jnp

def f(x):
    return (x * x).sum() + jnp.sin(x[0])

# grad는 함수를 함수로 변환한다
grad_f = jax.grad(f)

x = jnp.array([1.0, 2.0, 3.0])
print(grad_f(x))
# 출력: [2.5403023 4.        6.       ]
