from typing import Dict, List, Callable, Tuple, Any
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod


class DynamicsBase(ABC):
    def __init__(self, dt: float, nx: int, nu: int):
        self.nx: int = nx
        self.nu: int = nu
        self.dt: float = dt
        pass

    @abstractmethod
    def get_dynamics(self) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        raise NotImplementedError()


class ParallelTwoWheelVehicleModel(DynamicsBase):
    def __init__(self, dt: float):
        super(ParallelTwoWheelVehicleModel, self).__init__(dt, nx=5, nu=2)

    def get_dynamics(self) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        def kinematic_jax(state_: jnp.ndarray, input_: jnp.ndarray) -> jnp.ndarray:
            vel = (input_ + state_[3:]) * 0.5
            new_state = state_[:3] + jnp.array([vel[0] * self.dt * jnp.cos(state_[2]),
                                                vel[0] * self.dt * jnp.sin(state_[2]),
                                                vel[1] * self.dt])
            new_state = jnp.concatenate([new_state, input_])
            return new_state
        return jax.jit(kinematic_jax)
