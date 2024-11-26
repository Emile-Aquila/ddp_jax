from typing import Dict, List, Callable, Tuple, Any
import jax
import jax.numpy as jnp
import numpy as np
from ddp_jax.dynamics.dynamics_jax import ParallelTwoWheelVehicleModel, DynamicsBase
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ddp_jax.controllers.al_ddp import ALDDPController
from ddp_jax.objects.field import Field, GenTestField
import functools
import time
from ddp_jax.utils.defines import ReferenceTrajectory


def plot_ax(state: np.ndarray, ref_xs: np.ndarray, start: np.ndarray, goal: np.ndarray, ax):
    ax.plot(ref_xs[:, 0], ref_xs[:, 1], "r")
    ax.plot(state[0], state[1], "go")
    ax.plot(goal[0], goal[1], "ro")

    ax.add_patch(plt.Circle((state[0], state[1]), 0.2, fill=False))
    ax.quiver(state[0], state[1], np.cos(state[2]), np.sin(state[2]))
    ax.plot(start[0], start[1], "go")

    ax.set_aspect("equal")


@jax.jit
def input_constraint(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    return (u.T @ u - 2.0) / 6.0

@jax.jit
def state_constraint(x: jnp.ndarray, u:jnp.ndarray, other_x: jnp.ndarray, r: float) -> jnp.ndarray:
    return 2 * r - jnp.sqrt((x - other_x).T @ jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])) @ (x - other_x))



def main():
    # Problem Settings
    field: Field = GenTestField(0)
    robot_model: DynamicsBase = ParallelTwoWheelVehicleModel(dt=0.1)
    dynamics: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = robot_model.get_dynamics()
    horizon, nx, nu = 20, 5, 2

    ## Constraints
    n_constraints = 3
    f_constraints = [
        input_constraint,
        functools.partial(state_constraint, other_x=jnp.array([5.0, 5.0, 0.0, 0.0, 0.0]), r=1.0),
        functools.partial(state_constraint, other_x=jnp.array([4.0, 4.0, 0.0, 0.0, 0.0]), r=1.0)
    ]

    # Controller Settings
    Q = jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.0, 0.0]))
    Q_terminal = jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.0, 0.0])) * 8.0
    R = jnp.eye(2) * 0.5

    controller = ALDDPController(dynamics, f_constraints, Q, Q_terminal, R, horizon=horizon)
    ref_traj: ReferenceTrajectory = ReferenceTrajectory(
        ref_xs=jnp.zeros((horizon+1, nx)),  # (horizon+1, n_x)
        ref_us=jnp.zeros((horizon, nu)),  # (horizon, n_u)
        lambdas=jnp.zeros((horizon, n_constraints))  # (horizon, n_constraints)
    )


    # Setting Start and Goal
    start: jnp.ndarray = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])
    goal: jnp.ndarray = jnp.array([10.0, 10.0, 0.0, 0.0, 0.0])
    state: jnp.ndarray = start


    # jit compile
    print("start jit compile")
    _, _ = controller.calc_input(state, goal, ref_traj)
    print("end jit compile")


    # Rollout
    states = []
    predicted_traj_lists = []

    ## start time measurement
    start_time = time.perf_counter()

    for _ in range(200):
        input_u, ref_traj = controller.calc_input(state, goal, ref_traj)
        state = dynamics(state, input_u)
        jax.debug.print("{}", state)
        states.append(np.array(state))
        predicted_traj_lists.append(np.array(ref_traj.ref_xs))

    states = np.array(states)
    end_time = time.perf_counter()
    print(f"elapsed time: {end_time - start_time} [s], ave: {(end_time - start_time) / 200} [s]")


    # Animation
    fig, ax = plt.subplots()
    def update(vals, field):
        state_, state_traj_ = vals
        ax.cla()
        field.frame.plot(ax, non_fill=True)
        patches = [
            plt.Circle((5.0, 5.0), 1.0, fill=True),
            plt.Circle((4.0, 4.0), 1.0, fill=True),
        ]
        for patch in patches:
            ax.add_patch(patch)
        plot_ax(np.array(state_), np.array(state_traj_), np.array(start), np.array(goal), ax)


    anim = FuncAnimation(fig, func=functools.partial(update, field=field),
                         frames=zip(states, predicted_traj_lists), interval=100, cache_frame_data=False)
    anim.save("output.gif", writer="pillow")
    plt.close()


if __name__ == "__main__":
    main()