from typing import Dict, List, Callable, Tuple, Any
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from ddp_jax.controllers.ddp import DDPController
import time
from ddp_jax.utils.defines import ReferenceTrajectory


@jax.jit
def system(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    # 2次の線形システム
    A: jnp.ndarray = jnp.array([[1.0, 0.2], [0.0, 1.0]])
    B: jnp.ndarray = jnp.array([[0.0], [0.1]])
    return A @ x + B @ u


@jax.jit
def lqr_gains(A: jnp.ndarray, B: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray, Q_terminal: jnp.ndarray, horizon: int) -> jnp.ndarray:
    # LQR
    def for_func(i: int, val: jnp.ndarray) -> jnp.ndarray:
        P_pre: jnp.ndarray = val
        P = Q + A.T @ P_pre @ A - A.T @ P_pre @ B @ jnp.linalg.inv(R + B.T @ P_pre @ B) @ B.T @ P_pre @ A
        return P

    P0 = jax.lax.fori_loop(0, horizon, for_func, Q_terminal)
    K = jnp.linalg.inv(R + B.T @ P0 @ B) @ B.T @ P0 @ A
    return K

@jax.jit
def lqr_controller(x: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    # LQR
    A: jnp.ndarray = jnp.array([[1.0, 0.2], [0.0, 1.0]])
    B: jnp.ndarray = jnp.array([[0.0], [0.1]])
    K = lqr_gains(A, B, Q, R, Q, 10)
    return -K @ x



def main():
    # Problem Settings
    dynamics: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = system
    horizon, nx, nu = 10, 2, 1

    # Controller Settings
    Q = jnp.eye(2)
    R = jnp.eye(1) * 0.5

    controller = DDPController(dynamics, Q, Q, R, horizon=horizon)
    ref_traj: ReferenceTrajectory = ReferenceTrajectory(
        ref_xs=jnp.zeros((horizon+1, nx)),  # (horizon+1, n_x)
        ref_us=jnp.zeros((horizon, nu)),  # (horizon, n_u)
        lambdas=jnp.zeros((horizon, 1))  # (horizon, n_constraints)
    )

    # Setting Start and Goal
    start: jnp.ndarray = jnp.array([1.0, 1.0])
    goal: jnp.ndarray = jnp.array([0.0, 0.0])

    state: jnp.ndarray = start

    # jit compile
    print("start jit compile")
    _, _ = controller.calc_input(state, goal, ref_traj)
    print("end jit compile")


    # Rollout
    states = [state.copy()]
    states_lqr = [state.copy()]
    ## start time measurement
    start_time = time.perf_counter()

    for _ in range(200):
        input_u, ref_traj = controller.calc_input(state, goal, ref_traj)
        state = dynamics(state, input_u)
        jax.debug.print("{}", state)
        states.append(np.array(state))
    states = np.array(states)

    end_time = time.perf_counter()
    print(f"elapsed time: {end_time - start_time} [s], ave: {(end_time - start_time) / 200} [s]")

    state = start
    for _ in range(200):
        input_u = lqr_controller(state, Q, R)
        state = dynamics(state, input_u)
        states_lqr.append(np.array(state))
    states_lqr = np.array(states_lqr)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(states[:, 0], states[:, 1], label="trajectory (DDP)")
    ax.plot(states_lqr[:, 0], states_lqr[:, 1], label="trajectory (LQR)")
    ax.plot(start[0], start[1], marker="o", label="start")
    ax.plot(goal[0], goal[1], marker="x", label="goal")
    ax.legend()
    plt.savefig("./figs/example_ddp.png")
    plt.show()

    # Plot (x, yをそれぞれplot. subplotを使う)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.ones(200) * goal[0], label="goal", linestyle="--")
    ax[0].plot(states[:, 0], label="x (DDP)")
    ax[0].plot(states_lqr[:, 0], label="x (LQR)")
    ax[0].legend()

    ax[1].plot(np.ones(200) * goal[1], label="goal", linestyle="--")
    ax[1].plot(states[:, 1], label="y (DDP)")
    ax[1].plot(states_lqr[:, 1], label="y (LQR)")
    ax[1].legend()

    plt.savefig("./figs/example_ddp_xy.png")
    plt.show()


if __name__ == "__main__":
    main()