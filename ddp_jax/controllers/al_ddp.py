from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, hessian
from functools import partial
from jax.tree_util import register_pytree_node_class
from ..utils.defines import ReferenceTrajectory, ControllerBase
from ..utils.util_functions import regularize_matrix


@register_pytree_node_class
class ALDDPController(ControllerBase):
    def __init__(self, f_dynamics: Callable, f_constraints: list[Callable],
                 Q: jnp.ndarray, Q_terminal: jnp.ndarray, R: jnp.ndarray, max_iter: int = 30,
                 horizon: int = 20, tol: float = 1e-3, gamma: float = 0.25):
        super().__init__(horizon)
        self.f_dynamics: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = f_dynamics

        # 制約関数
        self.f_constraints: Callable = jax.jit(lambda x_, u_: jnp.array([f(x_, u_) for f in f_constraints]))
        self.f_consts: list[Callable] = f_constraints

        # ハイパーパラメータ
        self.max_iter: int = max_iter
        self.tol: float = tol  # 収束判定の閾値
        self.gamma: float = gamma
        self.mu_init: float = 5.0  # ペナルティの初期値

        # コスト関数のパラメータ
        self.Q: jnp.ndarray = Q
        self.Q_terminal: jnp.ndarray = Q_terminal
        self.R: jnp.ndarray = R

    @partial(jax.jit, static_argnames=["self"])
    def cost(self, x: jnp.ndarray, u: jnp.ndarray, target_x: jnp.ndarray) -> jnp.ndarray:
        return (x-target_x).T @ self.Q @ (x-target_x) + u.T @ self.R @ u

    @partial(jax.jit, static_argnames=["self"])
    def terminal_cost(self, x: jnp.ndarray, target_x: jnp.ndarray) -> jnp.ndarray:
        return (x-target_x).T @ self.Q_terminal @ (x-target_x)

    @staticmethod
    @jax.jit
    def _penalty_function(constraint_val: jnp.ndarray, lambda_: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        tmp = (mu * constraint_val) / lambda_
        return (lambda_ ** 2 / mu) * jnp.where(tmp >= -0.5, 0.5 * tmp ** 2 + tmp, -0.25 * jnp.log(-2 * tmp) - 3 / 8)


    @partial(jax.jit, static_argnames="self")
    def _augmented_lagrangian(self, x: jnp.ndarray, u: jnp.ndarray, target_x: jnp.ndarray, lambdas: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        constraint_vals = self.f_constraints(x, u)
        penalty = jnp.sum(jax.vmap(self._penalty_function, in_axes=(0, 0, None))(constraint_vals, lambdas, mu))
        return self.cost(x, u, target_x) + penalty


    @partial(jax.jit, static_argnames="self")
    def _second_order_dynamics_approximation(self, x: jnp.ndarray, u: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:  # ダイナミクスの二次近似
        f_x = jacfwd(self.f_dynamics, argnums=0)(x, u)  # df/dx
        f_u = jacfwd(self.f_dynamics, argnums=1)(x, u)  # df/du
        f_xx = hessian(self.f_dynamics, argnums=0)(x, u)  # d^2f/dx^2
        f_uu = hessian(self.f_dynamics, argnums=1)(x, u)  # d^2f/du^2
        f_ux = jacfwd(jacfwd(self.f_dynamics, argnums=1), argnums=0)(x, u)  # d^2f/dudx
        return f_x, f_u, f_ux, f_xx, f_uu

    @partial(jax.jit, static_argnames="self")
    def _second_order_cost_approximation(self, x: jnp.ndarray, u: jnp.ndarray, target_x: jnp.ndarray, lambdas: jnp.ndarray, mu: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        l_x = grad(self._augmented_lagrangian, argnums=0)(x, u, target_x, lambdas, mu)
        l_u = grad(self._augmented_lagrangian, argnums=1)(x, u, target_x, lambdas, mu)
        l_xx = hessian(self._augmented_lagrangian, argnums=0)(x, u, target_x, lambdas, mu)
        l_uu = hessian(self._augmented_lagrangian, argnums=1)(x, u, target_x, lambdas, mu)
        l_ux = jacfwd(jacfwd(self._augmented_lagrangian, argnums=1), argnums=0)(x, u, target_x, lambdas, mu)
        return l_x, l_u, l_ux, l_xx, l_uu

    @partial(jax.jit, static_argnames="self")
    def backward(self, ref_traj: ReferenceTrajectory, target_x: jnp.ndarray, mu: jnp.ndarray)\
            -> Tuple[jnp.ndarray, jnp.ndarray]:
        # 最終時刻でのコストと勾配の計算
        ref_xs, ref_us, lambdas = ref_traj.ref_xs, ref_traj.ref_us, ref_traj.lambdas
        V = self.terminal_cost(ref_xs[-1], target_x)
        V_x = grad(self.terminal_cost, argnums=0)(ref_xs[-1], target_x)
        V_xx = hessian(self.terminal_cost, argnums=0)(ref_xs[-1], target_x)

        # フィードバックゲインの計算
        def scan_func(val, input_val):
            ref_x, ref_u, lambda_ = input_val
            v, v_x, v_xx, mu_, time_step = val

            # 二次近似
            f_x, f_u, f_ux, f_xx, f_uu = self._second_order_dynamics_approximation(ref_x, ref_u)
            l_x, l_u, l_ux, l_xx, l_uu = self._second_order_cost_approximation(ref_x, ref_u, target_x, lambda_, mu_)

            # フィードバックゲイン
            q_x = l_x + f_x.T @ v_x
            q_u = l_u + f_u.T @ v_x
            q_xx = l_xx + f_x.T @ v_xx @ f_x + jnp.tensordot(v_x, f_xx, axes=1)
            q_uu = l_uu + f_u.T @ v_xx @ f_u + jnp.tensordot(v_x, f_uu, axes=1)
            q_ux = l_ux + f_u.T @ v_xx @ f_x + jnp.tensordot(v_x, f_ux, axes=1)

            # 正則化
            q_uu = regularize_matrix(q_uu, min_lambda=1e-2)

            # ヘッセ行列の逆行列
            q_uu_inv = jnp.linalg.inv(q_uu)
            K_now = - q_uu_inv @ q_ux
            k_now = - q_uu_inv @ q_u

            # ヘッセ行列の更新
            v_next = 0.5 * q_u.T @ k_now
            vx_next = q_x + q_ux.T @ k_now
            vxx_next = q_xx + q_ux.T @ K_now
            vxx_next = regularize_matrix(vxx_next, min_lambda=1e-2)

            return (v_next, vx_next, vxx_next, mu_, time_step-1), (K_now, k_now)

        _, (Ks, ks) = jax.lax.scan(
            scan_func,
            (V, V_x, V_xx, mu, self.horizon),
            (ref_xs[:-1][::-1], ref_us[::-1], lambdas[::-1])
        )
        return Ks[::-1], ks[::-1]

    @partial(jax.jit, static_argnames="self")
    def forward(self, x0: jnp.ndarray, ref_traj: ReferenceTrajectory, target_x: jnp.ndarray, gains: Tuple[jnp.ndarray, jnp.ndarray], alpha: float = 1.0):
        def scan_func(val, input_val):
            x, J_ = val
            ref_x, ref_u, (K, k) = input_val
            u_new = ref_u + alpha * k + K @ (x - ref_x)  # alpha: step size
            J_new = J_ + self.cost(x, u_new, target_x)
            x_new = self.f_dynamics(x, u_new)
            return (x_new, J_new), (x_new, u_new)

        ref_xs, ref_us = ref_traj.ref_xs, ref_traj.ref_us
        (_, j_new), (x_traj_new, u_traj_new) = jax.lax.scan(scan_func, (x0, 0.0), (ref_xs[:-1], ref_us, gains))
        x_traj_new = jnp.concatenate([x0[None], x_traj_new], axis=0)
        return x_traj_new, u_traj_new, j_new

    @partial(jax.jit, static_argnames=["self"])
    def calc_input(self, x: jnp.ndarray, target_x: jnp.ndarray, ref_traj_pre: ReferenceTrajectory) -> Tuple[jnp.ndarray, ReferenceTrajectory]:
        ref_traj = ReferenceTrajectory(ref_xs=ref_traj_pre.ref_xs, ref_us=ref_traj_pre.ref_us, lambdas=jnp.zeros_like(ref_traj_pre.lambdas) + 1e-6)
        traj_info = self.iterative_compute(x, ref_traj, target_x, self.max_iter)
        return traj_info.ref_us[0], traj_info

    @partial(jax.jit, static_argnames="self")
    def _linear_search(self, x0: jnp.ndarray, ref_traj: ReferenceTrajectory, target_x: jnp.ndarray, gains: Tuple[jnp.ndarray, jnp.ndarray],
                       mu: jnp.ndarray, cost_old: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        lambdas = ref_traj.lambdas
        def body_func(val):
            (_, _, _), cost_old_, _, alpha_ = val
            x_traj_new_, u_traj_new_, j_new_ = self.forward(x0, ref_traj, target_x, gains, alpha_ * 0.5)

            # penaltyもコストに含める
            cost_new_ = j_new_
            constraint_vals_ = jax.vmap(self.f_constraints, in_axes=(0, 0))(x_traj_new_[:-1], u_traj_new_)
            cost_new_ += jnp.sum(jax.vmap(self._penalty_function, in_axes=(0, 0, None))(constraint_vals_, lambdas, mu))

            return (x_traj_new_, u_traj_new_, cost_new_), cost_old_, j_new_, alpha_*0.5

        def cond_func(val):
            (_, _, cost_new_), cost_old_, j_new_, alpha_ = val
            return (cost_new_ >= cost_old_) & (alpha_ > 1e-2) #& (~jnp.isnan(cost_new).any())

        # 直線探索
        x_traj_new, u_traj_new, j_new = self.forward(x0, ref_traj, target_x, gains, 1.0)

        cost_new = j_new
        constraint_vals = jax.vmap(self.f_constraints, in_axes=(0, 0))(x_traj_new[:-1], u_traj_new)
        cost_new += jnp.sum(jax.vmap(self._penalty_function, in_axes=(0, 0, None))(constraint_vals, lambdas, mu))

        (x_traj_new, u_traj_new, cost_new), _, j_new, alpha = jax.lax.while_loop(
            cond_func,
            body_func,
            ((x_traj_new, u_traj_new, cost_new), cost_old, j_new, 1.0)
        )
        return x_traj_new, u_traj_new, j_new, cost_new


    @partial(jax.jit, static_argnames="self")
    def _check_convergence1(self, j_new: jnp.ndarray, j_old: jnp.ndarray, ref_trajs: ReferenceTrajectory) -> jnp.ndarray:
        # Check cost convergence
        is_cost_converged = jnp.abs(j_new - j_old).sum() < self.tol

        # Check constraint violations
        x_traj, u_traj = ref_trajs.ref_xs, ref_trajs.ref_us
        constraint_vals = jax.vmap(self.f_constraints, in_axes=(0, 0))(x_traj[:-1], u_traj)
        is_feasible = constraint_vals.max() < 0.005

        # Check numerical stability
        is_numerically_stable = ~(jnp.isnan(j_new).any() | jnp.isnan(constraint_vals).any())
        return is_cost_converged & is_feasible & is_numerically_stable

    @partial(jax.jit, static_argnames="self")
    def iterative_compute(self, x0: jnp.ndarray, ref_traj: ReferenceTrajectory, target_x: jnp.ndarray, max_iter: int) -> ReferenceTrajectory:
        def body_func(val):
            count_, (j_old_, _, cost_old, _), ref_traj_, mu_ = val
            gains: Tuple[jnp.ndarray, jnp.ndarray] = self.backward(ref_traj_, target_x, mu_)
            lambdas_: jnp.ndarray = ref_traj_.lambdas

            # 直線探索
            cost_old = jax.lax.cond(count_ == 0, lambda _: jnp.inf, lambda xx: xx, operand=cost_old)
            x_traj_new, u_traj_new, j_new_, cost_new = self._linear_search(x0, ref_traj_, target_x, gains, mu_, cost_old)

            # update multipliers
            constraint_vals = jax.vmap(self.f_constraints, in_axes=(0, 0))(x_traj_new[:-1], u_traj_new)  # (horizon, n_constraints)
            violations = jnp.maximum(constraint_vals, 0)  # 違反量
            lambdas_ += mu_ * violations  # update lambda (horizon, n_constraints)

            # update penalty coefficient
            mu_ *= 1.5  # 勾配が急変しない様に, 緩やかに増加させる

            ref_traj_new = ReferenceTrajectory(ref_xs=x_traj_new, ref_us=u_traj_new, lambdas=lambdas_)
            return count_+1, (j_new_, j_old_, cost_new, cost_old), ref_traj_new, mu_

        def f_cond(val):
            count_, (j_new_, j_old_, _, _), ref_traj_, mu_ = val
            return (count_ < max_iter) & (self._check_convergence1(j_new_, j_old_, ref_traj_) == False)

        # ラグランジュ乗数の初期化
        _, (j_new, j_old, _, _), ref_traj_ans, _ = jax.lax.while_loop(
            f_cond, body_func, (0, (0.0, jnp.inf, 0.0, jnp.inf), ref_traj, self.mu_init)
        )

        return ref_traj_ans

    def tree_flatten(self):
        children = (self.Q, self.Q_terminal, self.R)
        aux_data = (self.f_dynamics, self.f_consts, self.horizon, self.max_iter, self.tol, self.mu_init, self.gamma)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        Q, Q_terminal, R = children
        f_dynamics, f_constraints, horizon, max_iter, tol, mu_init, gamma = aux_data

        obj = cls(f_dynamics, f_constraints, Q, Q_terminal, R, max_iter, horizon, tol, gamma)
        obj.mu_init = mu_init
        return obj

