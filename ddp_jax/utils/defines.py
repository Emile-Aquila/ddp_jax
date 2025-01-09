from typing import Tuple
import jax.numpy as jnp
from abc import ABC, abstractmethod
from flax import struct


@struct.dataclass
class ReferenceTrajectory:  # 参照軌道
    ref_xs: jnp.ndarray  # (horizon+1, n_x)
    ref_us: jnp.ndarray  # (horizon, n_u)
    lambdas: jnp.ndarray  # (horizon, n_constraints)


@struct.dataclass
class ObstacleData:
    obs_pos: jnp.ndarray
    obs_r: jnp.ndarray

# Constraint Argsの定義
@struct.dataclass
class ConstraintArgs:
    obs_data1: ObstacleData
    obs_data2: ObstacleData


class ControllerBase(ABC):
    def __init__(self, horizon: int = 10):
        self.horizon: int = horizon
        pass

    @abstractmethod
    def calc_input(self, x: jnp.ndarray, target_x: jnp.ndarray, reference_trajectory: ReferenceTrajectory) -> Tuple[jnp.ndarray, ReferenceTrajectory]:
        """
        Args:
            x: 現在の状態
            target_x: 目標状態
            reference_trajectory: 参照軌道
        Returns:
            u: 制御入力
            ref_traj: 更新された参照軌道
        """
        raise NotImplementedError()
