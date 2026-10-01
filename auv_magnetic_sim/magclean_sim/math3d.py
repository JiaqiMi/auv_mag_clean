from typing import Callable

import numpy as np


MU0_OVER_4PI = 1.0e-7
TESLA_TO_NT = 1.0e9


def rotation_body_to_world(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the Z-Y-X Euler rotation matrix from body to world."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz.dot(ry).dot(rx)


def dipole_linear_map_nt(source_position_m: np.ndarray, evaluation_points_m: np.ndarray) -> np.ndarray:
    """Map a magnetic dipole moment [A m^2] to B [nT] at fixed points."""
    points = np.atleast_2d(evaluation_points_m).astype(float)
    relative = points - np.asarray(source_position_m, dtype=float)[None, :]
    radius = np.linalg.norm(relative, axis=1)
    if np.any(radius < 1.0e-4):
        raise ValueError("Evaluation point is too close to a simulated dipole source")
    rhat = relative / radius[:, None]
    identity = np.eye(3)[None, :, :]
    outer = np.einsum("pi,pj->pij", rhat, rhat)
    scale = MU0_OVER_4PI * TESLA_TO_NT / (radius ** 3)
    return scale[:, None, None] * (3.0 * outer - identity)


def dipole_field_nt(
    moment_am2: np.ndarray, source_position_m: np.ndarray, evaluation_points_m: np.ndarray
) -> np.ndarray:
    mapping = dipole_linear_map_nt(source_position_m, evaluation_points_m)
    moment = np.asarray(moment_am2, dtype=float)
    if moment.ndim == 1:
        return np.einsum("pij,j->pi", mapping, moment)
    return np.einsum("pij,nj->npi", mapping, moment)


def gradient_stencil_points(step_m: float = 0.01) -> np.ndarray:
    origin = np.zeros(3)
    points = [origin]
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step_m
        points.extend([delta.copy(), -delta.copy()])
    return np.asarray(points)


def stencil_fields_to_state8(fields_nt: np.ndarray, step_m: float = 0.01) -> np.ndarray:
    """Convert origin/+/- axis field samples to [B0, five gradient DOF]."""
    fields = np.asarray(fields_nt, dtype=float)
    single = fields.ndim == 2
    if single:
        fields = fields[None, :, :]
    if fields.shape[1:] != (7, 3):
        raise ValueError("Expected stencil field shape (..., 7, 3)")
    b0 = fields[:, 0, :]
    gradient = np.empty((fields.shape[0], 3, 3), dtype=float)
    for axis in range(3):
        plus = fields[:, 1 + 2 * axis, :]
        minus = fields[:, 2 + 2 * axis, :]
        gradient[:, :, axis] = (plus - minus) / (2.0 * step_m)
    # Numerical averaging enforces the source-free symmetry used by the bench solver.
    gradient = 0.5 * (gradient + np.swapaxes(gradient, 1, 2))
    trace = np.trace(gradient, axis1=1, axis2=2) / 3.0
    gradient[:, 0, 0] -= trace
    gradient[:, 1, 1] -= trace
    gradient[:, 2, 2] -= trace
    state = np.column_stack(
        [
            b0,
            gradient[:, 0, 0],
            gradient[:, 1, 1],
            gradient[:, 0, 1],
            gradient[:, 0, 2],
            gradient[:, 1, 2],
        ]
    )
    return state[0] if single else state


def array_design_matrix(positions_m: np.ndarray) -> np.ndarray:
    """Build the 3N x 8 first-order field matrix with Maxwell constraints."""
    positions = np.asarray(positions_m, dtype=float)
    rows = []
    for dx, dy, dz in positions:
        # Bx = B0x + Gxx*dx + Gxy*dy + Gxz*dz
        rows.append([1, 0, 0, dx, 0, dy, dz, 0])
        # By = B0y + Gxy*dx + Gyy*dy + Gyz*dz
        rows.append([0, 1, 0, 0, dy, dx, 0, dz])
        # Bz = B0z + Gxz*dx + Gyz*dy - (Gxx+Gyy)*dz
        rows.append([0, 0, 1, -dz, -dz, 0, dx, dy])
    return np.asarray(rows, dtype=float)


def state8_to_array_field(state8: np.ndarray, positions_m: np.ndarray) -> np.ndarray:
    state = np.asarray(state8, dtype=float)
    design = array_design_matrix(positions_m)
    if state.ndim == 1:
        return design.dot(state).reshape(len(positions_m), 3)
    return state.dot(design.T).reshape(state.shape[0], len(positions_m), 3)


def reconstruction_operator(
    positions_m: np.ndarray, regularization: float = 1.0e-10
) -> np.ndarray:
    design = array_design_matrix(positions_m)
    lhs = design.T.dot(design) + regularization * np.eye(design.shape[1])
    return np.linalg.solve(lhs, design.T)


def reconstruct_state8(array_fields_nt: np.ndarray, positions_m: np.ndarray) -> np.ndarray:
    fields = np.asarray(array_fields_nt, dtype=float)
    operator = reconstruction_operator(positions_m)
    if fields.ndim == 2:
        return operator.dot(fields.reshape(-1))
    return fields.reshape(fields.shape[0], -1).dot(operator.T)


def finite_difference_state8(
    field_function: Callable[[np.ndarray], np.ndarray], step_m: float = 0.01
) -> np.ndarray:
    fields = field_function(gradient_stencil_points(step_m))
    return stencil_fields_to_state8(fields, step_m)

