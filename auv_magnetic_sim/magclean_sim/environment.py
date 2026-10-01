from dataclasses import dataclass

import numpy as np

from .math3d import dipole_field_nt, rotation_body_to_world


@dataclass
class NavigationProfile:
    time_s: np.ndarray
    position_world_m: np.ndarray
    attitude_rpy_rad: np.ndarray
    source_states: np.ndarray
    temperature_c: np.ndarray
    environment_body_nt: np.ndarray
    target_body_nt: np.ndarray
    target_distance_m: np.ndarray


def generate_navigation_profile(
    duration_s: float,
    sample_rate_hz: float,
    target_enabled: bool,
    target_moment_am2: float,
    target_cross_track_m: float,
    target_depth_offset_m: float,
    high_maneuver_enabled: bool = False,
) -> NavigationProfile:
    sample_count = max(int(round(duration_s * sample_rate_hz)), 4)
    time_s = np.arange(sample_count, dtype=float) / sample_rate_hz
    phase = time_s / max(duration_s, 1.0)
    x = -30.0 + 60.0 * phase
    if high_maneuver_enabled:
        y = 2.2 * np.sin(4.0 * np.pi * phase) + 0.55 * np.sin(15.0 * np.pi * phase)
        z = -20.0 + 1.15 * np.sin(6.0 * np.pi * phase) + 0.30 * np.sin(18.0 * np.pi * phase)
    else:
        y = 0.7 * np.sin(2.0 * np.pi * phase)
        z = -20.0 + 0.35 * np.sin(4.0 * np.pi * phase)
    position = np.column_stack([x, y, z])

    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.arctan2(dy, dx)
    if high_maneuver_enabled:
        roll = np.deg2rad(12.0) * np.sin(11.0 * np.pi * phase)
        pitch = np.deg2rad(8.0) * np.sin(8.0 * np.pi * phase + 0.4)
    else:
        roll = np.deg2rad(2.0) * np.sin(5.0 * np.pi * phase)
        pitch = np.deg2rad(1.5) * np.sin(3.0 * np.pi * phase + 0.4)
    attitude = np.column_stack([roll, pitch, yaw])

    source_states = np.zeros((sample_count, 4), dtype=float)
    if high_maneuver_enabled:
        # Five disjoint manoeuvre events: launch burst, DC-bus load step,
        # steering cycle, thrust reversal and emergency acceleration. Edges are
        # about 0.10-0.18 s so a 20 Hz G-882-class output is challenged near
        # its useful temporal bandwidth, while the internal source state stays
        # deterministic and auditable.
        time_s = np.asarray(time_s)

        def pulse(center_fraction: float, width_s: float, rise_s: float) -> np.ndarray:
            center = center_fraction * duration_s
            start = center - 0.5 * width_s
            stop = center + 0.5 * width_s
            return 0.5 * (
                np.tanh((time_s - start) / max(rise_s, 1.0e-3))
                - np.tanh((time_s - stop) / max(rise_s, 1.0e-3))
            )

        launch = pulse(0.14, 2.1, 0.12)
        bus_step = pulse(0.32, 3.0, 0.16)
        steering = pulse(0.49, 2.6, 0.13)
        reversal = pulse(0.67, 2.0, 0.10)
        emergency = pulse(0.84, 2.5, 0.11)
        source_states[:, 0] = (
            0.28
            + 1.30 * launch
            + 0.82 * bus_step
            + 0.55 * steering
            - 1.42 * reversal
            + 1.45 * emergency
            + 0.15 * np.sin(2.0 * np.pi * 2.6 * time_s)
        )
        source_states[:, 1] = (
            0.26
            + 0.72 * np.abs(source_states[:, 0])
            + 0.48 * bus_step
            + 0.24 * emergency
        )
        source_states[:, 2] = (
            1.34 * np.sin(2.0 * np.pi * 1.7 * time_s) * steering
            - 1.18 * reversal
            + 0.72 * emergency
        )
        source_states[:, 3] = (
            0.40
            + 0.33 * np.abs(source_states[:, 0])
            + 0.36 * bus_step
            + 0.28 * steering
        )
        source_states[:, 0] = np.clip(source_states[:, 0], -1.55, 1.75)
        source_states[:, [1, 3]] = np.clip(source_states[:, [1, 3]], 0.0, 1.70)
        source_states[:, 2] = np.clip(source_states[:, 2], -1.45, 1.45)
    else:
        source_states[:, 0] = 0.55 + 0.20 * np.sin(6.0 * np.pi * phase) + 0.12 * (phase > 0.55)
        source_states[:, 1] = 0.42 + 0.18 * ((np.arange(sample_count) // max(sample_count // 10, 1)) % 2)
        source_states[:, 2] = 0.55 * np.sin(10.0 * np.pi * phase) * ((phase > 0.2) & (phase < 0.82))
        source_states[:, 3] = 0.62 + 0.05 * np.sin(2.0 * np.pi * phase + 1.1)
        source_states[:, [0, 1, 3]] = np.clip(source_states[:, [0, 1, 3]], 0.0, 1.0)
    temperature = 20.0 + 5.0 * phase + 0.6 * source_states[:, 0]

    world_field = np.array([21500.0, 1200.0, 40800.0])
    environment_body = np.zeros((sample_count, 3), dtype=float)
    target_body = np.zeros((sample_count, 3), dtype=float)
    target_position = np.array([0.0, target_cross_track_m, -20.0 + target_depth_offset_m])
    target_moment = np.array([0.25, 0.90, 0.35])
    target_moment = target_moment / np.linalg.norm(target_moment) * target_moment_am2
    target_distance = np.linalg.norm(position - target_position[None, :], axis=1)
    if target_enabled:
        target_world = dipole_field_nt(target_moment, target_position, position)
    else:
        target_world = np.zeros((sample_count, 3), dtype=float)
    for index in range(sample_count):
        body_to_world = rotation_body_to_world(*attitude[index])
        world_to_body = body_to_world.T
        environment_body[index] = world_to_body.dot(world_field)
        target_body[index] = world_to_body.dot(target_world[index])
    return NavigationProfile(
        time_s=time_s,
        position_world_m=position,
        attitude_rpy_rad=attitude,
        source_states=source_states,
        temperature_c=temperature,
        environment_body_nt=environment_body,
        target_body_nt=target_body,
        target_distance_m=target_distance,
    )
