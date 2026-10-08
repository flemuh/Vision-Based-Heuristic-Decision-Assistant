from __future__ import annotations

import math

from .geometry import AutoPlayPoint


def linear_motion_points(start: AutoPlayPoint, target: AutoPlayPoint, steps: int) -> list[AutoPlayPoint]:
    n = max(1, int(steps))
    points: list[AutoPlayPoint] = []
    for i in range(1, n + 1):
        t = i / n
        eased = t * t * (3.0 - 2.0 * t)
        point = AutoPlayPoint(
            int(round(start.x + (target.x - start.x) * eased)),
            int(round(start.y + (target.y - start.y) * eased)),
        )
        if not points or points[-1] != point:
            points.append(point)
    if not points or points[-1] != target:
        points.append(target)
    return points


def _control_points(start: AutoPlayPoint, target: AutoPlayPoint) -> tuple[tuple[float, float], tuple[float, float], float]:
    dx = float(target.x - start.x)
    dy = float(target.y - start.y)
    distance = math.hypot(dx, dy)
    if distance < 2.0:
        return (float(start.x), float(start.y)), (float(target.x), float(target.y)), 0.0
    nx, ny = -dy / distance, dx / distance
    # Conservative bow: enough to be visibly curved, capped so it never makes
    # a large detour across the game UI. Side is deterministic per endpoint.
    lateral = min(28.0, max(2.0, distance * 0.055))
    side = -1.0 if ((start.x + start.y + target.x + target.y) & 1) else 1.0
    offset = lateral * side
    c1 = (start.x + dx * 0.30 + nx * offset, start.y + dy * 0.30 + ny * offset)
    c2 = (start.x + dx * 0.70 + nx * offset, start.y + dy * 0.70 + ny * offset)
    return c1, c2, lateral


def _cubic(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    u = 1.0 - t
    return u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3


def _valid_curve(points: list[AutoPlayPoint], start: AutoPlayPoint, target: AutoPlayPoint, lateral_cap: float) -> bool:
    if not points or points[-1] != target:
        return False
    dx = float(target.x - start.x)
    dy = float(target.y - start.y)
    length = math.hypot(dx, dy)
    if length < 1.0:
        return True
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    previous_progress = -1e-6
    for point in points:
        rx, ry = point.x - start.x, point.y - start.y
        progress = rx * ux + ry * uy
        lateral = abs(rx * nx + ry * ny)
        if progress + 1.5 < previous_progress:
            return False
        if progress < -2.0 or progress > length + 2.0:
            return False
        if lateral > lateral_cap + 3.0:
            return False
        previous_progress = max(previous_progress, progress)
    return True


def bezier_motion_points(start: AutoPlayPoint, target: AutoPlayPoint, steps: int) -> list[AutoPlayPoint]:
    n = max(1, int(steps))
    c1, c2, lateral_cap = _control_points(start, target)
    points: list[AutoPlayPoint] = []
    for i in range(1, n + 1):
        raw_t = i / n
        t = raw_t * raw_t * (3.0 - 2.0 * raw_t)
        point = AutoPlayPoint(
            int(round(_cubic(start.x, c1[0], c2[0], target.x, t))),
            int(round(_cubic(start.y, c1[1], c2[1], target.y, t))),
        )
        if not points or points[-1] != point:
            points.append(point)
    if not points or points[-1] != target:
        points.append(target)
    if not _valid_curve(points, start, target, lateral_cap):
        return linear_motion_points(start, target, steps)
    return points


def motion_points(start: AutoPlayPoint, target: AutoPlayPoint, steps: int) -> list[AutoPlayPoint]:
    """Default live trajectory: V11-style straight smoothstep.

    The Bézier helper is kept for offline experiments, but live input deliberately
    uses the simpler V11 transport so intermediate waypoints do not take detours
    across the game UI.
    """
    return linear_motion_points(start, target, steps)
