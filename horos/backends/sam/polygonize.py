"""Binary mask -> simplified polygon, dependency-free.

The project deliberately has no OpenCV dependency, so the two classic pieces
are implemented directly: Moore-neighbor boundary tracing (with Jacob's
stopping criterion) and Douglas-Peucker simplification. Pixel coordinates use
cell centers, so a polygon vertex (x, y) sits on pixel (row y, col x).

The mask may be any 2D indexable (numpy array or nested lists); only the blob
containing the top-most/left-most foreground pixel is traced — SAM's
box-prompted masks are single blobs in practice.
"""

from __future__ import annotations

# Moore neighborhood, clockwise, starting west: (dx, dy)
_MOORE = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]


def _first_foreground(mask, height: int, width: int) -> tuple[int, int] | None:
    any_row = getattr(mask, "any", None)
    if any_row is not None and hasattr(mask, "shape"):  # numpy fast path
        rows = mask.any(axis=1)
        for y in range(height):
            if rows[y]:
                row = mask[y]
                for x in range(width):
                    if row[x]:
                        return x, y
        return None
    for y in range(height):
        row = mask[y]
        for x in range(width):
            if row[x]:
                return x, y
    return None


def _trace_boundary(mask, height: int, width: int) -> list[tuple[int, int]] | None:
    """Moore-neighbor tracing, keeping the last BACKGROUND cell examined —
    the clockwise scan around each new pixel starts from that background cell,
    which is what keeps the walk glued to the boundary."""
    start = _first_foreground(mask, height, width)
    if start is None:
        return None

    def on(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height and bool(mask[y][x])

    boundary = [start]
    p = start
    bg = (start[0] - 1, start[1])  # west neighbor: background by scan order
    first_step: tuple[tuple[int, int], tuple[int, int]] | None = None
    limit = 4 * height * width

    for _ in range(limit):
        idx0 = _MOORE.index((bg[0] - p[0], bg[1] - p[1]))
        found = None
        for step in range(1, 9):
            idx = (idx0 + step) % 8
            nxt = (p[0] + _MOORE[idx][0], p[1] + _MOORE[idx][1])
            if on(*nxt):
                found = (idx, nxt)
                break
        if found is None:
            return boundary  # isolated single pixel
        idx, nxt = found
        # stop when we are about to repeat the very first move
        if first_step is None:
            first_step = (p, nxt)
        elif (p, nxt) == first_step:
            break
        prev_idx = (idx - 1) % 8
        bg = (p[0] + _MOORE[prev_idx][0], p[1] + _MOORE[prev_idx][1])
        p = nxt
        if p != boundary[-1]:
            boundary.append(p)
    if len(boundary) > 1 and boundary[-1] == boundary[0]:
        boundary.pop()
    return boundary


def _perpendicular_distance(pt, a, b) -> float:
    (px, py), (ax, ay), (bx, by) = pt, a, b
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    return abs(dy * px - dx * py + bx * ay - by * ax) / (dx * dx + dy * dy) ** 0.5


def _douglas_peucker(points: list[tuple[int, int]], epsilon: float) -> list[tuple[int, int]]:
    if len(points) < 3:
        return points
    stack = [(0, len(points) - 1)]
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    while stack:
        first, last = stack.pop()
        max_dist, index = 0.0, first
        for i in range(first + 1, last):
            dist = _perpendicular_distance(points[i], points[first], points[last])
            if dist > max_dist:
                max_dist, index = dist, i
        if max_dist > epsilon:
            keep[index] = True
            stack.append((first, index))
            stack.append((index, last))
    return [p for p, k in zip(points, keep, strict=True) if k]


def mask_to_polygon(mask, *, epsilon: float = 1.5, min_points: int = 3) -> list[float] | None:
    """Trace the mask's boundary and return a simplified flat polygon
    [x1, y1, x2, y2, ...], or None when the mask is empty or degenerate."""
    height = len(mask)
    width = len(mask[0]) if height else 0
    if not height or not width:
        return None
    boundary = _trace_boundary(mask, height, width)
    if boundary is None or len(boundary) < min_points:
        return None
    # close the ring for DP, then drop the duplicate endpoint
    ring = [*boundary, boundary[0]]
    simplified = _douglas_peucker(ring, epsilon)[:-1]
    if len(simplified) < min_points:
        return None
    flat: list[float] = []
    for x, y in simplified:
        flat.extend((float(x), float(y)))
    return flat
