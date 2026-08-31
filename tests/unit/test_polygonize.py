"""Mask -> polygon tracing/simplification (dependency-free, no ML needed)."""

from horos.backends.sam.polygonize import mask_to_polygon


def _grid(text: str) -> list[list[int]]:
    rows = [line.strip() for line in text.strip().splitlines()]
    return [[1 if ch == "#" else 0 for ch in row] for row in rows]


def _points(flat):
    return {(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)}


def test_solid_square():
    mask = _grid("""
    ........
    .#####..
    .#####..
    .#####..
    .#####..
    ........
    """)
    poly = mask_to_polygon(mask, epsilon=0.9)
    assert poly is not None
    corners = _points(poly)
    assert {(1, 1), (5, 1), (5, 4), (1, 4)} <= corners
    assert len(poly) // 2 <= 8  # simplification collapses the edges


def test_l_shape_keeps_concave_corner():
    mask = _grid("""
    #####....
    #####....
    #########
    #########
    """)
    poly = mask_to_polygon(mask, epsilon=0.9)
    assert poly is not None
    points = _points(poly)
    # not collapsed to the outer rectangle: the concave step around x=4..5 survives
    assert len(points) >= 6
    assert any(0 < x < 8 and y < 2 for x, y in points)
    assert (8, 2) in points or (8, 3) in points


def test_empty_mask_is_none():
    assert mask_to_polygon(_grid("....\n....")) is None


def test_single_pixel_is_none():
    assert mask_to_polygon(_grid("....\n.#..\n....")) is None


def test_polygon_is_flat_floats():
    poly = mask_to_polygon(_grid("###\n###\n###"), epsilon=0.5)
    assert poly is not None
    assert len(poly) % 2 == 0
    assert all(isinstance(v, float) for v in poly)


def test_works_with_numpy_masks():
    numpy = __import__("numpy")
    mask = numpy.zeros((10, 12), dtype=bool)
    mask[2:8, 3:10] = True
    poly = mask_to_polygon(mask, epsilon=0.9)
    assert poly is not None
    xs, ys = poly[0::2], poly[1::2]
    assert min(xs) == 3 and max(xs) == 9 and min(ys) == 2 and max(ys) == 7
