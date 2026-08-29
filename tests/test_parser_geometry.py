from src.parser import StepTextParser

FIXTURE = "nist_sfa/holeTrain/HoleData01.step"


def _parsed() -> StepTextParser:
    parser = StepTextParser()
    parser.parse_file(FIXTURE)
    return parser


def test_planes_populated_with_normal_and_point() -> None:
    parser = _parsed()
    assert len(parser.primitives.planes) > 0
    surface_id, normal, point = parser.primitives.planes[0]
    assert isinstance(surface_id, int)
    assert len(normal) == 3
    assert len(point) == 3
    # at least one plane should have a non-default (resolved) normal
    assert any(n != (0.0, 0.0, 1.0) for _, n, _ in parser.primitives.planes)


def test_face_adjacency_matches_topology_maps() -> None:
    parser = _parsed()
    adjacency = parser.get_face_adjacency()
    face_edges, edge_to_faces, _, _ = parser._ensure_topology()

    assert set(adjacency.keys()) == set(face_edges.keys())
    for fid, edges in face_edges.items():
        expected: set = set()
        for ec in edges:
            expected |= edge_to_faces.get(ec, set())
        expected.discard(fid)
        assert adjacency[fid] == expected


def test_face_bounding_extents_match_known_rectangular_face() -> None:
    parser = _parsed()
    surface_id, normal, _ = parser.primitives.planes[0]
    face_ids = parser.get_surface_face_ids(surface_id)
    assert face_ids

    extents = parser.get_face_bounding_extents(face_ids[0], normal)
    assert extents is not None
    long_extent, short_extent = extents
    # HoleData01's first face is a 50x10 rectangle (verified against the raw
    # STEP vertex coordinates for this fixture: x spans -25..25, z spans 0..10)
    assert abs(long_extent - 50.0) < 0.5
    assert abs(short_extent - 10.0) < 0.5


def test_topology_is_cached_not_rebuilt() -> None:
    parser = _parsed()
    first = parser._ensure_topology()
    second = parser._ensure_topology()
    assert first is second
