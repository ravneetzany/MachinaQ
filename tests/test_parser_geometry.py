from src.parser import StepTextParser

FIXTURE = "nist_sfa/holeTrain/HoleData01.step"
CYLINDER_FIXTURE = "nist_sfa/nist_ctc_01_asme1_ap242-e1.stp"


def _parsed(fixture: str = FIXTURE) -> StepTextParser:
    parser = StepTextParser()
    parser.parse_file(fixture)
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


def test_cylinders_populated_with_resolved_axis_position() -> None:
    parser = _parsed(CYLINDER_FIXTURE)
    assert len(parser.primitives.cylinders) > 0
    surface_id, radius, point, direction = parser.primitives.cylinders[0]
    assert isinstance(surface_id, int)
    assert isinstance(radius, float)
    assert len(point) == 3
    assert len(direction) == 3
    # at least one cylinder should have a non-default (actually resolved) point
    assert any(pt != (0.0, 0.0, 0.0) for _, _, pt, _ in parser.primitives.cylinders)


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


def test_multiline_entity_is_parsed_not_dropped() -> None:
    # A B_SPLINE_SURFACE_WITH_KNOTS-shaped entity whose attribute list spans
    # multiple lines, like real FreeCAD STEP exports produce. Before the
    # re.DOTALL fix, `.` in the entity-matching regex didn't match the
    # embedded newlines and this entity silently failed to parse at all.
    content = (
        "#42 = B_SPLINE_SURFACE_WITH_KNOTS('',1,1,(\n"
        "    (#100,#101),\n"
        "    (#102,#103)),\n"
        "  .UNSPECIFIED.,.F.,.F.,.F.);\n"
        "#43 = PLANE('',#44);\n"
    )
    parser = StepTextParser()
    parser._parse_entities(content)
    assert 42 in parser.entities
    assert parser.entities[42].type == "B_SPLINE_SURFACE_WITH_KNOTS"
    assert 43 in parser.entities
    assert parser.entities[43].type == "PLANE"


def test_toroidal_surface_extracted_with_resolved_axis() -> None:
    content = (
        "#10 = CARTESIAN_POINT('',(1.0,2.0,3.0));\n"
        "#11 = DIRECTION('',(0.0,0.0,1.0));\n"
        "#12 = DIRECTION('',(1.0,0.0,0.0));\n"
        "#13 = AXIS2_PLACEMENT_3D('',#10,#11,#12);\n"
        "#14 = TOROIDAL_SURFACE('',#13,5.849324921932,1.5);\n"
    )
    parser = StepTextParser()
    parser._parse_entities(content)
    parser._extract_primitives()
    assert len(parser.primitives.toroids) == 1
    surface_id, major_radius, minor_radius, point, direction = parser.primitives.toroids[0]
    assert surface_id == 14
    assert major_radius == 5.849324921932
    assert minor_radius == 1.5
    assert point == (1.0, 2.0, 3.0)
    assert direction == (0.0, 0.0, 1.0)


def test_bspline_surface_recorded_as_freeform_surface_id() -> None:
    content = (
        "#42 = B_SPLINE_SURFACE_WITH_KNOTS('',1,1,(\n"
        "    (#100,#101),\n"
        "    (#102,#103)),\n"
        "  .UNSPECIFIED.,.F.,.F.,.F.);\n"
    )
    parser = StepTextParser()
    parser._parse_entities(content)
    parser._extract_primitives()
    assert parser.primitives.freeforms == [42]
