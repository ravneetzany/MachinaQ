from src.primitive import SurfacePrimitive


def test_surface_primitive_defaults_to_empty_adjacency() -> None:
    prim = SurfacePrimitive(face_id=1, type="planar", details={})
    assert prim.adjacent_face_ids == []


def test_surface_primitive_with_explicit_adjacency() -> None:
    prim = SurfacePrimitive(face_id=1, type="planar", details={}, adjacent_face_ids=[2, 3])
    assert prim.adjacent_face_ids == [2, 3]
