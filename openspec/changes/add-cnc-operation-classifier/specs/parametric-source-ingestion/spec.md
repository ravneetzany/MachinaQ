## Purpose

Provide an alternative input path into MachinaQ's existing primitive/feature representation by statically parsing OpenSCAD (`.scad`) and FreeCAD wrapper-script (`.py`) part definitions, so parts can be classified without exporting STEP files or running a CAD kernel.

## ADDED Requirements

### Requirement: OpenSCAD part parsing
The system SHALL parse a `.scad` file's primitive module calls (including at minimum `cyl`/`cylinder`, `cube`/`cuboid`, `translate`, `rotate`, `union`, `difference`) into a list of geometric primitives, each with a type, dimensions, and its transform relative to the part's origin, without invoking the OpenSCAD binary.

#### Scenario: Axisymmetric part
- **WHEN** the system parses a `.scad` file whose primitives are cylindrical calls sharing one central axis with no rotate/translate that breaks that alignment
- **THEN** the parser reports the part as having a single principal rotational axis

#### Scenario: Multi-axis prismatic part
- **WHEN** the system parses a `.scad` file whose top-level primitives are related by a rotate call that changes their axis alignment (e.g. two perpendicular plates)
- **THEN** the parser reports that the part has no single principal rotational axis, and returns each plate's own primitives with their respective transforms

### Requirement: FreeCAD wrapper-script parsing
The system SHALL statically parse the literal and keyword arguments passed to a part-constructor call (e.g. `make_fastener(...)`, `make_gear(...)`) in a `.py` script to derive the part's geometric primitives, without importing FreeCAD or executing the script.

#### Scenario: Statically resolvable call
- **WHEN** the system parses a `.py` script containing a `make_fastener("ISO4014", diameter="M8", length="30")` call
- **THEN** the parser derives a cylindrical, axisymmetric primitive for the fastener shaft, without requiring a FreeCAD installation

#### Scenario: Non-resolvable call
- **WHEN** a `.py` script's part-constructor arguments cannot be statically resolved (e.g. computed from external state or control flow)
- **THEN** the system reports that file as unparsed, with a reason, rather than raising an unhandled exception

### Requirement: Directory batch discovery
The system SHALL discover individual part definitions across a directory tree, distinguishing part files from shared/support files.

#### Scenario: OpenSCAD library
- **WHEN** the system scans a directory such as `openscad-parts-library`
- **THEN** it treats each `.scad` file outside of `lib/`, `docs/`, and `tests/` as one part to ingest

#### Scenario: FreeCAD library
- **WHEN** the system scans a directory such as `freecad-parts-library`
- **THEN** it treats each `.py` file defining a `make_*` part-constructor function, outside of `lib/`, as one part to ingest, and does not treat `lib/common.py` as a part

### Requirement: Output compatible with existing feature representation
Primitives produced by ingestion SHALL use the same primitive/feature data shape the STEP-based pipeline already emits, so existing consumers work unmodified.

#### Scenario: Feature detection on an ingested part
- **WHEN** a `.scad` or `.py` part's ingested primitives are passed to the existing feature detector
- **THEN** the detector returns features with the same field structure it returns for STEP-derived primitives
