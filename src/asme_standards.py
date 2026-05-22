"""ASME and ISO standard hole sizes for machined parts.

ASME Standards (inch units):
    ASME B18.2.8  — Clearance holes for bolts and screws (close / normal / loose fit)
    ASME B1.1     — Unified screw thread tap drill sizes (UNC / UNF)
    ASME B94.11M  — Standard drill sizes (numbered #1-#80, letter A-Z, fractional)
    ASME B18.3    — Socket head cap screw counterbore diameters
    ASME B18.6.3  — Flat-head countersink diameters (82°)

ISO Metric Standards (mm units):
    ISO 273       — Metric clearance hole diameters (fine / medium / coarse series)
    ISO 965-1     — Metric tap drill sizes (coarse pitch M1–M64 + fine pitch series)
    DIN 338       — Standard metric drill bit sizes (0.5–80 mm)
    ISO 2768      — General tolerance classes reference

STEP FILE UNIT AUTO-DETECTION:
    NIST STC (Sheet Metal Test Cases): declare 'INCH', geometry in inches
    NIST CTC/FTC/STC (Calibration Cases) + holeTrain: declare MILLIMETRE, geometry in mm
    The parser reads the header keyword to select ASME (inch) or ISO (mm) lookup.
"""

from __future__ import annotations
import bisect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Practical range limits (inches, radius)
# ---------------------------------------------------------------------------
#: Smallest ASME standard drill (#80 = 0.0135" dia → radius 0.00675")
MIN_STANDARD_RADIUS_IN: float = 0.00675

#: Largest ASME clearance hole (1-1/2" loose = 1.6563" dia → radius 0.828")
MAX_STANDARD_RADIUS_IN: float = 0.828

#: Below this radius the hole is considered micro / non-standard for general machining
PRACTICAL_MIN_RADIUS_IN: float = 0.030   # diameter 0.060" ≈ drill #52

#: Snap tolerance — how far (fractional) a detected diameter may deviate from
#: the nearest standard and still be considered a match (5 % default)
DEFAULT_TOLERANCE: float = 0.05


# ---------------------------------------------------------------------------
# ASME B18.2.8 — Clearance hole diameters (diameters in inches)
# ---------------------------------------------------------------------------
CLEARANCE_HOLES: Dict[str, Dict[str, float]] = {
    "#0":    {"close": 0.0700, "normal": 0.0760, "loose": 0.0960},
    "#2":    {"close": 0.0960, "normal": 0.1040, "loose": 0.1160},
    "#4":    {"close": 0.1200, "normal": 0.1285, "loose": 0.1440},
    "#6":    {"close": 0.1480, "normal": 0.1570, "loose": 0.1770},
    "#8":    {"close": 0.1770, "normal": 0.1935, "loose": 0.2010},
    "#10":   {"close": 0.2010, "normal": 0.2130, "loose": 0.2280},
    "1/4":   {"close": 0.2660, "normal": 0.2810, "loose": 0.2969},
    "5/16":  {"close": 0.3320, "normal": 0.3438, "loose": 0.3594},
    "3/8":   {"close": 0.3970, "normal": 0.4063, "loose": 0.4219},
    "7/16":  {"close": 0.4531, "normal": 0.4688, "loose": 0.4844},
    "1/2":   {"close": 0.5312, "normal": 0.5469, "loose": 0.5625},
    "9/16":  {"close": 0.5938, "normal": 0.6094, "loose": 0.6250},
    "5/8":   {"close": 0.6563, "normal": 0.6719, "loose": 0.6875},
    "3/4":   {"close": 0.8125, "normal": 0.8281, "loose": 0.8438},
    "7/8":   {"close": 0.9375, "normal": 0.9531, "loose": 0.9688},
    "1":     {"close": 1.0625, "normal": 1.0938, "loose": 1.1250},
    "1-1/4": {"close": 1.3281, "normal": 1.3594, "loose": 1.4063},
    "1-1/2": {"close": 1.5781, "normal": 1.6094, "loose": 1.6563},
}

# ---------------------------------------------------------------------------
# ASME B1.1 — Tap drill diameters (inches)
# ---------------------------------------------------------------------------
TAP_DRILLS: Dict[str, float] = {
    "#0-80-UNF":  0.0465, "#2-56-UNC":  0.0700, "#2-64-UNF":  0.0700,
    "#4-40-UNC":  0.0890, "#4-48-UNF":  0.0935, "#6-32-UNC":  0.1065,
    "#6-40-UNF":  0.1130, "#8-32-UNC":  0.1360, "#8-36-UNF":  0.1360,
    "#10-24-UNC": 0.1495, "#10-32-UNF": 0.1590,
    "1/4-20-UNC": 0.2010, "1/4-28-UNF": 0.2130,
    "5/16-18-UNC":0.2570, "5/16-24-UNF":0.2720,
    "3/8-16-UNC": 0.3125, "3/8-24-UNF": 0.3320,
    "7/16-14-UNC":0.3680, "7/16-20-UNF":0.3906,
    "1/2-13-UNC": 0.4219, "1/2-20-UNF": 0.4531,
    "9/16-12-UNC":0.4844, "9/16-18-UNF":0.5156,
    "5/8-11-UNC": 0.5313, "5/8-18-UNF": 0.5781,
    "3/4-10-UNC": 0.6563, "3/4-16-UNF": 0.6875,
    "7/8-9-UNC":  0.7656, "7/8-14-UNF": 0.8125,
    "1-8-UNC":    0.8750, "1-12-UNF":   0.9219,
}

# ---------------------------------------------------------------------------
# ASME B94.11M — Numbered drills #1–#80 (diameters in inches)
# ---------------------------------------------------------------------------
NUMBERED_DRILLS: Dict[int, float] = {
     1: 0.2280,  2: 0.2210,  3: 0.2130,  4: 0.2090,  5: 0.2055,
     6: 0.2040,  7: 0.2010,  8: 0.1990,  9: 0.1960, 10: 0.1935,
    11: 0.1910, 12: 0.1890, 13: 0.1850, 14: 0.1820, 15: 0.1800,
    16: 0.1770, 17: 0.1730, 18: 0.1695, 19: 0.1660, 20: 0.1610,
    21: 0.1590, 22: 0.1570, 23: 0.1540, 24: 0.1520, 25: 0.1495,
    26: 0.1470, 27: 0.1440, 28: 0.1405, 29: 0.1360, 30: 0.1285,
    31: 0.1200, 32: 0.1160, 33: 0.1130, 34: 0.1110, 35: 0.1100,
    36: 0.1065, 37: 0.1040, 38: 0.1015, 39: 0.0995, 40: 0.0980,
    41: 0.0960, 42: 0.0935, 43: 0.0890, 44: 0.0860, 45: 0.0820,
    46: 0.0810, 47: 0.0785, 48: 0.0760, 49: 0.0730, 50: 0.0700,
    51: 0.0670, 52: 0.0635, 53: 0.0595, 54: 0.0550, 55: 0.0520,
    56: 0.0465, 57: 0.0430, 58: 0.0420, 59: 0.0410, 60: 0.0400,
    61: 0.0390, 62: 0.0380, 63: 0.0370, 64: 0.0360, 65: 0.0350,
    66: 0.0330, 67: 0.0320, 68: 0.0310, 69: 0.0292, 70: 0.0280,
    71: 0.0260, 72: 0.0250, 73: 0.0240, 74: 0.0225, 75: 0.0210,
    76: 0.0200, 77: 0.0180, 78: 0.0160, 79: 0.0145, 80: 0.0135,
}

# ---------------------------------------------------------------------------
# ASME B94.11M — Letter drills A–Z (diameters in inches)
# ---------------------------------------------------------------------------
LETTER_DRILLS: Dict[str, float] = {
    'A': 0.2340, 'B': 0.2380, 'C': 0.2420, 'D': 0.2460, 'E': 0.2500,
    'F': 0.2570, 'G': 0.2610, 'H': 0.2660, 'I': 0.2720, 'J': 0.2770,
    'K': 0.2810, 'L': 0.2900, 'M': 0.2950, 'N': 0.3020, 'O': 0.3160,
    'P': 0.3230, 'Q': 0.3320, 'R': 0.3390, 'S': 0.3480, 'T': 0.3580,
    'U': 0.3680, 'V': 0.3770, 'W': 0.3860, 'X': 0.3970, 'Y': 0.4040,
    'Z': 0.4130,
}

# ---------------------------------------------------------------------------
# ASME B94.11M — Fractional drills (diameters in inches)
# ---------------------------------------------------------------------------
FRACTIONAL_DRILLS: Dict[str, float] = {
    "1/64":  0.0156, "1/32":  0.0313, "3/64":  0.0469, "1/16":  0.0625,
    "5/64":  0.0781, "3/32":  0.0938, "7/64":  0.1094, "1/8":   0.1250,
    "9/64":  0.1406, "5/32":  0.1563, "11/64": 0.1719, "3/16":  0.1875,
    "13/64": 0.2031, "7/32":  0.2188, "15/64": 0.2344, "1/4":   0.2500,
    "17/64": 0.2656, "9/32":  0.2813, "19/64": 0.2969, "5/16":  0.3125,
    "21/64": 0.3281, "11/32": 0.3438, "23/64": 0.3594, "3/8":   0.3750,
    "25/64": 0.3906, "13/32": 0.4063, "27/64": 0.4219, "7/16":  0.4375,
    "29/64": 0.4531, "15/32": 0.4688, "31/64": 0.4844, "1/2":   0.5000,
    "33/64": 0.5156, "17/32": 0.5313, "35/64": 0.5469, "9/16":  0.5625,
    "37/64": 0.5781, "19/32": 0.5938, "5/8":   0.6250, "21/32": 0.6563,
    "11/16": 0.6875, "3/4":   0.7500, "13/16": 0.8125, "7/8":   0.8750,
    "15/16": 0.9375, "1":     1.0000, "1-1/4": 1.2500, "1-1/2": 1.5000,
}

# ---------------------------------------------------------------------------
# ASME B18.3 — SHCS Counterbore diameters (inches)
# ---------------------------------------------------------------------------
COUNTERBORE_SHCS: Dict[str, Dict[str, float]] = {
    "#4":   {"cbore_dia": 0.2500, "cbore_depth": 0.1640},
    "#6":   {"cbore_dia": 0.3125, "cbore_depth": 0.1875},
    "#8":   {"cbore_dia": 0.3750, "cbore_depth": 0.2188},
    "#10":  {"cbore_dia": 0.4375, "cbore_depth": 0.2500},
    "1/4":  {"cbore_dia": 0.5000, "cbore_depth": 0.2500},
    "5/16": {"cbore_dia": 0.5625, "cbore_depth": 0.3125},
    "3/8":  {"cbore_dia": 0.6875, "cbore_depth": 0.3750},
    "7/16": {"cbore_dia": 0.8125, "cbore_depth": 0.4375},
    "1/2":  {"cbore_dia": 0.8750, "cbore_depth": 0.5000},
    "5/8":  {"cbore_dia": 1.0625, "cbore_depth": 0.6250},
    "3/4":  {"cbore_dia": 1.2500, "cbore_depth": 0.7500},
    "1":    {"cbore_dia": 1.6875, "cbore_depth": 1.0000},
}

# ---------------------------------------------------------------------------
# ASME B18.6.3 — Countersink diameters, 82° (inches)
# ---------------------------------------------------------------------------
COUNTERSINK_82DEG: Dict[str, float] = {
    "#4": 0.2720, "#6": 0.3240, "#8": 0.3760, "#10": 0.4280,
    "1/4": 0.5310, "5/16": 0.6560, "3/8": 0.7810,
    "1/2": 1.0000, "5/8": 1.2500,
}


# ---------------------------------------------------------------------------
# Master lookup table — built once at import time
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StandardEntry:
    """One entry in the ASME master list."""
    diameter_in: float     # hole diameter in inches
    radius_in: float       # = diameter_in / 2
    label: str             # human-readable, e.g. "1/4\" normal clearance"
    standard: str          # e.g. "ASME B18.2.8"
    category: str          # clearance | tap_drill | numbered_drill | letter_drill |
                           # fractional_drill | counterbore | countersink
    bolt_size: str         # e.g. "1/4", "#10", "3/8"
    fit: str               # close/normal/loose for clearance; UNC/UNF for tap; "" otherwise


def _build_master() -> List[StandardEntry]:
    entries: List[StandardEntry] = []

    # B18.2.8 clearance holes
    for bolt, fits in CLEARANCE_HOLES.items():
        for fit, dia in fits.items():
            entries.append(StandardEntry(
                diameter_in=dia, radius_in=dia / 2,
                label=f'{bolt}" {fit} clearance (B18.2.8)',
                standard="ASME B18.2.8",
                category="clearance",
                bolt_size=bolt, fit=fit,
            ))

    # B1.1 tap drills
    for name, dia in TAP_DRILLS.items():
        parts = name.rsplit('-', 1)
        thread = parts[0] if len(parts) == 2 else name
        fit    = parts[1] if len(parts) == 2 else ""
        entries.append(StandardEntry(
            diameter_in=dia, radius_in=dia / 2,
            label=f'{name} tap drill (B1.1)',
            standard="ASME B1.1",
            category="tap_drill",
            bolt_size=thread, fit=fit,
        ))

    # B94.11M numbered drills
    for num, dia in NUMBERED_DRILLS.items():
        entries.append(StandardEntry(
            diameter_in=dia, radius_in=dia / 2,
            label=f'Drill #{num} ({dia:.4f}") (B94.11M)',
            standard="ASME B94.11M",
            category="numbered_drill",
            bolt_size=f"#{num}", fit="",
        ))

    # B94.11M letter drills
    for letter, dia in LETTER_DRILLS.items():
        entries.append(StandardEntry(
            diameter_in=dia, radius_in=dia / 2,
            label=f'Drill {letter} ({dia:.4f}") (B94.11M)',
            standard="ASME B94.11M",
            category="letter_drill",
            bolt_size=letter, fit="",
        ))

    # B94.11M fractional drills
    for frac, dia in FRACTIONAL_DRILLS.items():
        entries.append(StandardEntry(
            diameter_in=dia, radius_in=dia / 2,
            label=f'{frac}" drill ({dia:.4f}") (B94.11M)',
            standard="ASME B94.11M",
            category="fractional_drill",
            bolt_size=frac, fit="",
        ))

    # B18.3 counterbores
    for bolt, dims in COUNTERBORE_SHCS.items():
        dia = dims["cbore_dia"]
        entries.append(StandardEntry(
            diameter_in=dia, radius_in=dia / 2,
            label=f'{bolt}" SHCS counterbore (B18.3)',
            standard="ASME B18.3",
            category="counterbore",
            bolt_size=bolt, fit="",
        ))

    # B18.6.3 countersinks
    for bolt, dia in COUNTERSINK_82DEG.items():
        entries.append(StandardEntry(
            diameter_in=dia, radius_in=dia / 2,
            label=f'{bolt}" 82° countersink (B18.6.3)',
            standard="ASME B18.6.3",
            category="countersink",
            bolt_size=bolt, fit="",
        ))

    # Sort by diameter for bisect-based lookup
    entries.sort(key=lambda e: e.diameter_in)
    return entries


#: All ASME standard entries, sorted by diameter ascending.
MASTER: List[StandardEntry] = _build_master()
_MASTER_DIAMETERS: List[float] = [e.diameter_in for e in MASTER]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def snap_to_standard(
    radius_in: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Optional[StandardEntry]:
    """Return the closest ASME standard entry for a detected radius.

    Args:
        radius_in:  Detected hole radius in inches.
        tolerance:  Maximum allowed fractional deviation from standard diameter.
                    E.g. 0.05 = 5%.  Returns None if no match within tolerance.

    Returns:
        The best-matching StandardEntry, or None if outside tolerance.
    """
    dia = radius_in * 2.0
    idx = bisect.bisect_left(_MASTER_DIAMETERS, dia)

    best: Optional[StandardEntry] = None
    best_err = float('inf')

    for i in (idx - 1, idx):
        if 0 <= i < len(MASTER):
            err = abs(MASTER[i].diameter_in - dia) / MASTER[i].diameter_in
            if err < best_err:
                best_err = err
                best = MASTER[i]

    if best is None or best_err > tolerance:
        return None
    return best


def classify_hole(
    radius_in: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Classify a detected hole radius against ASME standards.

    Returns a dict with keys:
        matched       bool — True if within tolerance of a standard size
        label         str  — ASME label or "non-standard"
        standard      str
        category      str
        bolt_size     str
        fit           str
        snap_radius   float — snapped standard radius (inches); equals radius_in if no match
        raw_radius    float — original detected radius
        error_pct     float — % deviation from nearest standard (0.0 if no match found)
        diameter_in   float — diameter in inches
        diameter_mm   float — diameter in mm (for display)
        note          str   — human-readable note
    """
    entry = snap_to_standard(radius_in, tolerance)
    dia   = radius_in * 2.0

    if entry is None:
        # Find nearest anyway for the error_pct field
        idx = bisect.bisect_left(_MASTER_DIAMETERS, dia)
        nearest_err = float('inf')
        for i in (idx - 1, idx):
            if 0 <= i < len(MASTER):
                e = abs(MASTER[i].diameter_in - dia) / MASTER[i].diameter_in
                if e < nearest_err:
                    nearest_err = e

        note = "non-standard"
        if radius_in < PRACTICAL_MIN_RADIUS_IN:
            note = "sub-standard (micro/EDM range)"
        elif radius_in < MIN_STANDARD_RADIUS_IN:
            note = "below ASME B94.11M minimum"

        return dict(
            matched=False,
            label="non-standard",
            standard="", category="", bolt_size="", fit="",
            snap_radius=radius_in, raw_radius=radius_in,
            error_pct=round(nearest_err * 100, 2),
            diameter_in=round(dia, 6),
            diameter_mm=round(dia * 25.4, 4),
            note=note,
        )

    return dict(
        matched=True,
        label=entry.label,
        standard=entry.standard,
        category=entry.category,
        bolt_size=entry.bolt_size,
        fit=entry.fit,
        snap_radius=entry.radius_in,
        raw_radius=radius_in,
        error_pct=round(abs(entry.diameter_in - dia) / entry.diameter_in * 100, 2),
        diameter_in=round(entry.diameter_in, 6),
        diameter_mm=round(entry.diameter_in * 25.4, 4),
        note=f"snapped {dia:.4f}\" -> {entry.diameter_in:.4f}\" ({entry.label})",
    )


def is_standard_size(radius_in: float, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """Return True if *radius_in* matches an ASME standard within *tolerance*."""
    return snap_to_standard(radius_in, tolerance) is not None


def filter_standard_holes(
    holes: list,
    radius_key: str = 'radius',
    tolerance: float = DEFAULT_TOLERANCE,
    min_radius: float = PRACTICAL_MIN_RADIUS_IN,
) -> list:
    """Filter a list of hole dicts, keeping only ASME-standard sizes.

    Each dict in *holes* must have a key *radius_key* with the radius in inches.
    Returns a new list where every hole has been snapped and annotated with
    ASME classification fields.  Holes below *min_radius* or outside *tolerance*
    are dropped.
    """
    result = []
    for h in holes:
        r = h.get(radius_key, 0.0)
        if r < min_radius:
            continue
        info = classify_hole(r, tolerance)
        if not info['matched']:
            continue
        enriched = dict(h)
        enriched.update({
            'asme_label':    info['label'],
            'asme_standard': info['standard'],
            'asme_category': info['category'],
            'asme_bolt_size':info['bolt_size'],
            'asme_fit':      info['fit'],
            'snap_radius':   info['snap_radius'],
            'diameter_in':   info['diameter_in'],
            'diameter_mm':   info['diameter_mm'],
            'snap_error_pct':info['error_pct'],
        })
        result.append(enriched)
    return result


# ---------------------------------------------------------------------------
# Training log analysis helper
# ---------------------------------------------------------------------------

def analyse_log_radii(radii: list[float], tolerance: float = DEFAULT_TOLERANCE) -> None:
    """Pretty-print ASME (inch) classification for a list of detected radii."""
    unique = sorted(set(round(r, 6) for r in radii))
    print(f"{'Radius (in)':>12}  {'Dia (in)':>9}  {'Dia (mm)':>9}  "
          f"{'Match':>6}  {'Err%':>5}  Label")
    print("-" * 90)
    for r in unique:
        info = classify_hole(r, tolerance)
        tick  = "✓" if info['matched'] else "✗"
        label = info['label'] if info['matched'] else f"non-standard (nearest ±{info['error_pct']:.1f}%)"
        print(f"{r:12.5f}  {info['diameter_in']:9.5f}  {info['diameter_mm']:9.4f}  "
              f"{tick:>6}  {info['error_pct']:5.1f}  {label}")


# ===========================================================================
#  ISO METRIC STANDARDS
# ===========================================================================

# ---------------------------------------------------------------------------
# ISO 273 — Metric clearance hole diameters (mm)
# Format: bolt_size -> (fine/close, medium/normal, coarse/loose)
# ---------------------------------------------------------------------------
ISO_273_CLEARANCE: Dict[str, Tuple[float, float, float]] = {
    "M1.6": (1.7,  1.8,  2.0),
    "M2":   (2.2,  2.4,  2.6),
    "M2.5": (2.7,  2.9,  3.1),
    "M3":   (3.2,  3.4,  3.6),
    "M4":   (4.3,  4.5,  4.8),
    "M5":   (5.3,  5.5,  5.8),
    "M6":   (6.4,  6.6,  7.0),
    "M8":   (8.4,  9.0,  10.0),
    "M10":  (10.5, 11.0, 12.0),
    "M12":  (13.0, 13.5, 14.5),
    "M14":  (15.0, 15.5, 16.5),
    "M16":  (17.0, 17.5, 18.5),
    "M18":  (19.0, 20.0, 21.0),
    "M20":  (21.0, 22.0, 24.0),
    "M22":  (23.0, 24.0, 26.0),
    "M24":  (25.0, 26.0, 28.0),
    "M27":  (28.0, 30.0, 32.0),
    "M30":  (31.0, 33.0, 35.0),
    "M33":  (34.0, 36.0, 38.0),
    "M36":  (37.0, 39.0, 42.0),
    "M39":  (40.0, 42.0, 45.0),
    "M42":  (43.0, 45.0, 48.0),
    "M45":  (46.0, 48.0, 52.0),
    "M48":  (50.0, 52.0, 56.0),
    "M52":  (54.0, 56.0, 62.0),
    "M56":  (58.0, 62.0, 66.0),
    "M60":  (62.0, 66.0, 70.0),
    "M64":  (66.0, 70.0, 74.0),
}

# ---------------------------------------------------------------------------
# ISO 965-1 — Coarse pitch tap drills (mm)
# Formula: tap_drill = nominal - pitch
# ---------------------------------------------------------------------------
ISO_TAP_DRILL_COARSE: Dict[str, float] = {
    "M1":    0.75,  "M1.2":  0.95,  "M1.4":  1.10,  "M1.6":  1.25,
    "M1.8":  1.45,  "M2":    1.60,  "M2.5":  2.05,  "M3":    2.50,
    "M3.5":  2.90,  "M4":    3.30,  "M5":    4.20,  "M6":    5.00,
    "M7":    6.00,  "M8":    6.80,  "M10":   8.50,  "M12":   10.20,
    "M14":   12.00, "M16":   14.00, "M18":   15.50, "M20":   17.50,
    "M22":   19.50, "M24":   21.00, "M27":   24.00, "M30":   26.50,
    "M33":   29.50, "M36":   32.00, "M39":   35.00, "M42":   37.50,
    "M45":   40.50, "M48":   43.00, "M52":   47.00, "M56":   50.50,
    "M60":   54.50, "M64":   58.00,
}

# ---------------------------------------------------------------------------
# ISO 965-1 — Fine pitch tap drills (mm)
# Format: "MdxP" -> tap drill diameter
# ---------------------------------------------------------------------------
ISO_TAP_DRILL_FINE: Dict[str, float] = {
    "M8x1":     7.00,  "M10x1.25": 8.75,  "M10x1":    9.00,
    "M12x1.5":  10.50, "M12x1.25": 10.75, "M14x1.5":  12.50,
    "M16x1.5":  14.50, "M18x1.5":  16.50, "M20x1.5":  18.50,
    "M22x1.5":  20.50, "M24x2":    22.00, "M27x2":    25.00,
    "M30x2":    28.00, "M33x2":    31.00, "M36x3":    33.00,
    "M39x3":    36.00, "M42x3":    39.00, "M45x3":    42.00,
    "M48x3":    45.00, "M52x4":    48.00, "M56x4":    52.00,
    "M60x4":    56.00, "M64x4":    60.00,
}

# ---------------------------------------------------------------------------
# DIN 338 / ISO 2306 — Standard metric drill bit diameters (mm)
# Covers 0.5 mm increments up to 13 mm, 0.5 mm steps 13–20, 1 mm steps 20–80
# ---------------------------------------------------------------------------
ISO_METRIC_DRILLS: List[float] = [
    # 0.5–1.0 mm (0.1 steps)
    0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
    # 1.0–3.0 mm (0.1 steps)
    1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00,
    2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00,
    # 3.0–13.0 mm (0.1 steps)
    3.10, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90, 4.00,
    4.10, 4.20, 4.30, 4.40, 4.50, 4.60, 4.70, 4.80, 4.90, 5.00,
    5.10, 5.20, 5.30, 5.40, 5.50, 5.60, 5.70, 5.80, 5.90, 6.00,
    6.10, 6.20, 6.30, 6.40, 6.50, 6.60, 6.70, 6.80, 6.90, 7.00,
    7.10, 7.20, 7.30, 7.40, 7.50, 7.60, 7.70, 7.80, 7.90, 8.00,
    8.10, 8.20, 8.30, 8.40, 8.50, 8.60, 8.70, 8.80, 8.90, 9.00,
    9.10, 9.20, 9.30, 9.40, 9.50, 9.60, 9.70, 9.80, 9.90, 10.00,
    10.20, 10.50, 11.00, 11.50, 12.00, 12.50, 13.00,
    # 13–20 mm (0.5 steps)
    13.50, 14.00, 14.50, 15.00, 15.50, 16.00, 16.50, 17.00, 17.50,
    18.00, 18.50, 19.00, 19.50, 20.00,
    # 20–80 mm (1.0 steps + key sizes)
    21.00, 22.00, 23.00, 24.00, 25.00, 26.00, 27.00, 28.00, 29.00,
    30.00, 31.00, 32.00, 33.00, 34.00, 35.00, 36.00, 37.00, 38.00,
    39.00, 40.00, 42.00, 44.00, 45.00, 46.00, 48.00, 50.00, 52.00,
    54.00, 56.00, 58.00, 60.00, 62.00, 65.00, 68.00, 70.00, 75.00,
    80.00,
]

# ---------------------------------------------------------------------------
# Practical range limits (mm, radius)
# ---------------------------------------------------------------------------
#: Smallest practical metric drill (DIN 338 starts at 0.5 mm dia → r=0.25 mm)
MIN_STANDARD_RADIUS_MM: float = 0.25

#: Largest ISO 273 clearance hole (M64 coarse = 74 mm dia → r=37 mm)
MAX_STANDARD_RADIUS_MM: float = 37.0

#: Below this radius the hole is considered sub-standard for general machining
PRACTICAL_MIN_RADIUS_MM: float = 0.50   # diameter 1.0 mm


# ---------------------------------------------------------------------------
# ISO metric master lookup table — built once at import time
# ---------------------------------------------------------------------------

def _build_master_mm() -> List[StandardEntry]:
    entries: List[StandardEntry] = []

    # ISO 273 clearance holes
    fit_names = ("fine", "medium", "coarse")
    for bolt, diameters in ISO_273_CLEARANCE.items():
        for fit, dia in zip(fit_names, diameters):
            entries.append(StandardEntry(
                diameter_in=dia,   # re-using field for mm value
                radius_in=dia / 2,
                label=f'{bolt} {fit} clearance (ISO 273)',
                standard="ISO 273",
                category="clearance",
                bolt_size=bolt, fit=fit,
            ))

    # ISO 965-1 coarse tap drills
    for thread, dia in ISO_TAP_DRILL_COARSE.items():
        entries.append(StandardEntry(
            diameter_in=dia, radius_in=dia / 2,
            label=f'{thread} coarse tap drill (ISO 965-1)',
            standard="ISO 965-1",
            category="tap_drill",
            bolt_size=thread, fit="coarse",
        ))

    # ISO 965-1 fine tap drills
    for thread, dia in ISO_TAP_DRILL_FINE.items():
        entries.append(StandardEntry(
            diameter_in=dia, radius_in=dia / 2,
            label=f'{thread} fine tap drill (ISO 965-1)',
            standard="ISO 965-1",
            category="tap_drill",
            bolt_size=thread, fit="fine",
        ))

    # DIN 338 metric drills
    for dia in ISO_METRIC_DRILLS:
        entries.append(StandardEntry(
            diameter_in=dia, radius_in=dia / 2,
            label=f'{dia:.2f} mm drill (DIN 338)',
            standard="DIN 338",
            category="metric_drill",
            bolt_size=f"{dia:.2f}mm", fit="",
        ))

    entries.sort(key=lambda e: e.diameter_in)
    return entries


#: All ISO metric standard entries, sorted by diameter ascending (values in mm).
MASTER_MM: List[StandardEntry] = _build_master_mm()
_MASTER_DIAMETERS_MM: List[float] = [e.diameter_in for e in MASTER_MM]


# ---------------------------------------------------------------------------
# ISO metric public API
# ---------------------------------------------------------------------------

def snap_to_standard_mm(
    radius_mm: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Optional[StandardEntry]:
    """Return the closest ISO metric standard entry for a detected radius (mm)."""
    dia = radius_mm * 2.0
    idx = bisect.bisect_left(_MASTER_DIAMETERS_MM, dia)

    best: Optional[StandardEntry] = None
    best_err = float('inf')

    for i in (idx - 1, idx):
        if 0 <= i < len(MASTER_MM):
            err = abs(MASTER_MM[i].diameter_in - dia) / MASTER_MM[i].diameter_in
            if err < best_err:
                best_err = err
                best = MASTER_MM[i]

    if best is None or best_err > tolerance:
        return None
    return best


def classify_hole_mm(
    radius_mm: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Classify a detected hole radius (mm) against ISO metric standards.

    Returns the same dict shape as :func:`classify_hole` but with mm units.
    Keys ``diameter_in`` / ``diameter_mm`` both carry the mm value for
    consistency with the training pipeline.
    """
    entry = snap_to_standard_mm(radius_mm, tolerance)
    dia   = radius_mm * 2.0

    if entry is None:
        idx = bisect.bisect_left(_MASTER_DIAMETERS_MM, dia)
        nearest_err = float('inf')
        for i in (idx - 1, idx):
            if 0 <= i < len(MASTER_MM):
                e = abs(MASTER_MM[i].diameter_in - dia) / MASTER_MM[i].diameter_in
                if e < nearest_err:
                    nearest_err = e

        note = "non-standard"
        if radius_mm < PRACTICAL_MIN_RADIUS_MM:
            note = "sub-standard (micro/EDM range)"
        elif radius_mm < MIN_STANDARD_RADIUS_MM:
            note = "below DIN 338 minimum"

        return dict(
            matched=False,
            label="non-standard",
            standard="", category="", bolt_size="", fit="",
            snap_radius=radius_mm, raw_radius=radius_mm,
            error_pct=round(nearest_err * 100, 2),
            diameter_in=round(dia, 4),
            diameter_mm=round(dia, 4),
            note=note,
        )

    return dict(
        matched=True,
        label=entry.label,
        standard=entry.standard,
        category=entry.category,
        bolt_size=entry.bolt_size,
        fit=entry.fit,
        snap_radius=entry.radius_in,
        raw_radius=radius_mm,
        error_pct=round(abs(entry.diameter_in - dia) / entry.diameter_in * 100, 2),
        diameter_in=round(entry.diameter_in, 4),
        diameter_mm=round(entry.diameter_in, 4),
        note=f"snapped {dia:.4f}mm -> {entry.diameter_in:.4f}mm ({entry.label})",
    )


def is_standard_size_mm(radius_mm: float, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """Return True if *radius_mm* matches an ISO metric standard within *tolerance*."""
    return snap_to_standard_mm(radius_mm, tolerance) is not None


def analyse_log_radii_mm(radii: list, tolerance: float = DEFAULT_TOLERANCE) -> None:
    """Pretty-print ISO metric classification for a list of detected radii (mm)."""
    unique = sorted(set(round(r, 4) for r in radii))
    print(f"{'Radius (mm)':>12}  {'Dia (mm)':>9}  {'Match':>6}  {'Err%':>5}  Label")
    print("-" * 85)
    for r in unique:
        info = classify_hole_mm(r, tolerance)
        tick  = "✓" if info['matched'] else "✗"
        label = info['label'] if info['matched'] else f"non-standard (nearest ±{info['error_pct']:.1f}%)"
        print(f"{r:12.4f}  {info['diameter_mm']:9.4f}  "
              f"{tick:>6}  {info['error_pct']:5.1f}  {label}")
