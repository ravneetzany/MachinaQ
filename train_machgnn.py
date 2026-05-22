"""Train MachinaQ GNN for per-face machining feature recognition.

MachinaQ GNN is a Graph Transformer on Attributed Adjacency Graphs (gAAG) built
from B-rep STEP files. It outputs per-face semantic labels (24 classes: slot,
hole, chamfer, pocket …) plus instance and bottom-face segmentation.

Prerequisites
-------------
1. Run setup_machgnn.bat  (clones AAGNet, creates pyocc conda env)
2. Download MFInstSeg to dataset/MFInstSeg/
   Google Drive: https://drive.google.com/file/d/1T2sHlL-4qlsXTxu3AMBB4h-N1I5VqNky/view

Usage
-----
    conda run -n pyocc python train_machgnn.py
    conda run -n pyocc python train_machgnn.py --epochs 50 --batch_size 16
    conda run -n pyocc python train_machgnn.py --nist_only   # smoke-test, no download needed
"""

import argparse
import re
import shutil
import subprocess
import sys
import torch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.resolve()
MACHGNN_DIR  = ROOT / "machgnn"
DATASET_DIR = MACHGNN_DIR / "dataset" / "MFInstSeg"
NIST_DIR    = ROOT / "nist_sfa"
MODELS_DIR  = ROOT / "models"

# Exact string from machgnn/engine/inst_trainer.py line 60
_AAGNET_DATASET_KEY = '"dataset": "../traning_data/data2"'
_AAGNET_DEVICE_KEY  = '"device": \'cuda\''

MFINSTSEG_GDRIVE = (
    "https://drive.google.com/file/d/1T2sHlL-4qlsXTxu3AMBB4h-N1I5VqNky/view"
)


# ── Prerequisite checks ───────────────────────────────────────────────────────
def check_prerequisites(nist_only: bool) -> None:
    errors = []

    if not MACHGNN_DIR.exists():
        errors.append(
            f"AAGNet not cloned at {MACHGNN_DIR}.\n"
            "  Run:  setup_machgnn.bat"
        )

    trainer = MACHGNN_DIR / "engine" / "inst_trainer.py"
    if MACHGNN_DIR.exists() and not trainer.exists():
        errors.append(
            f"inst_trainer.py missing at {trainer}.\n"
            "  The clone may be incomplete — delete machgnn/ and re-run setup_machgnn.bat."
        )

    if not nist_only:
        if not (DATASET_DIR / "aag").exists():
            errors.append(
                f"MFInstSeg dataset not found at {DATASET_DIR}.\n"
                f"  Download: {MFINSTSEG_GDRIVE}\n"
                "  Extract so you have:\n"
                "    dataset/MFInstSeg/aag/\n"
                "    dataset/MFInstSeg/labels/\n"
                "    dataset/MFInstSeg/steps/"
            )

    if errors:
        print("\n[ERROR] Prerequisites not satisfied:\n")
        for e in errors:
            print(f"  • {e}\n")
        sys.exit(1)

    print("[OK] All prerequisites satisfied.")


# ── NIST smoke-test dataset via AAGExtractor ──────────────────────────────────
def prepare_nist_dataset() -> Path:
    """Build a minimal MFInstSeg-layout dataset from the 17 local NIST STEP files.

    Uses machgnn/dataset/AAGExtractor.py (pythonocc) to generate gAAG graphs.
    The result is too small for real training but enough to verify the pipeline.
    """
    nist_out = ROOT / "dataset" / "MFInstSeg_nist"
    steps_out = nist_out / "steps"
    aag_out   = nist_out / "aag"
    (nist_out / "labels").mkdir(parents=True, exist_ok=True)
    steps_out.mkdir(parents=True, exist_ok=True)
    aag_out.mkdir(parents=True, exist_ok=True)

    step_files = (
        list(NIST_DIR.glob("*.stp"))
        + list((NIST_DIR / "holeTrain").glob("*.step"))
    )
    if not step_files:
        print("[ERROR] No STEP files found in nist_sfa/.")
        sys.exit(1)

    for f in step_files:
        shutil.copy2(f, steps_out / f.name)
    print(f"[INFO] Copied {len(step_files)} STEP files to {steps_out}")

    aag_extractor = MACHGNN_DIR / "dataset" / "AAGExtractor.py"
    if not aag_extractor.exists():
        print(f"[WARN] AAGExtractor.py not found at {aag_extractor} — skipping graph generation.")
        print("       Training will fail without .bin graph files.")
    else:
        print(f"[INFO] Running AAGExtractor on {len(step_files)} NIST files...")
        subprocess.run(
            [sys.executable, str(aag_extractor),
             "--step_path", str(steps_out),
             "--output",    str(aag_out),
             "--num_workers", "4"],
            cwd=str(MACHGNN_DIR),
            check=True,
        )
        print(f"[OK] gAAG graphs written to {aag_out}")

    # Minimal train/val/test split files required by MFInstSegDataset
    _write_nist_split(nist_out, step_files)

    print(f"[OK] NIST smoke-test dataset at {nist_out}")
    return nist_out


def _write_nist_split(nist_out: Path, step_files: list) -> None:
    stems = [f.stem for f in step_files]
    n = len(stems)
    train_end = max(1, int(n * 0.7))
    val_end   = max(train_end + 1, int(n * 0.85))

    split_dir = MACHGNN_DIR / "MFInstseg_partition"
    split_dir.mkdir(exist_ok=True)

    for fname, items in [
        ("train.txt", stems[:train_end]),
        ("val.txt",   stems[train_end:val_end]),
        ("test.txt",  stems[val_end:]),
    ]:
        (split_dir / fname).write_text("\n".join(items) + "\n", encoding="utf-8")

    print(f"[OK] Split files written to {split_dir} "
          f"(train={train_end}, val={val_end - train_end}, test={n - val_end})")


# ── Config patching ───────────────────────────────────────────────────────────
def build_patched_trainer(dataset_path: Path, epochs: int, batch_size: int) -> Path:
    """Write inst_trainer_patched.py with dataset path, device, epochs, batch injected.

    Patches the copy rather than the original so machgnn/ stays clean.
    """
    src = (MACHGNN_DIR / "engine" / "inst_trainer.py").read_text(encoding="utf-8")

    data_str = str(dataset_path).replace("\\", "/")
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import dgl
        if not dgl.backend.cuda_available():
            device = "cpu"
    except ImportError:
        device = "cpu"

    # Patch dataset path (exact match on known line 60)
    src = src.replace(
        _AAGNET_DATASET_KEY,
        f'"dataset": r"{data_str}"',
    )

    # Patch device
    src = src.replace(_AAGNET_DEVICE_KEY, f'"device": "{device}"')
    src = re.sub(r'"device":\s*\'(cuda|cpu)\'', f'"device": "{device}"', src)

    # Patch epochs (inside wandb config dict)
    src = re.sub(r'("epochs":\s*)\d+', rf'\g<1>{epochs}', src)

    # Patch batch size (inside wandb config dict)
    src = re.sub(r'("batch_size":\s*)\d+', rf'\g<1>{batch_size}', src)

    # Disable wandb upload (already offline mode, but ensure no stray API key warning)
    src = src.replace(
        'os.environ["WANDB_MODE"] = "offline"',
        'os.environ["WANDB_MODE"] = "disabled"',
    )

    out = MACHGNN_DIR / "engine" / "inst_trainer_patched.py"
    out.write_text(src, encoding="utf-8")
    return out


# ── Copy best weights into models/ ───────────────────────────────────────────
def copy_best_weights() -> None:
    output_root = MACHGNN_DIR / "output"
    if not output_root.exists():
        return
    runs = sorted(
        [p for p in output_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        return
    for candidate in ("best.pth", "best_model.pth", "checkpoint_best.pth"):
        ckpt = runs[-1] / candidate
        if ckpt.exists():
            MODELS_DIR.mkdir(exist_ok=True)
            dest = MODELS_DIR / "machinaq_gnn_best.pth"
            shutil.copy2(ckpt, dest)
            print(f"\n[OK] Best weights → {dest}")
            return
    print(f"[INFO] Weights in {runs[-1]}/  (no standard best.pth name found)")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_training(args: argparse.Namespace) -> None:
    check_prerequisites(args.nist_only)

    dataset_path = prepare_nist_dataset() if args.nist_only else DATASET_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print("  MachinaQ GNN Training")
    print(f"{'='*60}")
    print(f"  Dataset   : {dataset_path}")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch_size}")
    print(f"{'='*60}\n")

    patched = build_patched_trainer(dataset_path, args.epochs, args.batch_size)
    print(f"[INFO] Launching: python -m engine.inst_trainer_patched\n")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "engine.inst_trainer_patched"],
            cwd=str(MACHGNN_DIR),
            check=False,
        )
        if result.returncode != 0:
            print(f"\n[ERROR] Training exited with code {result.returncode}")
            sys.exit(result.returncode)
    finally:
        if patched.exists():
            patched.unlink()

    copy_best_weights()
    print(f"\n[DONE] MachinaQ GNN training complete.")
    print(f"       Artifacts : {MACHGNN_DIR / 'output'}")
    print(f"       Best model: {MODELS_DIR / 'machinaq_gnn_best.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MachinaQ GNN on MFInstSeg (or NIST smoke-test)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="32 for ≤8 GB VRAM; 128–256 for 16 GB+")
    parser.add_argument("--nist_only",  action="store_true",
                        help="Smoke-test on 17 local NIST files; no dataset download needed")
    run_training(parser.parse_args())
