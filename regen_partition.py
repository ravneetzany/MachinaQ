"""
Regenerate MFInstseg_partition/{train,val,test}.txt from actual label filenames.
Split: 80% train / 10% val / 10% test  (fixed seed for reproducibility)
"""
import os, random, pathlib

LABELS_DIR   = pathlib.Path(r"E:\AiTools\AIModel_StepAnalyze\machgnn\dataset\MFInstSeg\labels")
PARTITION_DIR = pathlib.Path(r"E:\AiTools\AIModel_StepAnalyze\machgnn\MFInstseg_partition")

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = remainder

SEED = 42

# ── Collect all IDs ────────────────────────────────────────────────────────────
ids = sorted(p.stem for p in LABELS_DIR.glob("*.json"))
print(f"Total label files found: {len(ids):,}")

# ── Shuffle deterministically ──────────────────────────────────────────────────
rng = random.Random(SEED)
rng.shuffle(ids)

# ── Split ──────────────────────────────────────────────────────────────────────
n       = len(ids)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

train = ids[:n_train]
val   = ids[n_train:n_train + n_val]
test  = ids[n_train + n_val:]

print(f"Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")

# ── Write partition files ──────────────────────────────────────────────────────
PARTITION_DIR.mkdir(parents=True, exist_ok=True)

for name, split in [("train", train), ("val", val), ("test", test)]:
    out = PARTITION_DIR / f"{name}.txt"
    with open(out, "w") as f:
        f.write("\n".join(split) + "\n")
    print(f"Written: {out}")

print("Done.")
