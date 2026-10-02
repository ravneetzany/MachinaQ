# MachinaQ GNN — Training Help Guide

## Environment

| Item | Value |
|---|---|
| Conda install | `E:\ProgramFilez\Anaconda` |
| Conda env | `pyoccMachinaQ` |
| Python | 3.10.9 (`E:\AiTools\python3.10\python.exe`) |
| PyTorch | 2.3.1+cu121 |
| DGL | 2.2.1 |
| GPU | NVIDIA GeForce RTX 2060 SUPER (8 GB VRAM) |
| Dataset | `machgnn\dataset\MFInstSeg\aag\graphs.json` (~20 GB, 14 shards) |

---

## How to Run

### Option 1 — One-liner (cmd)
```
call E:\ProgramFilez\Anaconda\Scripts\activate.bat pyoccMachinaQ && set DGLBACKEND=pytorch && cd E:\AiTools\AIModel_StepAnalyze && python run_train.py
```

### Option 2 — Step by step (cmd)
```
call E:\ProgramFilez\Anaconda\Scripts\activate.bat pyoccMachinaQ
set DGLBACKEND=pytorch
cd E:\AiTools\AIModel_StepAnalyze
python run_train.py
```

> **Note:** Always run from `E:\AiTools\AIModel_StepAnalyze` — the dataset loader
> resolves `MFInstseg_partition/train.txt` relative to the working directory.

---

## What to Expect on Startup

Dataset loading is the slow part. Wait **20–60 minutes** before Epoch 1 begins.

```
MachinaQ GNN Training  |  25 classes  |  dataset: ...
Loading train data...
Streaming graphs.json — loading 14 items...
Train: X,XXX  |  Val: X,XXX
Device: cuda
Parameters: X,XXX,XXX
Starting training: 100 epochs  batch=64  lr=0.01
```

---

## Output Files

| File | Purpose |
|---|---|
| `outputs\machinaq_train.log` | Per-epoch metrics (loss, IoU, F1) |
| `outputs\train_stderr.log` | Errors / tracebacks (when launched via script) |
| `outputs\machinaq_gnn.pth` | Best model weights — saved when val score improves (★) |

### Sample log line (per epoch)
```
Ep   1/100  312s  loss=1.2341/1.1892  seg_iou=0.412/0.389  inst_f1=0.731/0.698  bot_iou=0.654/0.621 ★ saved
```

---

## Resuming Interrupted Training

The `pointnet`, `through-hole`, and `unified` trainers save a resumable
checkpoint (model + optimizer + scheduler state + epoch number) to
`outputs/checkpoints/<model>_ckpt.pth` after **every** epoch — not just at
the end. If training is interrupted (crash, reboot, Ctrl+C), just re-run the
exact same `python run_train.py --model <model> ...` command: it detects the
checkpoint and continues from the next epoch instead of starting over.
`operation-classifier` already saved its best checkpoint on every improving
epoch and needs no separate resume step.

`train_all_models.sh` (repo root) runs all four models — pointnet,
through-hole, unified (`--merge-weights`), operation-classifier — in order,
skipping any whose final `outputs/machinaq_<model>.pth` already exists and
resuming any in-progress one from its checkpoint. Safe to re-run after any
interruption:

```bash
./train_all_models.sh
```

Note: DataLoader multi-worker loading (`num_workers>0`) is intentionally
disabled on Windows in these trainers, since `run_train.py` has no
`if __name__ == '__main__':` guard and Windows' spawn-based multiprocessing
would otherwise re-execute the whole script in each worker process. Windows
runs are single-process; Linux/macOS use up to 4 workers.

## Installed Dependencies

| Package | Location |
|---|---|
| `torch` 2.3.1+cu121 | `E:\AiTools\python3.10\Lib\site-packages\torch` |
| `torchvision` 0.18.1+cu121 | `E:\AiTools\python3.10\Lib\site-packages\torchvision` |
| `dgl` 2.2.1 | `E:\AiTools\python3.10\Lib\site-packages\dgl` |
| `torch_ema` | `E:\AiTools\python3.10\Lib\site-packages\torch_ema` |
| `torchmetrics` | `E:\AiTools\python3.10\Lib\site-packages\torchmetrics` |
| `timm` | `E:\AiTools\python3.10\Lib\site-packages\timm` |
| `wandb` | `C:\Users\PC\AppData\Roaming\Python\Python310\site-packages` |
| `huggingface_hub` 0.36.x | user site-packages |
| `safetensors` | user site-packages |

---

## Known Fixes Applied

| Issue | Fix |
|---|---|
| `torch` not found in `python3.10` | Installed `torch==2.3.1+cu121` via PyTorch whl index |
| `pip` upgrade failed (`t64.exe` missing) | Reinstalled pip via `get-pip.py --force-reinstall` |
| `ModuleNotFoundError: huggingface_hub` | Installed `huggingface_hub>=0.23.0,<1.0` (pinned for `transformers 4.41.2`) |
| DGL graphbolt `ImportError` (torchdata 0.11) | Patched `graphbolt/__init__.py` to catch `ImportError` |
| DGL graphbolt DLL mismatch (PyTorch 2.3.1) | Copied `graphbolt_pytorch_2.3.0.dll` → `graphbolt_pytorch_2.3.1.dll` |
| `pythonocc` import error | Wrapped OCC import in `try/except` in `machgnn/utils/data_utils.py` |
| Dataset partition path not found | Added `os.chdir(MACHGNN)` in `run_train.py` before dataset load |

---

## Training Config (run_train.py)

| Setting | Value |
|---|---|
| Epochs | 100 |
| Batch size | 64 |
| Learning rate | 0.01 (CosineAnnealing decay) |
| Weight decay | 0.01 |
| Classes | 25 (MFInstSeg full) |
| Loss weights | seg=1.0, inst=1.0, bot=1.0 |
| Pretrained weights | `machgnn\weights\weight_on_MFInstseg.pth` (loaded if present) |
| Save criterion | best `seg_iou + inst_f1 + bot_iou` on validation set |

---

## Troubleshooting

**`CondaError: Run 'conda init' before 'conda activate'`**
→ Use `call ...activate.bat pyoccMachinaQ` instead of `conda activate`.

**`ModuleNotFoundError: No module named 'torch'`**
→ Make sure you activated `pyoccMachinaQ` and run with the correct Python.
→ Verify: `python -c "import torch; print(torch.__version__)"`

**`CUDA not available` / training on CPU**
→ Check: `python -c "import torch; print(torch.cuda.is_available())"`
→ Must show `True`. If not, reinstall `torch==2.3.1+cu121`.

**Dataset load hangs or crashes**
→ Normal to take 20–60 min for 20 GB `graphs.json`.
→ Watch RAM — dataset load can use 16–32 GB RAM.
→ Any crash during load will print a traceback to the terminal.

**OOM (Out of Memory) on GPU**
→ Reduce `BATCH_SIZE` in `run_train.py` (line 29). Try 32 or 16.
