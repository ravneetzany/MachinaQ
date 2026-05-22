"""MachinaQ Training — GNN (MFInstSeg 25-class) or PointNet (NIST + HoleData)."""

import os, sys, argparse, time, logging

ROOT    = os.path.dirname(os.path.abspath(__file__))
MACHGNN = os.path.join(ROOT, 'machgnn')
sys.path.insert(0, ROOT)
sys.path.insert(0, MACHGNN)

# ── CLI args ───────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(
    description='MachinaQ training launcher',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
ap.add_argument(
    '--model',
    choices=['gnn', 'pointnet', 'through-hole', 'unified'],
    default='gnn',
    help=(
        'Model to train: '
        'gnn = AAGNet GNN on MFInstSeg 25-class dataset; '
        'pointnet = PointNet on NIST .stp + nist_sfa/holeTrain .step files; '
        'through-hole = binary PointNet classifier (through vs blind holes); '
        'unified = MachinaQUnified multi-task model (feature type + through/blind) '
        '          combining all three architectures into one system'
    ),
)
ap.add_argument('--epochs',     type=int,   default=None, help='Override epoch count')
ap.add_argument('--batch-size', type=int,   default=None, help='Override batch size')
ap.add_argument('--lr',         type=float, default=None, help='Override learning rate')
ap.add_argument('--aug',        type=int,   default=15,
                help='[pointnet / through-hole / unified] augmentation factor per hole sample')
ap.add_argument(
    '--merge-weights', action='store_true',
    help='[unified only] initialise MachinaQUnified from pre-trained PointNet + '
         'through-hole weights before fine-tuning (outputs/machinaq_pointnet.pth '
         'and outputs/machinaq_through_hole.pth must exist)',
)
args = ap.parse_args()

os.makedirs(os.path.join(ROOT, 'outputs'), exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_NAME = {
    'pointnet':    'pointnet_train.log',
    'through-hole':'through_hole_train.log',
    'unified':     'unified_train.log',
}.get(args.model, 'machinaq_train.log')
LOG_PATH = os.path.join(ROOT, 'outputs', LOG_NAME)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger('MachinaQ')

# ==============================================================================
#  PointNet branch
# ==============================================================================
if args.model == 'pointnet':
    from pathlib import Path
    from src.train_enhanced import train_enhanced_pointnet, save_trained_model

    NIST_DIR  = Path(ROOT) / 'nist_sfa'
    HOLE_DIR  = NIST_DIR / 'holeTrain'
    SAVE_PATH = os.path.join(ROOT, 'outputs', 'machinaq_pointnet.pth')

    EPOCHS     = args.epochs     or 20
    BATCH_SIZE = args.batch_size or 16
    LR         = args.lr         or 1e-3
    AUG        = args.aug

    # Collect NIST .stp files + holeTrain .step files
    nist_files = sorted(NIST_DIR.glob('nist_*.stp'))
    hole_files = sorted(HOLE_DIR.glob('*.step'))
    all_files  = [str(f) for f in nist_files + hole_files]

    log.info('=' * 70)
    log.info('MachinaQ PointNet Training')
    log.info(f'  NIST .stp files  : {len(nist_files)} ({NIST_DIR})')
    log.info(f'  HoleData files   : {len(hole_files)} ({HOLE_DIR})')
    log.info(f'  Total files      : {len(all_files)}')
    log.info(f'  Epochs           : {EPOCHS}')
    log.info(f'  Batch size       : {BATCH_SIZE}')
    log.info(f'  Learning rate    : {LR}')
    log.info(f'  Augmentation     : {AUG}x per hole sample')
    log.info(f'  Save path        : {SAVE_PATH}')
    log.info(f'  Log              : {LOG_PATH}')
    log.info('=' * 70)

    if not all_files:
        log.error('No STEP/STP files found. Check nist_sfa/ and nist_sfa/holeTrain/')
        sys.exit(1)

    model = train_enhanced_pointnet(
        all_files,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        augmentation_factor=AUG,
    )

    if model:
        save_trained_model(model, SAVE_PATH)
        log.info(f'PointNet training complete  ->  {SAVE_PATH}')
    else:
        log.error('Training produced no model — no holes were parsed from any file.')
        sys.exit(1)

    sys.exit(0)

# ==============================================================================
#  Through-hole binary classifier branch
# ==============================================================================
if args.model == 'through-hole':
    from pathlib import Path as _Path
    from src.train_through_hole import train_through_hole_classifier
    from models.pointnet import save_model

    NIST_DIR  = _Path(ROOT) / 'nist_sfa'
    SAVE_PATH = os.path.join(ROOT, 'outputs', 'machinaq_through_hole.pth')

    EPOCHS     = args.epochs     or 30
    BATCH_SIZE = args.batch_size or 32
    LR         = args.lr         or 1e-3
    AUG        = args.aug        or 20

    # STC (Sheet Metal Test Cases) = all through holes by definition
    # FTC (Flat Test Cases)        = also mostly through holes
    stc_files = sorted(NIST_DIR.glob('nist_stc_*.stp'))
    ftc_files = sorted(NIST_DIR.glob('nist_ftc_*.stp'))
    all_files = [str(f) for f in stc_files + ftc_files]

    log.info('=' * 70)
    log.info('MachinaQ Through-Hole Binary Classifier Training')
    log.info(f'  STC files        : {len(stc_files)}')
    log.info(f'  FTC files        : {len(ftc_files)}')
    log.info(f'  Epochs           : {EPOCHS}')
    log.info(f'  Batch size       : {BATCH_SIZE}')
    log.info(f'  Learning rate    : {LR}')
    log.info(f'  Aug factor       : {AUG}x per hole')
    log.info(f'  Save path        : {SAVE_PATH}')
    log.info(f'  Log              : {LOG_PATH}')
    log.info('=' * 70)

    if not all_files:
        log.error('No STC/FTC STEP files found in nist_sfa/. Cannot train.')
        sys.exit(1)

    model = train_through_hole_classifier(
        all_files,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        aug_factor=AUG,
    )
    save_model(model, SAVE_PATH)
    log.info(f'Through-hole model saved  ->  {SAVE_PATH}')
    sys.exit(0)

# ==============================================================================
#  Unified branch  (MachinaQUnified: feature classification + through/blind)
# ==============================================================================
if args.model == 'unified':
    from pathlib import Path as _Path
    from src.train_unified import train_unified
    from models.machinaq_unified import save_unified

    NIST_DIR  = _Path(ROOT) / 'nist_sfa'
    SAVE_PATH = os.path.join(ROOT, 'outputs', 'machinaq_unified.pth')

    # Pre-trained weight paths for optional --merge-weights initialisation
    POINTNET_PTH    = os.path.join(ROOT, 'outputs', 'machinaq_pointnet.pth')
    THROUGH_HOLE_PTH = os.path.join(ROOT, 'outputs', 'machinaq_through_hole.pth')

    EPOCHS     = args.epochs     or 30
    BATCH_SIZE = args.batch_size or 32
    LR         = args.lr         or 1e-3
    AUG        = args.aug        or 20

    # Use STC + FTC files (all through holes by design) — same as through-hole branch
    stc_files = sorted(NIST_DIR.glob('nist_stc_*.stp'))
    ftc_files = sorted(NIST_DIR.glob('nist_ftc_*.stp'))
    all_files  = [str(f) for f in stc_files + ftc_files]

    # Resolve --merge-weights paths
    pn_path  = POINTNET_PTH    if (args.merge_weights and os.path.exists(POINTNET_PTH))    else None
    th_path  = THROUGH_HOLE_PTH if (args.merge_weights and os.path.exists(THROUGH_HOLE_PTH)) else None

    log.info('=' * 70)
    log.info('MachinaQ Unified Model Training')
    log.info('  Combines: PointNet (feature type) + PointNetBinary (through/blind)')
    log.info('            + AAGNet GNN (loaded at inference via MachinaQPipeline)')
    log.info(f'  STC files        : {len(stc_files)}')
    log.info(f'  FTC files        : {len(ftc_files)}')
    log.info(f'  Epochs           : {EPOCHS}')
    log.info(f'  Batch size       : {BATCH_SIZE}')
    log.info(f'  Learning rate    : {LR}')
    log.info(f'  Aug factor       : {AUG}x per hole')
    log.info(f'  Merge weights    : {args.merge_weights}')
    if pn_path:
        log.info(f'    PointNet src   : {pn_path}')
    if th_path:
        log.info(f'    ThroughHole src: {th_path}')
    log.info(f'  Save path        : {SAVE_PATH}')
    log.info(f'  Log              : {LOG_PATH}')
    log.info('=' * 70)

    if not all_files:
        log.error('No STC/FTC STEP files found in nist_sfa/. Cannot train.')
        sys.exit(1)

    model = train_unified(
        all_files,
        epochs              = EPOCHS,
        batch_size          = BATCH_SIZE,
        lr                  = LR,
        aug_factor          = AUG,
        feat_weight         = 0.5,
        hole_weight         = 1.0,
        pretrained_pointnet = pn_path,
        pretrained_binary   = th_path,
    )

    save_unified(model, SAVE_PATH)
    log.info(f'Unified model saved  ->  {SAVE_PATH}')
    log.info('')
    log.info('To run full inference with all three models:')
    log.info('    from models.machinaq_unified import MachinaQPipeline')
    log.info('    pipeline = MachinaQPipeline.from_pretrained(')
    log.info(f'        unified_path = "{SAVE_PATH}",')
    log.info(f'        gnn_path     = "{os.path.join(ROOT, "outputs", "machinaq_gnn.pth")}",')
    log.info('    )')
    log.info('    result = pipeline.predict_file("part.stp")')
    log.info('    print(result.summary())')
    sys.exit(0)

# ==============================================================================
#  GNN branch (default)
# ==============================================================================
os.environ['DGLBACKEND'] = 'pytorch'

import numpy as np
import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassJaccardIndex,
    BinaryAccuracy, BinaryF1Score, BinaryJaccardIndex,
)
from tqdm import tqdm

from dataloader.mfinstseg import MFInstSegDataset
from models.inst_segmentors import AAGNetSegmentor

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = os.path.join(MACHGNN, 'dataset', 'MFInstSeg')
DATASET_TYPE = 'full'          # 25 classes
EPOCHS       = args.epochs     or 100
BATCH_SIZE   = args.batch_size or 64
LR           = args.lr         or 1e-2
WEIGHT_DECAY = 1e-2
SEG_W = INST_W = BOT_W = 1.0
SAVE_PATH    = os.path.join(ROOT, 'outputs', 'machinaq_gnn.pth')

# MFInstseg_partition paths are looked up relative to CWD in mfinstseg.py
os.chdir(MACHGNN)

# ── Dataset ────────────────────────────────────────────────────────────────────
N_CLASSES = MFInstSegDataset.num_classes(DATASET_TYPE)
log.info(f'MachinaQ GNN Training  |  {N_CLASSES} classes  |  dataset: {DATASET_DIR}')

train_ds = MFInstSegDataset(
    root_dir=DATASET_DIR, split='train',
    center_and_scale=False, normalize=True, random_rotate=False,
    dataset_type=DATASET_TYPE, num_threads=4,
)

val_ds = MFInstSegDataset(
    root_dir=DATASET_DIR, split='val',
    center_and_scale=False, normalize=True,
    dataset_type=DATASET_TYPE, num_threads=4,
)

log.info(f'Train: {len(train_ds):,}  |  Val: {len(val_ds):,}')

train_loader = train_ds.get_dataloader(batch_size=BATCH_SIZE, pin_memory=True)
val_loader   = val_ds.get_dataloader(batch_size=BATCH_SIZE, shuffle=False,
                                     drop_last=False, pin_memory=True)

# ── Model ──────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f'Device: {DEVICE}')

model = AAGNetSegmentor(
    num_classes=N_CLASSES, arch='AAGNetGraphEncoder',
    edge_attr_dim=12, node_attr_dim=10,
    edge_attr_emb=64, node_attr_emb=64,
    edge_grid_dim=0,  node_grid_dim=7,
    edge_grid_emb=0,  node_grid_emb=64,
    num_layers=3, delta=2, mlp_ratio=2,
    drop=0.25, drop_path=0.25, head_hidden_dim=64,
    conv_on_edge=False, use_uv_gird=True,
    use_edge_attr=True, use_face_attr=True,
).to(DEVICE)

total = sum(p.numel() for p in model.parameters())
log.info(f'Parameters: {total:,}')

# Load pretrained weights if available
pretrained = os.path.join(MACHGNN, 'weights', 'weight_on_MFInstseg.pth')
if os.path.exists(pretrained):
    model.load_state_dict(torch.load(pretrained, map_location=DEVICE))
    log.info(f'Loaded pretrained weights from {pretrained}')

# ── Optimiser & losses ─────────────────────────────────────────────────────────
seg_loss  = nn.CrossEntropyLoss()
inst_loss = nn.BCEWithLogitsLoss()
bot_loss  = nn.BCEWithLogitsLoss()
opt       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=0)

iters     = len(train_loader)
ema_decay = (1.0 / 2.0) ** (1.0 / iters)
ema       = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

# ── Metrics ────────────────────────────────────────────────────────────────────
def make_metrics():
    return dict(
        seg_acc  = MulticlassAccuracy(num_classes=N_CLASSES).to(DEVICE),
        seg_iou  = MulticlassJaccardIndex(num_classes=N_CLASSES).to(DEVICE),
        inst_acc = BinaryAccuracy().to(DEVICE),
        inst_f1  = BinaryF1Score().to(DEVICE),
        bot_acc  = BinaryAccuracy().to(DEVICE),
        bot_iou  = BinaryJaccardIndex().to(DEVICE),
    )

def compute_reset(m):
    r = {k: v.compute().item() for k, v in m.items()}
    for v in m.values(): v.reset()
    return r

# ── Training loop ──────────────────────────────────────────────────────────────
best_score = 0.0
log.info(f'Starting training: {EPOCHS} epochs  batch={BATCH_SIZE}  lr={LR}')

for epoch in range(EPOCHS):
    t0 = time.time()
    model.train()
    tm = make_metrics()
    train_losses = []

    for data in tqdm(train_loader, desc=f'Ep {epoch+1}/{EPOCHS} train', leave=False):
        g          = data['graph'].to(DEVICE, non_blocking=True)
        inst_lbl   = data['inst_labels'].to(DEVICE, non_blocking=True)
        seg_lbl    = g.ndata['seg_y']
        bot_lbl    = g.ndata['bottom_y']

        opt.zero_grad(set_to_none=True)
        seg_p, inst_p, bot_p = model(g)

        loss = (SEG_W  * seg_loss(seg_p, seg_lbl) +
                INST_W * inst_loss(inst_p, inst_lbl) +
                BOT_W  * bot_loss(bot_p, bot_lbl))
        loss.backward()
        opt.step()
        ema.update()

        train_losses.append(loss.item())
        tm['seg_acc'].update(seg_p, seg_lbl)
        tm['seg_iou'].update(seg_p, seg_lbl)
        tm['inst_acc'].update(inst_p, inst_lbl)
        tm['inst_f1'].update(inst_p, inst_lbl)
        tm['bot_acc'].update(bot_p, bot_lbl)
        tm['bot_iou'].update(bot_p, bot_lbl)

    scheduler.step()
    tr = compute_reset(tm)
    tr['loss'] = float(np.mean(train_losses))

    # Validation
    vm = make_metrics()
    val_losses = []
    with torch.no_grad(), ema.average_parameters():
        model.eval()
        for data in tqdm(val_loader, desc=f'Ep {epoch+1}/{EPOCHS} val', leave=False):
            g        = data['graph'].to(DEVICE)
            inst_lbl = data['inst_labels'].to(DEVICE)
            seg_lbl  = g.ndata['seg_y']
            bot_lbl  = g.ndata['bottom_y']

            seg_p, inst_p, bot_p = model(g)
            loss = (SEG_W  * seg_loss(seg_p, seg_lbl) +
                    INST_W * inst_loss(inst_p, inst_lbl) +
                    BOT_W  * bot_loss(bot_p, bot_lbl))
            val_losses.append(loss.item())
            vm['seg_acc'].update(seg_p, seg_lbl)
            vm['seg_iou'].update(seg_p, seg_lbl)
            vm['inst_acc'].update(inst_p, inst_lbl)
            vm['inst_f1'].update(inst_p, inst_lbl)
            vm['bot_acc'].update(bot_p, bot_lbl)
            vm['bot_iou'].update(bot_p, bot_lbl)

    vl = compute_reset(vm)
    vl['loss'] = float(np.mean(val_losses))

    score = vl['seg_iou'] + vl['inst_f1'] + vl['bot_iou']
    star = ''
    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), SAVE_PATH)
        star = ' * saved'

    elapsed = time.time() - t0
    log.info(
        f"Ep {epoch+1:3d}/{EPOCHS}  {elapsed:.0f}s  "
        f"loss={tr['loss']:.4f}/{vl['loss']:.4f}  "
        f"seg_iou={tr['seg_iou']:.3f}/{vl['seg_iou']:.3f}  "
        f"inst_f1={tr['inst_f1']:.3f}/{vl['inst_f1']:.3f}  "
        f"bot_iou={tr['bot_iou']:.3f}/{vl['bot_iou']:.3f}{star}"
    )

log.info(f'Training complete. Best score: {best_score:.4f}  ->  {SAVE_PATH}')
