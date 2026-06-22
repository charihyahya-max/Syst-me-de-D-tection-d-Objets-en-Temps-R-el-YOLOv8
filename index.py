"""
Projet 2 : Système de Détection d'Objets en Temps Réel — YOLOv8
Technologies: Python, YOLOv8 (simulé), OpenCV (via matplotlib), NumPy

Ce script simule un pipeline YOLOv8 complet :
  1. Génération d'une "scène" réaliste avec des objets
  2. Simulation de bounding boxes + scores de confiance
  3. Évaluation des performances (mAP, Precision, Recall)
  4. Dashboard de visualisation complet
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── Palette ──────────────────────────────────────────────
BG      = '#0B0D11'
SURFACE = '#13161D'
BORDER  = '#1F2430'
TEXT    = '#CDD5F0'
BRIGHT  = '#E8EDFF'
SUBTLE  = '#6B728E'
TEAL    = '#3BFFB8'
YELLOW  = '#FFD23F'
PURPLE  = '#7B61FF'
RED     = '#FF6B6B'
ORANGE  = '#FF9F43'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': SURFACE,
    'axes.edgecolor':   BORDER, 'axes.labelcolor': TEXT,
    'xtick.color':      SUBTLE,  'ytick.color': SUBTLE,
    'text.color':       TEXT,    'grid.color': BORDER,
    'grid.linewidth':   0.6,     'font.family': 'monospace',
    'axes.spines.top':  False,   'axes.spines.right': False,
})

np.random.seed(7)

# ── Class definitions ─────────────────────────────────────
CLASSES = ['Personne', 'Voiture', 'Téléphone', 'Vélo', 'Chien', 'Sac', 'Ordinateur']
CLASS_COLORS = {
    'Personne':   '#3BFFB8',
    'Voiture':    '#FFD23F',
    'Téléphone':  '#7B61FF',
    'Vélo':       '#FF9F43',
    'Chien':      '#FF6B6B',
    'Sac':        '#4ECDC4',
    'Ordinateur': '#A29BFE',
}

# ── Simulate detections on a synthetic scene ──────────────
def generate_scene(img_w=640, img_h=480):
    """Generate background scene as a numpy array."""
    scene = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    # Sky gradient
    for y in range(img_h // 2):
        v = int(11 + y * 0.06)
        scene[y, :] = [v, v+2, v+5]
    # Ground
    for y in range(img_h // 2, img_h):
        v = int(14 + (y - img_h // 2) * 0.04)
        scene[y, :] = [v, v, v+1]
    # Road
    scene[300:, 140:500] = [22, 24, 28]
    return scene

def make_boxes():
    """Simulate YOLOv8 predicted bounding boxes [x1,y1,x2,y2, conf, cls]."""
    return [
        # x1,  y1,  x2,  y2,  conf, class
        [  80, 160, 175, 420, 0.94, 'Personne'],
        [ 200, 150, 310, 430, 0.89, 'Personne'],
        [ 340,  80, 570, 310, 0.87, 'Voiture'],
        [  50, 300, 130, 390, 0.91, 'Vélo'],
        [ 420, 310, 510, 430, 0.76, 'Chien'],
        [ 120, 170, 155, 220, 0.82, 'Téléphone'],
        [ 500, 100, 620, 200, 0.73, 'Ordinateur'],
        [ 230, 350, 300, 440, 0.68, 'Sac'],
    ]

# ── Per-class performance metrics (simulated) ─────────────
metrics = {
    cls: {
        'precision': round(np.random.uniform(0.72, 0.97), 3),
        'recall':    round(np.random.uniform(0.68, 0.95), 3),
        'AP50':      round(np.random.uniform(0.65, 0.96), 3),
        'samples':   int(np.random.randint(80, 400)),
    }
    for cls in CLASSES
}
mAP50  = round(np.mean([m['AP50']      for m in metrics.values()]), 3)
mPrec  = round(np.mean([m['precision'] for m in metrics.values()]), 3)
mRecall= round(np.mean([m['recall']    for m in metrics.values()]), 3)
f1     = round(2 * mPrec * mRecall / (mPrec + mRecall), 3)

# Training curves
epochs  = 50
train_loss = 2.8 * np.exp(-np.linspace(0, 3.5, epochs)) + np.random.normal(0, 0.04, epochs)
val_loss   = 2.9 * np.exp(-np.linspace(0, 3.0, epochs)) + np.random.normal(0, 0.06, epochs)
map_curve  = mAP50 * (1 - np.exp(-np.linspace(0, 4, epochs))) + np.random.normal(0, 0.008, epochs)
map_curve  = np.clip(map_curve, 0, 1)

# Confusion matrix (normalised)
n_cls = len(CLASSES)
conf_mat = np.eye(n_cls) * np.random.uniform(0.70, 0.95, n_cls)
for i in range(n_cls):
    off = 1 - conf_mat[i, i]
    others = np.random.dirichlet(np.ones(n_cls - 1)) * off
    idx = [j for j in range(n_cls) if j != i]
    for k, j in enumerate(idx):
        conf_mat[i, j] = others[k]

# ── Build figure ──────────────────────────────────────────
fig = plt.figure(figsize=(18, 13), facecolor=BG)

fig.text(0.03, 0.97, 'DÉTECTION D\'OBJETS EN TEMPS RÉEL — YOLOv8',
         fontsize=13, fontweight='bold', color=BRIGHT, va='top')
fig.text(0.03, 0.938, f'mAP@0.5 : {mAP50:.3f}  ·  Précision : {mPrec:.3f}  ·  Rappel : {mRecall:.3f}  ·  F1 : {f1:.3f}  ·  {n_cls} classes',
         fontsize=9, color=SUBTLE, va='top')

# KPI strip
kpi_items = [
    (f'{mAP50:.1%}',  'mAP@0.5'),
    (f'{mPrec:.1%}',  'Précision'),
    (f'{mRecall:.1%}','Rappel'),
    (f'{f1:.1%}',     'Score F1'),
    ('640×640',       'Résolution'),
    ('~45 FPS',       'Inférence'),
]
for i, (val, lbl) in enumerate(kpi_items):
    x = 0.03 + i * 0.158
    fig.text(x, 0.895, val, fontsize=13, fontweight='bold', color=TEAL, va='top')
    fig.text(x, 0.868, lbl, fontsize=7.5, color=SUBTLE, va='top')

line = plt.Line2D([0.03, 0.97], [0.855, 0.855], transform=fig.transFigure,
                  color=BORDER, linewidth=1)
fig.add_artist(line)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.03, right=0.97,
                       top=0.845, bottom=0.04,
                       hspace=0.42, wspace=0.32)

# ── Panel 1 · Detection scene ─────────────────────────────
ax_scene = fig.add_subplot(gs[0, :2])
scene = generate_scene()
ax_scene.imshow(scene, aspect='auto')
ax_scene.set_title('Scène détectée — Bounding Boxes YOLOv8', color=BRIGHT,
                    fontsize=10, pad=8, loc='left')

boxes = make_boxes()
for x1, y1, x2, y2, conf, cls in boxes:
    color = CLASS_COLORS[cls]
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=2, edgecolor=color, facecolor='none')
    ax_scene.add_patch(rect)
    # Header box
    lbl = f'{cls}  {conf:.2f}'
    ax_scene.text(x1 + 3, y1 - 5, lbl, color=BG, fontsize=7.5,
                  fontweight='bold', va='bottom',
                  bbox=dict(boxstyle='square,pad=0.25', facecolor=color, linewidth=0))
    # Corner dots
    for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        ax_scene.plot(cx, cy, 'o', color=color, ms=3, zorder=5)

ax_scene.set_xlim(0, 640); ax_scene.set_ylim(480, 0)
ax_scene.axis('off')

# Overlay: detection count badge
ax_scene.text(626, 12, f'{len(boxes)} obj.', color=TEAL, fontsize=8,
              fontweight='bold', ha='right', va='top',
              bbox=dict(boxstyle='round,pad=0.4', facecolor=SURFACE, linewidth=0))

# ── Panel 2 · AP per class ────────────────────────────────
ax_ap = fig.add_subplot(gs[0, 2])
ap_vals = [metrics[c]['AP50'] for c in CLASSES]
bar_colors = [CLASS_COLORS[c] for c in CLASSES]
bars = ax_ap.barh(CLASSES, ap_vals, color=bar_colors, height=0.6, alpha=0.9)
ax_ap.axvline(mAP50, color=YELLOW, linewidth=1.2, linestyle='--', zorder=5)
ax_ap.text(mAP50 + 0.005, len(CLASSES) - 0.3, f'mAP {mAP50:.2f}',
           color=YELLOW, fontsize=7.5)
ax_ap.set_xlim(0, 1.05)
ax_ap.set_title('AP@0.5 par classe', color=BRIGHT, fontsize=10, pad=8, loc='left')
for bar, val in zip(bars, ap_vals):
    ax_ap.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
               f'{val:.2f}', va='center', fontsize=7.5, color=TEXT)
ax_ap.grid(axis='x', zorder=0)

# ── Panel 3 · Training curves ─────────────────────────────
ax_loss = fig.add_subplot(gs[1, :2])
ep = range(1, epochs + 1)
ax_loss.plot(ep, train_loss, color=TEAL,   linewidth=2,   label='Train loss', zorder=3)
ax_loss.plot(ep, val_loss,   color=YELLOW, linewidth=2,   label='Val loss',   zorder=3)
ax_loss2 = ax_loss.twinx()
ax_loss2.plot(ep, map_curve, color=PURPLE, linewidth=2, linestyle='--', label='mAP@0.5', zorder=3)
ax_loss2.set_ylabel('mAP@0.5', color=PURPLE, fontsize=8)
ax_loss2.tick_params(colors=PURPLE)
ax_loss2.set_ylim(0, 1)
ax_loss2.spines['right'].set_color(BORDER)
ax_loss2.spines['top'].set_visible(False)
ax_loss.set_xlabel('Époque', fontsize=8)
ax_loss.set_ylabel('Loss', fontsize=8)
ax_loss.set_title('Courbes d\'entraînement', color=BRIGHT, fontsize=10, pad=8, loc='left')
lines1, lbl1 = ax_loss.get_legend_handles_labels()
lines2, lbl2 = ax_loss2.get_legend_handles_labels()
ax_loss.legend(lines1 + lines2, lbl1 + lbl2,
               loc='upper right', fontsize=7.5,
               facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
ax_loss.grid(zorder=0)

# ── Panel 4 · Confusion matrix ────────────────────────────
ax_cm = fig.add_subplot(gs[1, 2])
im = ax_cm.imshow(conf_mat, cmap='Blues', vmin=0, vmax=1, aspect='auto')
ax_cm.set_xticks(range(n_cls))
ax_cm.set_yticks(range(n_cls))
short = [c[:4] for c in CLASSES]
ax_cm.set_xticklabels(short, fontsize=6.5, rotation=45, ha='right')
ax_cm.set_yticklabels(short, fontsize=6.5)
for i in range(n_cls):
    for j in range(n_cls):
        val = conf_mat[i, j]
        color = BG if val > 0.5 else TEXT
        ax_cm.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)
ax_cm.set_title('Matrice de confusion (norm.)', color=BRIGHT, fontsize=10, pad=8, loc='left')
plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7, colors=SUBTLE)

plt.tight_layout()

plt.savefig(
    "projet2_yolov8_detection.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG
)

plt.show()
print("✅ Projet 2 saved → projet2_yolov8_detection.png")

print("\n─── MÉTRIQUES YOLOv8 ───────────────────────────")
print(f"  mAP@0.5          : {mAP50:.3f}")
print(f"  Précision globale: {mPrec:.3f}")
print(f"  Rappel global    : {mRecall:.3f}")
print(f"  Score F1         : {f1:.3f}")
print(f"  Objets détectés  : {len(boxes)}")
print(f"  Classes          : {', '.join(CLASSES)}")
print("────────────────────────────────────────────────")