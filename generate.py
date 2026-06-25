"""
generate.py — a precise CONTROL dataset for testing ONE hypothesis of SSKD.

Hypothesis under test (Xu et al., "Knowledge Distillation Meets Self-Supervision",
ECCV 2020, https://arxiv.org/abs/2006.07114):

    SSKD's gain over vanilla KD comes specifically from the SPATIAL/RELATIONAL
    structure in the images --- information the class label need not carry, but
    which the self-supervised (contrastive) task, built on spatial transforms
    (crop/rotate/translate), is designed to capture.

CONTROLLED EXPERIMENT --- two datasets, one variable.
Every image contains two objects, one RED and one BLUE. Each image independently
carries two attributes:

    C (SHAPE identity)        : both objects are CIRCLES  or  both are SQUARES.
                                Read from pixels WITHOUT looking at position.
                                --> a purely NON-SPATIAL attribute.

    S (RELATIVE POSITION)     : the two objects are arranged HORIZONTALLY (side by
                                side) or VERTICALLY (stacked). Read from their
                                relative layout, regardless of shape or colour.
                                --> a purely SPATIAL/STRUCTURAL attribute.

C and S are drawn independently and uniformly, so the 4 combinations
{circle/horiz, circle/vert, square/horiz, square/vert} are balanced. The two
datasets use the SAME image-generation pool; only the LABELLING RULE differs:

    Dataset A  (CONTROL,  structure-irrelevant) :  label = C  (shape)
        --> changing the spatial layout does NOT change the label
        --> spatial structure is USELESS for the task
        --> prediction: SSKD ~= KD   (no extra structural signal to exploit)

    Dataset B  (EXPERIMENT, structure-decisive) :  label = S  (relative position)
        --> changing the spatial layout FLIPS the label
        --> the task CANNOT be solved without understanding spatial structure
        --> prediction: SSKD  >  KD   (SSKD's structural channel pays off)

If gap(B) = acc(SSKD)-acc(KD) is clearly larger than gap(A) ~= 0, the improvement
is attributable to capturing spatial structure --- the paper's claim.

NO label noise anywhere: every label is the deterministic, correct answer under
its dataset's rule. The ONLY thing that differs between A and B is which attribute
the label reads.

Confounders closed:
  * Same task type (binary classification) and same image pool in A and B; only
    the labelling rule (read C vs read S) changes.            -> van Gemert c1
  * C and S independent & balanced, so neither dataset is biased toward an attr.
  * Which colour occupies the first slot is randomised PER IMAGE and is
    independent of the label, so the model cannot cheat via "red is always
    left/top".
  * Both shapes and both layouts appear in BOTH datasets, so the self-supervision
    task always has spatial structure to learn from -- a flat gap in A means
    "structure is useless", not "the SS task had nothing to learn".

Usage:
    python generate.py --dataset A --out ../data/A      # control  (label = shape)
    python generate.py --dataset B --out ../data/B      # experiment(label = position)
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw

# --- fixed constants, identical for A and B --------------------------------- #
IMG = 32
OBJ = 9                 # bounding-box side of each object
RED = (220, 40, 40)
BLUE = (40, 70, 220)
BG = (0, 0, 0)

# slot centres for the two objects, for each layout
MARGIN = 8
_C0 = MARGIN + OBJ // 2
_C1 = IMG - MARGIN - OBJ // 2
LAYOUT = {
    "horizontal": [(_C0, IMG // 2), (_C1, IMG // 2)],   # (cx, cy) left, right
    "vertical":   [(IMG // 2, _C0), (IMG // 2, _C1)],   # top, bottom
}


def _draw_object(draw, center, shape, colour, rng):
    cx, cy = center
    # small label-irrelevant jitter so the cue is relational, not a fixed pixel
    cx += int(rng.integers(-1, 2))
    cy += int(rng.integers(-1, 2))
    r = OBJ // 2
    box = [cx - r, cy - r, cx + r, cy + r]
    if shape == "circle":
        draw.ellipse(box, fill=colour)
    else:
        draw.rectangle(box, fill=colour)


def render(shape, layout, first_is_red, rng):
    """Render one image.
        shape       : 'circle' | 'square'   (attribute C, same for both objects)
        layout      : 'horizontal' | 'vertical'  (attribute S)
        first_is_red: bool -- which colour sits in slot 0 (label-irrelevant)
    """
    img = Image.new("RGB", (IMG, IMG), BG)
    d = ImageDraw.Draw(img)
    slots = LAYOUT[layout]
    c0, c1 = (RED, BLUE) if first_is_red else (BLUE, RED)
    _draw_object(d, slots[0], shape, c0, rng)
    _draw_object(d, slots[1], shape, c1, rng)
    return np.asarray(img, dtype=np.uint8)


def make_split(n, dataset, rng):
    """Generate n samples. Attributes C and S are drawn independently; the label
    is C for dataset A and S for dataset B (deterministic, no noise)."""
    X = np.empty((n, IMG, IMG, 3), dtype=np.uint8)
    Y = np.empty((n,), dtype=np.int64)
    shapes = ["circle", "square"]
    layouts = ["horizontal", "vertical"]
    for i in range(n):
        c = int(rng.integers(0, 2))                 # 0=circle, 1=square   (C)
        s = int(rng.integers(0, 2))                 # 0=horizontal,1=vertical (S)
        first_is_red = bool(rng.integers(0, 2))     # label-irrelevant nuisance
        X[i] = render(shapes[c], layouts[s], first_is_red, rng)
        Y[i] = c if dataset == "A" else s           # the ONLY difference
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["A", "B"],
                    help="A = control (label=shape), B = experiment (label=position)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_png", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng_tr = np.random.default_rng(args.seed)
    rng_te = np.random.default_rng(args.seed + 10_000)

    Xtr, Ytr = make_split(args.n_train, args.dataset, rng_tr)
    Xte, Yte = make_split(args.n_test, args.dataset, rng_te)

    np.savez_compressed(os.path.join(args.out, "train.npz"), X=Xtr, y=Ytr)
    np.savez_compressed(os.path.join(args.out, "test.npz"), X=Xte, y=Yte)

    sdir = os.path.join(args.out, "samples")
    os.makedirs(sdir, exist_ok=True)
    for i in range(min(args.n_png, len(Xtr))):
        Image.fromarray(Xtr[i]).resize((128, 128), Image.NEAREST).save(
            os.path.join(sdir, f"train_{i}_label{Ytr[i]}.png"))

    rule = "shape (C, non-spatial)" if args.dataset == "A" else "relative position (S, spatial)"
    print(f"[generate] dataset {args.dataset}: label = {rule}")
    print(f"[generate] train={args.n_train} test={args.n_test} -> {args.out}")
    print(f"[generate] train class balance: {np.bincount(Ytr)}")


if __name__ == "__main__":
    main()
