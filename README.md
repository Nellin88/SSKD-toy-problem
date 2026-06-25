# sskd-toy-problem

A **control dataset** that isolates a *single* hypothesis from
**"Knowledge Distillation Meets Self-Supervision"** (Xu et al., ECCV 2020,
[arXiv:2006.07114](https://arxiv.org/abs/2006.07114)):

> SSKD's improvement over vanilla KD comes specifically from the **spatial /
> relational structure** in the images — information the class label need not
> carry, but which the self-supervised contrastive task (built on spatial
> transforms) is designed to capture.

This repo contains **only the dataset and the code that generates it**. Training
the teacher/student under KD and SSKD is done with the authors' released code
(<https://github.com/xuguodong03/SSKD>); just point it at the two datasets below.

## Design: two datasets, one variable

Every 32×32 image has two objects, one **red** and one **blue**, carrying two
**independent** attributes:

- **C — shape** : both objects are **circles** or both are **squares**
  (read from pixels, *no position needed* → **non-spatial**).
- **S — relative position** : the two objects are arranged **horizontally** or
  **vertically** (read from layout, *regardless of shape/colour* → **spatial**).

The two datasets share the **same image pool**; only the **labelling rule** differs:

| dataset | label rule | spatial structure is… | prediction |
|---------|------------|------------------------|------------|
| **A** (control)    | `label = C` (shape)          | **useless** (layout doesn't change the label) | SSKD ≈ KD |
| **B** (experiment) | `label = S` (relative position) | **decisive** (layout determines the label)    | SSKD > KD |

The *same* image gets label "square" in A and "vertical" in B. There is **no label
noise** — every label is the deterministic correct answer under its rule.

If `gap(B) = acc(SSKD) − acc(KD)` is clearly larger than `gap(A) ≈ 0`, the gain is
attributable to capturing spatial structure — the paper's claim.

### Confounders closed
- Same task type and same image pool in A and B; only the label rule changes (van Gemert `c1`).
- C and S drawn independently and balanced; neither dataset is biased toward an attribute.
- Which colour sits in the first slot is randomised per image and independent of the label → no "red is always left" shortcut.
- Both shapes and both layouts appear in **both** datasets → the self-supervision task always has spatial structure to learn from, so a flat `gap(A)` means "structure useless", not "the SS task had nothing to learn".

## Files

| file | role |
|------|------|
| `generate.py` | builds dataset A or B (the only code required) |
| `requirements.txt` | `numpy`, `pillow` |

## Generate

```bash
pip install -r requirements.txt

python generate.py --dataset A --out ../data/A     # control:    label = shape
python generate.py --dataset B --out ../data/B     # experiment: label = position
```

Each call writes `train.npz` / `test.npz` (`X`: uint8 N×32×32×3, `y`: int64) plus a
few PNG samples. Datasets are fully reproducible from `(seed, dataset)`.

Licensed for coursework use.
