# Datasets

This folder is the expected location of the **WeaponSenseV2** dataset used by
all training and evaluation scripts in this repository. The actual image and
label files are **not** distributed here — they are subject to the original
data-sharing agreement of the dataset.

The directory tree, the YOLO-format `data.yaml` template, and this README are
tracked so that anyone reproducing the experiments knows exactly where to drop
the dataset and what layout the code expects.

---

## Expected layout

```
data/
└── WeaponSenseV2/
    ├── data.yaml                    ← class names + split paths (TRACKED)
    ├── train/
    │   ├── images/                  ← put .jpg frames here
    │   │   ├── video01_frame000.jpg
    │   │   ├── video01_frame001.jpg
    │   │   └── ...
    │   └── labels/                  ← put YOLO .txt labels here
    │       ├── video01_frame000.txt
    │       ├── video01_frame001.txt
    │       └── ...
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

Each `*.txt` label file contains one detection per line in the standard
YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```
with all coordinates normalised to the `[0, 1]` range relative to the
image dimensions. `class_id` is `0` for `Handgun` and `1` for `Knife`,
as defined in `WeaponSenseV2/data.yaml`.

---

## Dataset statistics (after the video-level split used in this work)

| Subset      | Frames | Handgun boxes | Knife boxes |
|-------------|--------|---------------|-------------|
| Train       | 2098   | 1292          | 1029        |
| Validation  | 474    | 313           | 272         |
| Test        | 451    | 211           | 244         |
| **Total**   | **3023** | **1816**    | **1545**    |

The split is **video-level**: every frame from a given source video is
assigned to a single subset (train / val / test). This prevents the
cross-frame data leakage that would arise from a naive random split, in
which highly correlated frames from the same video could appear in both
the training and the test set.

---

## How to obtain WeaponSenseV2

The WeaponSenseV2 dataset is described in:

> D. Berardini, L. Migliorelli, A. Galdelli, M. J. Marín-Jiménez.
> *Benchmark Analysis of YOLOv8 for Edge AI Video-Surveillance
> Applications.* IEEE International Symposium on Measurements and
> Networking (M&N), 2024.

> D. Berardini, L. Migliorelli, A. Galdelli, E. Frontoni, A. Mancini,
> S. Moccia. *A deep-learning framework running on edge devices for
> handgun and knife detection from indoor video-surveillance cameras.*
> Multimedia Tools and Applications, 83(7): 19109–19127, 2024.

Access to the raw frames and labels should be requested directly from the
dataset authors via the channels indicated in the above references.

---

## Reproducing the split used in this work

The exact list of training, validation, and test video IDs used in our
experiments is encoded directly in the script
[`src/data/split_test.py`](../src/data/split_test.py). After placing the
full WeaponSenseV2 train and val frames at the locations indicated above
(originally distributed by the dataset authors as a single `train` /
`val` partition), running

```bash
PYTHONPATH=src python -m data.split_test
```

from the project root will move the eight test videos out of `train/`
and into `test/`, producing exactly the partition used in this work.
The script also performs a final overlap check to verify that no video
ID appears in more than one subset.

---

## Sanity check

Once the dataset has been placed at the expected location, the following
command from the project root should print non-zero counts for every
subset:

```bash
python - <<'PY'
from pathlib import Path
for split in ("train", "val", "test"):
    n_imgs = len(list(Path(f"data/WeaponSenseV2/{split}/images").glob("*.jpg")))
    n_lbls = len(list(Path(f"data/WeaponSenseV2/{split}/labels").glob("*.txt")))
    print(f"{split:>5}: {n_imgs} images, {n_lbls} labels")
PY
```

Expected output (for the split used in this work):
```
train: 2098 images, 2098 labels
  val: 474 images, 474 labels
 test: 451 images, 451 labels
```
