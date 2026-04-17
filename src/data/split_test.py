"""
Split test set from train set at the VIDEO level to prevent data leakage.
Frames from the same video will never appear in both train and test sets.

Target: ~15% of train frames for test (~380 frames from ~8 videos).
Ensures balanced class representation (class 0 and class 1).
"""

import os
import shutil
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent / "data" / "WeaponSenseV2"
TRAIN_IMG = BASE_DIR / "train" / "images"
TRAIN_LBL = BASE_DIR / "train" / "labels"
TEST_IMG = BASE_DIR / "test" / "images"
TEST_LBL = BASE_DIR / "test" / "labels"

def get_video_id(filename):
    """Extract video ID from filename like 'market_video1_frame123.jpg'."""
    name = Path(filename).stem
    parts = name.rsplit("_frame", 1)
    return parts[0]

def get_video_classes(video_id, label_dir):
    """Get the set of classes present in a video's labels."""
    classes = set()
    for lbl_file in label_dir.glob(f"{video_id}_frame*.txt"):
        with open(lbl_file) as f:
            for line in f:
                cls_id = line.strip().split()[0]
                classes.add(int(cls_id))
    return classes

def main():
    # Group frames by video
    video_frames = defaultdict(list)
    for img_file in sorted(TRAIN_IMG.glob("*.jpg")):
        vid = get_video_id(img_file.name)
        video_frames[vid].append(img_file.name)

    total_frames = sum(len(frames) for frames in video_frames.values())
    print(f"Total train frames: {total_frames}")
    print(f"Total videos: {len(video_frames)}")

    # Get class info per video
    video_info = {}
    for vid, frames in video_frames.items():
        classes = get_video_classes(vid, TRAIN_LBL)
        video_info[vid] = {"frames": len(frames), "classes": classes}
        print(f"  {vid}: {len(frames)} frames, classes: {classes}")

    # Select test videos: aim for ~15% of frames with balanced classes
    # Selected to include both class-0-only, class-1-only, and mixed videos
    test_videos = [
        "video35",       # 73 frames, classes {0, 1} - mixed
        "video14",       # 65 frames, classes {1}    - handgun
        "video33",       # 59 frames, classes {0}    - knife
        "video20",       # 54 frames, classes {1}    - handgun
        "video21",       # 53 frames, classes {0}    - knife
        "video19",       # 50 frames, classes {0, 1} - mixed
        "video36",       # 49 frames, classes {1}    - handgun
        "video1",        # 48 frames, classes {0}    - knife
    ]

    test_frame_count = sum(video_info[v]["frames"] for v in test_videos)
    test_classes = set()
    for v in test_videos:
        test_classes.update(video_info[v]["classes"])

    print(f"\nSelected test videos: {test_videos}")
    print(f"Test frames: {test_frame_count} ({test_frame_count/total_frames*100:.1f}%)")
    print(f"Test classes covered: {test_classes}")
    print(f"Remaining train frames: {total_frames - test_frame_count}")

    # Create test directories
    TEST_IMG.mkdir(parents=True, exist_ok=True)
    TEST_LBL.mkdir(parents=True, exist_ok=True)

    # Move files
    moved_count = 0
    for vid in test_videos:
        for img_name in video_frames[vid]:
            lbl_name = Path(img_name).stem + ".txt"

            src_img = TRAIN_IMG / img_name
            src_lbl = TRAIN_LBL / lbl_name
            dst_img = TEST_IMG / img_name
            dst_lbl = TEST_LBL / lbl_name

            shutil.move(str(src_img), str(dst_img))
            if src_lbl.exists():
                shutil.move(str(src_lbl), str(dst_lbl))

            moved_count += 1

    print(f"\nMoved {moved_count} frames to test set.")

    # Verify no overlap
    train_vids = set()
    for img_file in TRAIN_IMG.glob("*.jpg"):
        train_vids.add(get_video_id(img_file.name))

    test_vids = set()
    for img_file in TEST_IMG.glob("*.jpg"):
        test_vids.add(get_video_id(img_file.name))

    overlap = train_vids & test_vids
    if overlap:
        print(f"WARNING: Video overlap detected: {overlap}")
    else:
        print("VERIFIED: No video overlap between train and test sets.")

    print(f"\nFinal split:")
    print(f"  Train: {len(list(TRAIN_IMG.glob('*.jpg')))} frames, {len(train_vids)} videos")
    print(f"  Val:   {len(list((BASE_DIR / 'val' / 'images').glob('*.jpg')))} frames")
    print(f"  Test:  {len(list(TEST_IMG.glob('*.jpg')))} frames, {len(test_vids)} videos")


if __name__ == "__main__":
    main()
