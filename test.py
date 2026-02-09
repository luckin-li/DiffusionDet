# language: python
#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_aug_files(aug_dir):
    aug_dir = Path(aug_dir)
    if not aug_dir.exists():
        return []
    files = []
    for p in aug_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_SUFFIXES:
            files.append(p)
    return sorted(files)


def build_image_id_to_catids(annotations):
    mapping = defaultdict(list)
    for ann in annotations:
        img_id = ann.get("image_id")
        cat = ann.get("category_id")
        if img_id is None or cat is None:
            continue
        mapping[img_id].append(cat)
    # dedupe while preserving order
    for k, v in mapping.items():
        seen = set()
        uniq = []
        for x in v:
            if x not in seen:
                uniq.append(x); seen.add(x)
        mapping[k] = uniq
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Add aug fields into COCO json and write to a new file")
    parser.add_argument("--input", required=True, help="COCO json input")
    parser.add_argument("--aug-dir", required=True, help="directory containing augmented images")
    parser.add_argument("--images-root", required=False, default="", help="images root for relative paths")
    parser.add_argument("--output", required=False, help="output json path (if omitted, add _with_aug.json)")
    parser.add_argument("--relative", action="store_true", help="store aug_file_names as relative to --images-root")
    parser.add_argument("--preview", type=int, default=0, help="print first N entries for quick inspection")
    args = parser.parse_args()

    inp_path = Path(args.input)
    if not inp_path.exists():
        print(f"input json not found: {inp_path}")
        return

    with inp_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    imgid2cats = build_image_id_to_catids(annotations)

    aug_files = find_aug_files(args.aug_dir)
    # map stem -> list(paths)
    stem_map = defaultdict(list)
    for p in aug_files:
        stem = p.stem  # filename without suffix
        # For matching "原名开头" 的增强图，需要原名是 prefix of p.stem.
        stem_map[stem].append(p)

    # Also allow matching by original stem being prefix of aug stem.
    aug_by_prefix = defaultdict(list)
    for p in aug_files:
        for img_entry in images:
            orig_fname = Path(img_entry.get("file_name", ""))
            orig_stem = orig_fname.stem
            if p.stem.startswith(orig_stem):
                aug_by_prefix[orig_stem].append(p)

    images_with_aug = 0
    for img in images:
        fname = img.get("file_name", "")
        orig_stem = Path(fname).stem
        # collect candidate aug files that start with orig_stem
        cands = aug_by_prefix.get(orig_stem, [])
        file_list = []
        for p in cands:
            if args.relative and args.images_root:
                file_list.append(os.path.relpath(str(p), start=args.images_root))
            else:
                file_list.append(str(p))
        img["aug_file_names"] = file_list
        # get category ids for this image id
        img_id = img.get("id")
        img["aug_gt_classes"] = imgid2cats.get(img_id, [])
        if file_list:
            images_with_aug += 1

    print(f"Processed {len(images)} images, {images_with_aug} with >=1 aug files found.")

    if args.preview and args.preview > 0:
        n = min(args.preview, len(images))
        print(f"Preview first {n} images (file_name, aug_file_names, aug_gt_classes):")
        for i in range(n):
            it = images[i]
            print(i, it.get("file_name"), it.get("aug_file_names")[:5], it.get("aug_gt_classes"))

    out_path = Path(args.output) if args.output else inp_path.with_name(inp_path.stem + "_with_aug.json")
    # write new coco dict (preserve other keys)
    coco_out = dict(coco)
    coco_out["images"] = images

    os.makedirs(out_path.parent, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(coco_out, f, ensure_ascii=False, indent=2)
    print(f"Written augmented json to: {out_path}")


if __name__ == "__main__":
    main()
