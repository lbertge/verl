#!/usr/bin/env python3
"""
Create separate folders train_2_11 and eval_2_11 containing
only problems where meta.num_vars is in [2, 11]. Split those 80% train / 20% eval.
Does NOT delete or modify anything in train/ or eval/.
"""
import json
import os
import random
import shutil

random.seed(42)
TRAIN_DIR = "train"
EVAL_DIR = "eval"
OUT_TRAIN_DIR = "train_2_11"
OUT_EVAL_DIR = "eval_2_11"
MIN_VARS = 2
MAX_VARS = 11
TRAIN_FRAC = 0.8


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base, TRAIN_DIR)
    eval_dir = os.path.join(base, EVAL_DIR)
    out_train = os.path.join(base, OUT_TRAIN_DIR)
    out_eval = os.path.join(base, OUT_EVAL_DIR)
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_eval, exist_ok=True)

    # Collect (filename, path) for all problems with num_vars in [MIN_VARS, MAX_VARS]
    candidates = []
    for folder in [train_dir, eval_dir]:
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if not f.endswith(".json"):
                continue
            path = os.path.join(folder, f)
            try:
                with open(path, "r") as fp:
                    data = json.load(fp)
                nv = data.get("meta", {}).get("num_vars", -1)
                if MIN_VARS <= nv <= MAX_VARS:
                    candidates.append((f, path))
            except Exception as e:
                print("Error reading", path, e)

    # Deterministic split
    random.shuffle(candidates)
    n = len(candidates)
    n_train = int(round(n * TRAIN_FRAC))
    train_list = candidates[:n_train]
    eval_list = candidates[n_train:]

    # Copy into separate folders (do not touch originals)
    for filename, src in train_list:
        shutil.copy2(src, os.path.join(out_train, filename))
    for filename, src in eval_list:
        shutil.copy2(src, os.path.join(out_eval, filename))

    print(f"Problems with num_vars in [{MIN_VARS}, {MAX_VARS}]: {n}")
    print(f"Copied to {OUT_TRAIN_DIR}: {len(train_list)} files")
    print(f"Copied to {OUT_EVAL_DIR}: {len(eval_list)} files")
    print("Original train/ and eval/ were not modified.")


if __name__ == "__main__":
    main()
