# src/utils/print_.pkl_shape.py
"""
python src/utils/print_.pkl_shape.py data/tasks/task11/demo_rollout/demo.pkl
"""
import pickle, sys, numpy as np
from pathlib import Path

def summarize_array(name, arr, max_items=3):
    arr = np.asarray(arr)
    print(f"- {name}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.ndim == 1:
        print(f"  head: {arr[:max_items]}")
    elif arr.ndim >= 2:
        flat = arr.reshape(-1, arr.shape[-1]) if arr.shape[-1] <= 8 else arr.reshape(-1)  # best effort
        print(f"  sample rows: {flat[:max_items]}")

def main(pkl_path):
    p = Path(pkl_path)
    d = pickle.load(open(p, "rb"))
    print(f"== {p} ==")
    if isinstance(d, dict):
        for k, v in d.items():
            if hasattr(v, "shape"):
                summarize_array(k, v)
            elif isinstance(v, (list, tuple)) and len(v) > 0 and hasattr(v[0], "shape"):
                print(f"- {k}: list len={len(v)}, sample shape={getattr(v[0], 'shape', None)}")
            else:
                print(f"- {k}: type={type(v)}")
    else:
        print(type(d))

if __name__ == "__main__":
    main(sys.argv[1])
