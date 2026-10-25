"""
Bước 0: kiểm tra thư viện và file dữ liệu.     Chạy:  python step0_check_env.py
"""
import importlib
import os
import sys

from lr_utils import *

print("Python:", sys.version.split()[0])
for pkg, hint in [("pandas", "pandas"), ("numpy", "numpy"), ("sklearn", "scikit-learn"),
                  ("scipy", "scipy"), ("joblib", "joblib"), ("textual", "textual")]:
    try:
        m = importlib.import_module(pkg)
        print(f"[OK]    {pkg} {getattr(m, '__version__', '')}")
    except ImportError:
        print(f"[THIẾU] {pkg}  ->  pip install {hint}")

print(f"\nFile dữ liệu: {CSV}", "[OK]" if os.path.exists(CSV) else "[KHÔNG THẤY - đặt file cùng thư mục với code]")
if os.path.exists(CSV):
    import pandas as pd
    cols = pd.read_csv(CSV, nrows=5, encoding="utf-8-sig").columns.tolist()
    print("Các cột:", cols)
    need = [C_TARGET, C_VOTES, C_GENRES, C_OVERVIEW]
    print("Cột bắt buộc còn thiếu:", [c for c in need if c not in cols] or "không có")
    opt = [C_TAGLINE, C_KEYWORDS, C_YEAR, C_RUNTIME, C_LANG, C_TITLE]
    print("Cột tùy chọn còn thiếu:", [c for c in opt if c not in cols] or "không có")
