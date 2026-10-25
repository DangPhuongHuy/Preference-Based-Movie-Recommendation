"""
Bước 1: đọc file TMDB, giữ phim có đủ lượt vote, tạo các cột cần cho mô hình.

Dữ liệu đã sạch nên không lọc lại. Chỉ giữ phim có đủ vote vì phim ít người chấm
thì điểm trung bình rất nhiễu (có thể là 0 hoặc 10 chỉ vì 1-2 người chấm).

Chạy:  python step1_prepare_data.py --min-votes 50
Đầu ra: lr_artifacts/movies.pkl
"""
import argparse
import os

import numpy as np
import pandas as pd

from lr_utils import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--min-votes", type=int, default=50)
    ap.add_argument("--max-rows", type=int, default=200_000, help="0 = dùng hết")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(ART, exist_ok=True)

    want = {C_TARGET, C_VOTES, C_GENRES, C_OVERVIEW, C_TAGLINE, C_KEYWORDS, C_YEAR, C_DATE, C_RUNTIME, C_LANG, C_TITLE}
    df = pd.read_csv(a.csv, usecols=lambda c: c in want, low_memory=False, encoding="utf-8-sig")
    miss = [c for c in (C_TARGET, C_VOTES, C_GENRES, C_OVERVIEW) if c not in df.columns]
    if miss:
        raise SystemExit(f"File thiếu cột {miss}. Sửa tên cột trong lr_utils.py")
    n0 = len(df)
    log(f"Đã đọc {n0:,} phim")

    df[C_TARGET] = pd.to_numeric(df[C_TARGET], errors="coerce")
    df[C_VOTES] = pd.to_numeric(df[C_VOTES], errors="coerce").fillna(0)
    df = df[(df[C_VOTES] >= a.min_votes) & (df[C_TARGET] > 0)]
    log(f"Giữ phim có vote_count >= {a.min_votes} và điểm > 0: {n0:,} -> {len(df):,}")

    if C_YEAR in df.columns:
        year = pd.to_numeric(df[C_YEAR], errors="coerce")
    elif C_DATE in df.columns:
        year = pd.to_datetime(df[C_DATE], errors="coerce").dt.year
    else:
        year = pd.Series(np.nan, index=df.index)
    df["year"] = year.where(year.between(1880, 2030))
    rt = pd.to_numeric(df[C_RUNTIME], errors="coerce") if C_RUNTIME in df.columns else pd.Series(np.nan, index=df.index)
    df["runtime"] = rt.where(rt.between(1, 400))          # 0 hoặc vô lý = thiếu
    df["lang"] = df[C_LANG].fillna("unknown").astype(str) if C_LANG in df.columns else "unknown"
    df["genres_list"] = df[C_GENRES].apply(split_genres)
    for c in (C_KEYWORDS, C_TAGLINE):
        if c not in df.columns:
            df[c] = ""
    df["text"] = (df[C_KEYWORDS].fillna("").astype(str) + ". " + df[C_TAGLINE].fillna("").astype(str) + ". "
                  + df[C_OVERVIEW].fillna("").astype(str).str.slice(0, 1000))
    df["title"] = df[C_TITLE].fillna("").astype(str) if C_TITLE in df.columns else ""
    df["votes"] = df[C_VOTES]
    df["rating"] = df[C_TARGET]

    if a.max_rows and len(df) > a.max_rows:
        df = df.sample(a.max_rows, random_state=a.seed)
        log(f"Lấy ngẫu nhiên {a.max_rows:,} phim để chạy nhanh hơn")
    df = df[["title", "rating", "votes", "year", "runtime", "lang", "genres_list", "text"]].reset_index(drop=True)
    df.to_pickle(f"{ART}/movies.pkl")

    print(f"\nXong. Lưu {len(df):,} phim vào {ART}/movies.pkl")
    print(f"Điểm: trung bình {df.rating.mean():.3f}, độ lệch chuẩn {df.rating.std():.3f}, "
          f"min {df.rating.min():.1f}, max {df.rating.max():.1f}")
    print(f"Thiếu năm {df.year.isna().mean()*100:.1f}% | thiếu thời lượng {df.runtime.isna().mean()*100:.1f}%")
    print(df[["title", "rating", "votes", "year", "runtime", "lang"]].head())


if __name__ == "__main__":
    main()
