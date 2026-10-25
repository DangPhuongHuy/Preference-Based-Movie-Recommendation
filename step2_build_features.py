"""
Bước 2: chia train / validation / test (70/15/15) và biến mỗi phim thành các con số.

  - Thể loại (0/1), năm, thời lượng, ngôn ngữ
  - 300 từ khóa quan trọng nhất trong mô tả (TF-IDF)
  - Chuẩn hóa tất cả cột về cùng thang (trung bình 0, độ lệch chuẩn 1) để so sánh hệ số
Mọi thứ "học" chỉ lấy từ tập train, tránh nhìn trộm tập test.

Chạy:  python step2_build_features.py
Đầu ra (lr_artifacts/): data.npz, pipeline.joblib
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lr_utils import *


def main():
    df = pd.read_pickle(f"{ART}/movies.pkl")
    y = df["rating"].values.astype("float64")
    idx = np.arange(len(df))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, random_state=42)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, random_state=42)
    tr, va, te = df.iloc[i_tr], df.iloc[i_va], df.iloc[i_te]
    log(f"Train {len(tr):,} | Validation {len(va):,} | Test {len(te):,}")

    feats = DenseFeatures(tr)
    tfidf = TfidfVectorizer(max_features=N_WORDS, min_df=5, stop_words="english", sublinear_tf=True, dtype=np.float32)
    tfidf.fit(tr["text"])
    dense_names = list(feats.transform(tr.head(2)).columns)
    word_names = [f"Từ khóa: {w}" for w in tfidf.get_feature_names_out()]
    names = dense_names + word_names

    raw = lambda d: np.hstack([feats.transform(d).values.astype("float32"),
                               tfidf.transform(d["text"]).toarray().astype("float32")])
    scaler = StandardScaler().fit(raw(tr))
    X = {k: scaler.transform(raw(d)).astype("float32") for k, d in (("tr", tr), ("va", va), ("te", te))}
    n_dense = len(dense_names)
    log(f"Mỗi phim = {X['tr'].shape[1]} con số ({n_dense} đặc trưng bảng + {len(word_names)} từ khóa)")

    np.savez_compressed(f"{ART}/data.npz", X_tr=X["tr"], X_va=X["va"], X_te=X["te"],
                        y_tr=y[i_tr], y_va=y[i_va], y_te=y[i_te], i_te=i_te)
    joblib.dump(dict(feats=feats, tfidf=tfidf, scaler=scaler, names=names, n_dense=n_dense), f"{ART}/pipeline.joblib")
    print(f"\nXong. Đã lưu vào {ART}/")
    print("Ví dụ 1 phim sau khi biến thành số (5 giá trị đầu):", X["tr"][0][:5].round(2))


if __name__ == "__main__":
    main()
