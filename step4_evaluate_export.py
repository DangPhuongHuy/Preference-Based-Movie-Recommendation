"""
Bước 4: đánh giá trên tập TEST và XUẤT DỮ LIỆU cho giao diện terminal ở bước 5.

Chạy:  python step4_evaluate_export.py
Đầu ra (lr_artifacts/app_data/): summary.json, results.csv, coefficients.csv, genre_stats.csv,
                                  test_predictions.csv, catalog_X.npy, catalog_T.npy, catalog_pred.npy
(3 file catalog_* là "kho phim" đã biến thành số, để step 5 gợi ý phim mà không phải tính lại)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from lr_utils import *


def main():
    d = np.load(f"{ART}/data.npz")
    pipe = joblib.load(f"{ART}/pipeline.joblib")
    M = joblib.load(f"{ART}/models.joblib")
    df = pd.read_pickle(f"{ART}/movies.pkl")
    out = f"{ART}/app_data"
    os.makedirs(out, exist_ok=True)
    nd, y = pipe["n_dense"], d["y_te"]
    te = df.iloc[d["i_te"]].reset_index(drop=True)

    p0 = np.full(len(y), M["base"])
    p1 = M["m1"].predict(d["X_te"][:, :nd])
    p2 = M["m2"].predict(d["X_te"])
    names = ["M0 Baseline (đoán trung bình)", "M1 Tuyến tính (đặc trưng bảng)", "M2 Tuyến tính (+ từ khóa)"]
    rows = []
    for n, p in zip(names, (p0, p1, p2)):
        m = metrics(y, p)
        rows.append(dict(model=n, RMSE=m["RMSE"], MAE=m["MAE"], R2=m["R2"], within1=m["within1"],
                         rmse_gain=1 - m["RMSE"] / metrics(y, p0)["RMSE"]))
    res = pd.DataFrame(rows)
    res.to_csv(f"{out}/results.csv", index=False, encoding="utf-8-sig")

    coef = pd.DataFrame({"feature": pipe["names"], "coef": M["m2"].coef_})
    coef["abs"] = coef["coef"].abs()
    coef.sort_values("abs", ascending=False).drop(columns="abs").to_csv(f"{out}/coefficients.csv", index=False, encoding="utf-8-sig")

    pred = np.clip(p2, 0, 10)
    tp = pd.DataFrame({"title": te["title"], "year": te["year"], "genres": te["genres_list"].apply(", ".join),
                       "actual": y, "predicted": pred, "error": pred - y})
    tp.to_csv(f"{out}/test_predictions.csv", index=False, encoding="utf-8-sig")

    g = tp.assign(genre=tp["genres"].str.split(", ")).explode("genre")
    g = g[g["genre"].notna() & (g["genre"] != "")]
    gs = g.groupby("genre").agg(n=("actual", "size"), actual=("actual", "mean"), predicted=("predicted", "mean")).reset_index()
    gs[gs["n"] >= 20].sort_values("actual", ascending=False).to_csv(f"{out}/genre_stats.csv", index=False, encoding="utf-8-sig")

    tr_n = len(np.load(f"{ART}/data.npz")["y_tr"])
    summary = dict(n_train=int(tr_n), n_val=int(len(d["y_va"])), n_test=int(len(y)), n_features=int(len(pipe["names"])),
                   n_dense=int(nd), baseline=float(M["base"]), intercept=float(M["m2"].intercept_),
                   mean_actual=float(y.mean()), std_actual=float(y.std()))
    json.dump(summary, open(f"{out}/summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- kho phim cho step 5: đặc trưng chuẩn hóa, TF-IDF, điểm dự đoán của mô hình M2
    nX, nT, nP = [], [], []
    for a in range(0, len(df), 20000):
        chunk = df.iloc[a:a + 20000]
        X = build_matrix(chunk, pipe)
        nX.append(X.astype("float32"))
        nT.append(pipe["tfidf"].transform(chunk["text"]).toarray().astype("float32"))
        nP.append(np.clip(M["m2"].predict(X), 0, 10).astype("float32"))
    np.save(f"{out}/catalog_X.npy", np.vstack(nX))
    np.save(f"{out}/catalog_T.npy", np.vstack(nT))
    np.save(f"{out}/catalog_pred.npy", np.concatenate(nP))
    log(f"Đã lưu kho {len(df):,} phim cho step 5")

    print("=" * 74)
    print(f"KẾT QUẢ TRÊN TẬP TEST ({len(y):,} phim chưa từng thấy khi huấn luyện)")
    print("=" * 74)
    show = res.copy()
    for c in ("RMSE", "MAE", "R2"):
        show[c] = show[c].map("{:.3f}".format)
    show["within1"] = (show["within1"] * 100).map("{:.1f}%".format)
    show["rmse_gain"] = (show["rmse_gain"] * 100).map("{:.1f}%".format)
    print(show.to_string(index=False))
    print("\n10 hệ số lớn nhất của M2:")
    print(coef.sort_values("abs", ascending=False).head(10)[["feature", "coef"]].round(3).to_string(index=False))
    print(f"\nXong. Dữ liệu cho giao diện đã lưu ở {out}/")


if __name__ == "__main__":
    main()