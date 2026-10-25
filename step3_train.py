"""
Bước 3: HUẤN LUYỆN hồi quy tuyến tính.

Công thức:  điểm = b0 + b1*x1 + b2*x2 + ...   (x = các đặc trưng, b = hệ số mô hình tự học)
Huấn luyện = tìm b sao cho tổng bình phương sai số trên tập train nhỏ nhất.

3 mô hình để so sánh:
  M0  Baseline: luôn đoán điểm trung bình
  M1  Hồi quy tuyến tính chỉ dùng đặc trưng bảng (thể loại, năm, thời lượng, ngôn ngữ)
  M2  Hồi quy tuyến tính dùng thêm 300 từ khóa trong mô tả

Chạy:  python step3_train.py
Đầu ra: lr_artifacts/models.joblib
"""
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from lr_utils import *


def main():
    d = np.load(f"{ART}/data.npz")
    pipe = joblib.load(f"{ART}/pipeline.joblib")
    nd = pipe["n_dense"]
    X_tr, y_tr, X_va, y_va = d["X_tr"], d["y_tr"], d["X_va"], d["y_va"]
    log(f"Train {len(y_tr):,} phim | Validation {len(y_va):,} phim")

    base = float(y_tr.mean())
    log(f"M0 Baseline: luôn đoán {base:.3f}")
    print(f"    RMSE validation = {metrics(y_va, np.full(len(y_va), base))['RMSE']:.4f}")

    log("M1 Hồi quy tuyến tính (đặc trưng bảng)")
    m1 = LinearRegression().fit(X_tr[:, :nd], y_tr)
    print(f"    RMSE validation = {metrics(y_va, m1.predict(X_va[:, :nd]))['RMSE']:.4f}  | hệ số b0 = {m1.intercept_:.3f}")

    log("M2 Hồi quy tuyến tính (đặc trưng bảng + từ khóa)")
    m2 = LinearRegression().fit(X_tr, y_tr)
    print(f"    RMSE validation = {metrics(y_va, m2.predict(X_va))['RMSE']:.4f}  | hệ số b0 = {m2.intercept_:.3f}")

    joblib.dump(dict(base=base, m1=m1, m2=m2), f"{ART}/models.joblib")
    print(f"\nXong. Đã lưu mô hình vào {ART}/models.joblib")
    print("RMSE validation càng thấp càng tốt. Kết quả cuối cùng (tập test) nằm ở step 4.")


if __name__ == "__main__":
    main()
