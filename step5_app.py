"""
Bước 5 (cuối): chương trình gợi ý phim chạy ngay trong terminal.

Luồng chạy:
  GIAI ĐOẠN 1  Bạn nhập thể loại, năm, thời lượng, ngôn ngữ, mô tả (bỏ trống nếu không muốn nhập)
               -> in 10 phim khớp nhất với mô tả của bạn.
  GIAI ĐOẠN 2  Bạn chọn phim nào THÍCH / KHÔNG THÍCH trong 10 phim đó
               -> hồi quy tuyến tính (Ridge) học khẩu vị của bạn từ các lựa chọn này, rồi gợi ý thêm 5 phim.

Chỉ đọc dữ liệu đã tạo ở step 1-4, không train lại.
Chạy:   python step5_recommend.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from lr_utils import *

AD = f"{ART}/app_data"


def ask(prompt, default=""):
    s = input(f"{prompt}: ").strip()
    return s if s else default


def parse_range(s):
    """'2010-2020' -> (2010, 2020); '2015' -> (2015, 2015); rỗng -> None"""
    s = s.replace(" ", "")
    if not s:
        return None
    try:
        if "-" in s:
            a, b = s.split("-", 1)
            return float(a), float(b)
        return float(s), float(s)
    except ValueError:
        print("   (không hiểu, bỏ qua tiêu chí này)")
        return None


def range_score(v, rng, scale):
    """1 nếu nằm trong khoảng mong muốn, càng xa càng giảm."""
    a, b = rng
    d = np.where(v < a, a - v, np.where(v > b, v - b, 0.0))
    d = np.where(np.isnan(v), scale, d)           # thiếu dữ liệu -> coi như hơi lệch
    return np.exp(-d / scale)


def pick_genres(s, known):
    low = {g.lower(): g for g in known}
    out = []
    for t in s.split(","):
        t = t.strip().lower()
        if t in low:
            out.append(low[t])
        elif t:
            print(f"   (không có thể loại '{t}', bỏ qua)")
    return out


def parse_nums(s, n):
    out = []
    for t in s.replace(",", " ").split():
        if t.isdigit() and 1 <= int(t) <= n:
            out.append(int(t) - 1)
    return list(dict.fromkeys(out))


def show(df, pred, rows, extra=None):
    for k, i in enumerate(rows, 1):
        r = df.iloc[i]
        yr = "?" if pd.isna(r.year) else int(r.year)
        rt = "?" if pd.isna(r.runtime) else f"{int(r.runtime)}p"
        print(f"{k:>2}. {r.title[:48]} ({yr}, {rt}, {r.lang})  Điểm thật {r.rating:.1f} | mô hình đoán {pred[i]:.1f}")
        print(f"      Thể loại: {', '.join(r.genres_list) or '?'}")
        if extra is not None:
            print(f"      {extra[k - 1]}")


def stage1(df, T, pipe, M, known_genres):
    print("\n--- Nhập sở thích (bỏ trống = không quan tâm) ---")
    print("Thể loại có sẵn:", ", ".join(known_genres))
    genres = pick_genres(ask("Thể loại (cách nhau dấu phẩy, tiếng Anh)"), known_genres)
    yr = parse_range(ask("Năm phát hành (vd 2010-2020 hoặc 1995)"))
    rt = parse_range(ask("Thời lượng phút (vd 90-120)"))
    lang = ask("Ngôn ngữ gốc (vd en, fr, ja, ko)").lower()
    desc = ask("Mô tả phim bạn thích (tiếng Anh, vd: space survival, friendship)")

    score, wsum = np.zeros(len(df)), 0.0
    if genres:
        gl = df["genres_list"].values
        cov = np.array([len(set(genres) & set(g)) / len(genres) for g in gl])
        score += 4.0 * cov; wsum += 4.0
    if yr:
        score += 1.5 * range_score(df["year"].values, yr, 10); wsum += 1.5
    if rt:
        score += 1.0 * range_score(df["runtime"].values, rt, 30); wsum += 1.0
    if lang:
        score += 1.5 * (df["lang"].values == lang); wsum += 1.5
    if desc:
        q = pipe["tfidf"].transform([desc]).toarray()[0].astype("float32")
        if q.any():
            score += 2.0 * (T @ q); wsum += 2.0
        else:
            print("   (mô tả không có từ khóa mô hình biết, bỏ qua)")
    if wsum == 0:
        print("Bạn chưa nhập gì nên mình gợi ý các phim điểm cao.")
        match = np.zeros(len(df))
    else:
        match = score / wsum
    final = match + 0.1 * df["rating"].values / 10      # điểm cao hơn thì xếp trước khi bằng độ khớp
    top = np.argsort(-final)[:10]

    # dự đoán điểm cho "phim lý tưởng" mà bạn vừa mô tả (dùng đúng mô hình hồi quy đã train)
    ideal = dict(title="", genres=", ".join(genres), year=(yr[0] + yr[1]) / 2 if yr else np.nan,
                 runtime=(rt[0] + rt[1]) / 2 if rt else np.nan, lang=lang or "en", text=desc)
    fr = pd.DataFrame([{"title": "", "genres_list": genres, "year": ideal["year"], "runtime": ideal["runtime"],
                        "lang": ideal["lang"], "text": desc}])
    p_ideal = float(np.clip(M["m2"].predict(build_matrix(fr, pipe))[0], 0, 10))
    return top, match, p_ideal


def stage2(df, X, pred, pipe, top, liked, disliked):
    """Hồi quy tuyến tính Ridge: x = đặc trưng phim, y = +1 thích / -1 không thích -> w là 'khẩu vị' của bạn."""
    ids = liked + disliked
    y = np.array([1.0] * len(liked) + [-1.0] * len(disliked))
    Xs = X[[top[i] for i in ids]]
    if len(liked) and len(disliked):
        m = Ridge(alpha=10.0).fit(Xs, y)
        w = m.coef_.copy()
    else:                                              # chỉ có một loại: dùng trung bình đặc trưng
        w = Xs.mean(axis=0) * (1.0 if liked else -1.0)
    taste = X @ w
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    final = z(taste) + 0.3 * z(df["rating"].values)    # khẩu vị là chính, thêm chút điểm chất lượng
    final[top] = -1e9                                  # không gợi ý lại 10 phim đã xem
    rec = np.argsort(-final)[:5]
    why = []
    for i in rec:
        c = w * np.maximum(X[i], 0)        # chỉ tính đặc trưng mà phim đó CÓ
        best = np.argsort(-c)[:3]
        why.append("Vì bạn có xu hướng thích: " + ", ".join(pipe["names"][j] for j in best if c[j] > 0))
    return rec, why, w


def main():
    try:
        df = pd.read_pickle(f"{ART}/movies.pkl")
        X = np.load(f"{AD}/catalog_X.npy")
        T = np.load(f"{AD}/catalog_T.npy")
        pred = np.load(f"{AD}/catalog_pred.npy")
        pipe, M = joblib.load(f"{ART}/pipeline.joblib"), joblib.load(f"{ART}/models.joblib")
        s = json.load(open(f"{AD}/summary.json", encoding="utf-8"))
    except FileNotFoundError as e:
        raise SystemExit(f"Thiếu file ({e.filename}). Hãy chạy đủ step 1 đến step 4 trước.")
    known = list(pipe["feats"].mlb.classes_)

    print("=" * 70)
    print(" CINEMATCH - gợi ý phim bằng hồi quy tuyến tính")
    print(f" Kho {len(df):,} phim | mô hình đoán điểm có RMSE test = {pd.read_csv(f'{AD}/results.csv').RMSE.iloc[-1]:.3f}")
    print("=" * 70)

    while True:
        top, match, p_ideal = stage1(df, T, pipe, M, known)
        print(f"\nMô hình hồi quy đoán một phim như bạn mô tả sẽ có điểm khoảng {p_ideal:.1f}/10.")
        print("\n=== GIAI ĐOẠN 1: 10 phim hợp với mô tả của bạn ===")
        show(df, pred, top, [f"Độ khớp với yêu cầu: {match[i] * 100:.0f}%" for i in top])

        print("\nChọn phim bạn THÍCH (gõ số, cách nhau dấu cách, vd: 1 3 4)")
        liked = parse_nums(ask("Phim thích"), 10)
        rest = [i for i in range(10) if i not in liked]
        dis_in = ask("Phim KHÔNG thích (Enter = tất cả phim còn lại)")
        disliked = [i for i in parse_nums(dis_in, 10) if i not in liked] if dis_in else rest
        if not liked and not disliked:
            print("Bạn chưa chọn phim nào.")
        else:
            rec, why, w = stage2(df, X, pred, pipe, top, liked, disliked)
            print(f"\n=== GIAI ĐOẠN 2: 5 phim gợi ý theo khẩu vị của bạn (dựa trên {len(liked)} thích, {len(disliked)} không thích) ===")
            show(df, pred, rec, why)
            like_f = np.argsort(-w)[:5]
            dis_f = np.argsort(w)[:3]
            print("\nMô hình học được khẩu vị của bạn:")
            print("  Thích:       ", ", ".join(pipe["names"][j] for j in like_f))
            print("  Không thích: ", ", ".join(pipe["names"][j] for j in dis_f))
        if ask("\nThử lại với sở thích khác? (y/n)", "n").lower() != "y":
            break
    print("Cảm ơn bạn!")


if __name__ == "__main__":
    main()