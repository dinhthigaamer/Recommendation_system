"""
Hệ Gợi Ý (Recommender Systems)
Thuật toán: Lọc Dựa Trên Nội Dung (Content-Based Filtering)
Dataset: Goodbooks-10k (books_mapped.jsonl & ratings.csv)
Môi trường thực thi: rs_env (Python 3.10, scikit-learn, pandas, numpy)

Các tính năng chính:
1. Trích xuất đặc trưng nội dung (Content Features):
   - Thể loại (Genres): phân loại chủ đề sách.
   - Tác giả (Authors): phong cách viết của tác giả.
   - Tóm tắt nội dung (Description): ngữ cảnh và tóm tắt tác phẩm.
   - Tên sách (Title): các từ khóa chính.
2. Biểu diễn đặc trưng văn bản bằng TF-IDF (Term Frequency - Inverse Document Frequency).
3. Đề xuất Item-to-Item: Tìm các cuốn sách tương đồng nhất với một cuốn sách cho trước (Cosine Similarity).
4. Đề xuất Cá nhân hóa User-to-Item:
   - Xây dựng Hồ sơ người dùng (User Profile Vector) dựa trên lịch sử đánh giá có trọng số.
   - Dự đoán điểm và gợi ý Top-N cuốn sách phù hợp nhất chưa đọc.
5. Đánh giá mô hình: Đo lường RMSE, MAE và Precision@K trên tập kiểm thử (Test set).
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def find_data_path(filename: str) -> str:
    """Tìm đường dẫn tới file dữ liệu một cách linh hoạt trong thư mục hiện tại hoặc goodbooks-10k."""
    candidates = [
        filename,
        os.path.join("goodbooks-10k", filename),
        os.path.join("..", "goodbooks-10k", filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), "goodbooks-10k", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Không tìm thấy file '{filename}'. Vui lòng đảm bảo file tồn tại trong thư mục dự án!"
    )


def load_books_metadata() -> pd.DataFrame:
    """
    Tải dữ liệu sách: Ưu tiên file books_mapped.jsonl (đầy đủ description & genres),
    nếu không có sẽ tự động tìm books_data_en.jsonl hoặc books.csv.
    """
    mapped_candidates = [
        "books_mapped.jsonl",
        "books_data_en.jsonl",
        os.path.join(os.path.dirname(__file__), "books_mapped.jsonl"),
        os.path.join(os.path.dirname(__file__), "books_data_en.jsonl"),
    ]

    jsonl_path = None
    for cand in mapped_candidates:
        if os.path.exists(cand):
            jsonl_path = cand
            break

    if jsonl_path:
        print(f"[*] Đang tải dữ liệu thuộc tính sách từ: {jsonl_path}")
        books = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    books.append(json.loads(line))
        df_books = pd.DataFrame(books)

        # Chuẩn hóa trường id và book_id
        if "id" in df_books.columns:
            df_books["book_id"] = df_books["id"].astype(int)
        elif "book_id" in df_books.columns:
            df_books["book_id"] = df_books["book_id"].astype(int)
            df_books["id"] = df_books["book_id"]
    else:
        # Dự phòng bằng books.csv nếu chưa có file jsonl
        csv_path = find_data_path("books.csv")
        print(f"[!] Không thấy file JSONL, tải từ CSV: {csv_path}")
        df_books = pd.read_csv(csv_path)
        df_books["id"] = df_books["book_id"]
        df_books["genres"] = ""
        df_books["description"] = ""

    print(f"    - Đã tải {len(df_books):,} cuốn sách vào bộ nhớ.")
    return df_books


def load_ratings_dataset(
    num_users: int = 1500, num_books: int = 2000, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tải dữ liệu ratings và chia thành tập huấn luyện (Train) / kiểm thử (Test).
    """
    ratings_file = find_data_path("ratings.csv")
    print(f"[*] Đang tải dữ liệu đánh giá (ratings) từ: {ratings_file}")
    df_ratings = pd.read_csv(ratings_file)
    print(f"    - Tổng bản ghi gốc: {len(df_ratings):,} đánh giá")

    # Lọc top người dùng tích cực và top sách có nhiều đánh giá để tối ưu bộ nhớ & thời gian
    top_user_ids = df_ratings["user_id"].value_counts().head(num_users).index
    top_book_ids = df_ratings["book_id"].value_counts().head(num_books).index

    df_filtered = df_ratings[
        df_ratings["user_id"].isin(top_user_ids)
        & df_ratings["book_id"].isin(top_book_ids)
    ].copy()

    n_u = df_filtered["user_id"].nunique()
    n_b = df_filtered["book_id"].nunique()
    n_r = len(df_filtered)
    sparsity = 1.0 - (n_r / (n_u * n_b))

    print(f"[*] Dữ liệu ratings sau khi chọn mẫu ({num_users} users, {num_books} books):")
    print(f"    - Số người dùng (Users): {n_u:,}")
    print(f"    - Số cuốn sách (Books):  {n_b:,}")
    print(f"    - Số lượt đánh giá:      {n_r:,}")
    print(f"    - Độ thưa (Sparsity):    {sparsity * 100:.2f}%")

    train_df, test_df = train_test_split(df_filtered, test_size=test_size, random_state=42)
    print(f"[*] Phân chia tập dữ liệu: Train = {len(train_df):,} | Test = {len(test_df):,}")

    return train_df, test_df, df_filtered


class ContentBasedFiltering:
    """
    Hệ thống gợi ý dựa trên nội dung (Content-Based Recommender).
    - Biểu diễn sách bằng TF-IDF từ: Tên sách + Tác giả + Thể loại + Tóm tắt mô tả.
    - Xây dựng User Profile Vector từ lịch sử đánh giá có trọng số.
    - Tính toán độ tương đồng Cosine Similarity để gợi ý sách.
    """

    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )

        self.df_books: pd.DataFrame = pd.DataFrame()
        self.tfidf_matrix = None
        self.book_id_to_idx: Dict[int, int] = {}
        self.idx_to_book_id: Dict[int, int] = {}

        # Hồ sơ người dùng: user_id -> vector TF-IDF 1D
        self.user_profiles: Dict[int, np.ndarray] = {}
        # Thống kê người dùng: user_id -> (mean_rating, std_rating)
        self.user_stats: Dict[int, Tuple[float, float]] = {}
        self.global_mean: float = 3.0

    @staticmethod
    def _create_content_soup(row: pd.Series) -> str:
        """Kết hợp các thuộc tính nội dung thành một chuỗi văn bản giàu ngữ nghĩa."""
        # 1. Thể loại (genres): loại bỏ dấu cách để tạo thành các token chủ đề duy nhất
        raw_genres = row.get("genres") or []
        if isinstance(raw_genres, str):
            raw_genres = [g.strip() for g in raw_genres.split(",") if g.strip()]
        genres_tokens = [
            re.sub(r"[^\w]", "", g.lower())
            for g in raw_genres
            if g not in ("...more", "...less", "more", "less")
        ]
        genres_str = " ".join(genres_tokens)

        # 2. Tác giả (authors): gộp họ tên tác giả bằng dấu gạch dưới (vd: J_K_Rowling)
        raw_authors = row.get("authors") or []
        if isinstance(raw_authors, str):
            raw_authors = [a.strip() for a in raw_authors.split(",") if a.strip()]
        authors_tokens = [
            re.sub(r"[^\w]", "_", a.strip().lower()) for a in raw_authors if a.strip()
        ]
        authors_str = " ".join(authors_tokens)

        # 3. Tiêu đề và mô tả
        title_str = str(row.get("title") or "").strip()
        desc_str = str(row.get("description") or "").strip()

        # Nhấn mạnh thể loại và tác giả bằng cách lặp lại (tăng trọng số TF)
        soup = f"{title_str} {authors_str} {authors_str} {genres_str} {genres_str} {genres_str} {desc_str}"
        return soup

    def fit_items(self, df_books: pd.DataFrame) -> "ContentBasedFiltering":
        """Xây dựng ma trận TF-IDF đặc trưng nội dung cho toàn bộ danh mục sách."""
        start_time = time.time()
        print("\n[*] Đang trích xuất đặc trưng nội dung và xây dựng ma trận TF-IDF...")

        self.df_books = df_books.copy().reset_index(drop=True)
        self.book_id_to_idx = {int(row["book_id"]): i for i, row in self.df_books.iterrows()}
        self.idx_to_book_id = {i: int(row["book_id"]) for i, row in self.df_books.iterrows()}

        # Tạo chuỗi soup cho từng cuốn sách
        soups = self.df_books.apply(self._create_content_soup, axis=1)

        # Huấn luyện TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(soups)
        elapsed = time.time() - start_time
        print(
            f"[+] Xây dựng ma trận TF-IDF hoàn tất ({elapsed:.2f}s)! Kích thước ma trận: {self.tfidf_matrix.shape}"
        )
        return self

    def fit_users(self, df_train: pd.DataFrame) -> "ContentBasedFiltering":
        """
        Xây dựng hồ sơ người dùng (User Profile Vectors) từ tập dữ liệu huấn luyện:
        Vector hồ sơ người dùng là trung bình có trọng số của các vector sách đã đọc:
            W_u = sum_i ( (R_ui - mean_R_u) * V_i )
        Sau đó chuẩn hóa L2 về vector đơn vị.
        """
        start_time = time.time()
        print("\n[*] Đang xây dựng hồ sơ sở thích (User Profiles) cho từng người dùng...")

        self.global_mean = float(df_train["rating"].mean())
        self.user_profiles.clear()
        self.user_stats.clear()

        n_features = self.tfidf_matrix.shape[1]

        for user_id, group in df_train.groupby("user_id"):
            u_mean = float(group["rating"].mean())
            u_std = float(group["rating"].std())
            if np.isnan(u_std) or u_std == 0:
                u_std = 1.0
            self.user_stats[int(user_id)] = (u_mean, u_std)

            # Khởi tạo vector hồ sơ người dùng
            user_vec = np.zeros(n_features, dtype=np.float32)

            for row in group.itertuples():
                bid = int(row.book_id)
                if bid in self.book_id_to_idx:
                    idx = self.book_id_to_idx[bid]
                    # Trọng số: điểm lệch chuẩn so với điểm trung bình của chính user đó
                    weight = float(row.rating) - u_mean
                    book_vec = self.tfidf_matrix[idx].toarray().flatten()
                    user_vec += weight * book_vec

            # Chuẩn hóa L2 norm
            norm = np.linalg.norm(user_vec)
            if norm > 0:
                user_vec /= norm

            self.user_profiles[int(user_id)] = user_vec

        elapsed = time.time() - start_time
        print(f"[+] Đã xây dựng hồ sơ cho {len(self.user_profiles):,} người dùng ({elapsed:.2f}s)!")
        return self

    def recommend_similar_books(
        self,
        book_id: Optional[int] = None,
        book_title: Optional[str] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Gợi ý sách tương tự (Item-to-Item Content-Based):
        Tính độ tương đồng Cosine Similarity giữa cuốn sách mục tiêu và tất cả các cuốn sách khác.
        """
        target_idx = None

        if book_id is not None and int(book_id) in self.book_id_to_idx:
            target_idx = self.book_id_to_idx[int(book_id)]
        elif book_title:
            # Tìm kiếm gần đúng theo tên sách
            query = book_title.strip().lower()
            matches = self.df_books[
                self.df_books["title"].str.lower().str.contains(query, regex=False, na=False)
            ]
            if not matches.empty:
                target_idx = matches.index[0]
            else:
                print(f"[!] Không tìm thấy cuốn sách nào khớp với từ khóa: '{book_title}'")
                return pd.DataFrame()

        if target_idx is None:
            print(f"[!] Không tìm thấy cuốn sách chỉ định (book_id={book_id})")
            return pd.DataFrame()

        target_book = self.df_books.iloc[target_idx]
        target_vec = self.tfidf_matrix[target_idx : target_idx + 1]

        # Tính toán Cosine Similarity
        sim_scores = cosine_similarity(target_vec, self.tfidf_matrix).flatten()

        # Lấy Top N cuốn sách tương đồng nhất (bỏ qua chính cuốn sách đó)
        top_indices = sim_scores.argsort()[::-1]
        top_indices = [idx for idx in top_indices if idx != target_idx][:top_n]

        results = []
        for idx in top_indices:
            row = self.df_books.iloc[idx].to_dict()
            row["similarity_score"] = float(sim_scores[idx])
            results.append(row)

        df_rec = pd.DataFrame(results)
        return df_rec

    def predict_rating(self, user_id: int, book_id: int) -> float:
        """Dự đoán điểm số người dùng sẽ chấm cho 1 cuốn sách dựa trên độ tương đồng nội dung."""
        if user_id not in self.user_profiles or book_id not in self.book_id_to_idx:
            return self.global_mean

        user_vec = self.user_profiles[user_id]
        book_idx = self.book_id_to_idx[book_id]
        u_mean, u_std = self.user_stats.get(user_id, (self.global_mean, 1.0))

        # Tính Cosine Similarity giữa hồ sơ người dùng và cuốn sách
        sim = float(self.tfidf_matrix[book_idx].dot(user_vec).item())

        # Công thức hồi quy điểm số: Điểm = Mean + Sim * Std * Scale
        pred = u_mean + (sim * u_std * 2.0)
        return float(np.clip(pred, 1.0, 5.0))

    def recommend_for_user(
        self,
        user_id: int,
        df_ratings_history: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Gợi ý sách cá nhân hóa cho một người dùng:
        - So khớp User Profile Vector với toàn bộ sách trong hệ thống.
        - Lọc bỏ các cuốn sách người dùng đã đọc/đánh giá.
        - Xếp hạng theo độ tương đồng và điểm dự đoán.
        """
        if user_id not in self.user_profiles:
            print(f"[!] Người dùng ID {user_id} chưa có dữ liệu huấn luyện.")
            return pd.DataFrame()

        user_vec = self.user_profiles[user_id]
        u_mean, u_std = self.user_stats.get(user_id, (self.global_mean, 1.0))

        # 1. Tính toán điểm tương đồng cho toàn bộ 10,000 cuốn sách
        sim_scores = self.tfidf_matrix.dot(user_vec).flatten()

        # 2. Loại bỏ các sách người dùng đã đánh giá
        rated_book_ids = set(
            df_ratings_history[df_ratings_history["user_id"] == user_id]["book_id"]
        )
        for bid in rated_book_ids:
            if bid in self.book_id_to_idx:
                idx = self.book_id_to_idx[bid]
                sim_scores[idx] = -999.0

        # 3. Lấy Top-N vị trí có điểm cao nhất
        top_indices = sim_scores.argsort()[::-1][:top_n]

        results = []
        for idx in top_indices:
            bid = self.idx_to_book_id[idx]
            sim = float(sim_scores[idx])
            pred_rating = np.clip(u_mean + sim * u_std * 2.0, 1.0, 5.0)

            row = self.df_books.iloc[idx].to_dict()
            row["similarity_score"] = sim
            row["predicted_rating"] = float(pred_rating)
            results.append(row)

        return pd.DataFrame(results)

    def evaluate(
        self, df_test: pd.DataFrame, sample_size: int = 3000, top_k: int = 10
    ) -> Dict[str, float]:
        """
        Đánh giá hiệu năng mô hình trên tập kiểm thử (Test Set):
        - RMSE (Root Mean Squared Error): Đo độ lệch căn bậc hai trung bình.
        - MAE (Mean Absolute Error): Đo độ lệch tuyệt đối trung bình.
        - Precision@K và Recall@K: Đánh giá chất lượng xếp hạng gợi ý cho sách được yêu thích (rating >= 4.0).
        """
        print(f"\n{'='*70}")
        print(f"ĐÁNH GIÁ MÔ HÌNH CONTENT-BASED FILTERING TRÊN TẬP TEST")
        print(f"{'='*70}")

        test_sample = df_test.head(sample_size).copy()
        y_true = []
        y_pred = []

        for row in test_sample.itertuples():
            u = int(row.user_id)
            b = int(row.book_id)
            if u in self.user_profiles and b in self.book_id_to_idx:
                pred = self.predict_rating(u, b)
                y_true.append(float(row.rating))
                y_pred.append(pred)

        if not y_true:
            print("[!] Không có mẫu kiểm thử hợp lệ.")
            return {}

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        print(f"[*] Số lượng mẫu test đánh giá: {len(y_true):,} lượt đánh giá")
        print(f"    - RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"    - MAE  (Mean Absolute Error):     {mae:.4f}")

        # Đánh giá Ranking Metrics (Precision@K và Recall@K) trên nhóm User mẫu
        eval_users = list(set(test_sample["user_id"]))[:100]
        precisions = []
        recalls = []

        for u in eval_users:
            u_test = test_sample[test_sample["user_id"] == u]
            # Các sách mà user thực sự thích trong tập test (rating >= 4.0)
            relevant_items = set(u_test[u_test["rating"] >= 4.0]["book_id"])
            if not relevant_items:
                continue

            # Sinh danh sách dự đoán cho các sách trong tập test của user
            candidate_books = list(u_test["book_id"])
            scored_candidates = []
            for b in candidate_books:
                score = self.predict_rating(u, b)
                scored_candidates.append((b, score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_rec = [b for b, _ in scored_candidates[:top_k]]

            hits = len(set(top_rec) & relevant_items)
            precisions.append(hits / min(top_k, len(top_rec)))
            recalls.append(hits / len(relevant_items))

        avg_precision = float(np.mean(precisions)) if precisions else 0.0
        avg_recall = float(np.mean(recalls)) if recalls else 0.0

        print(f"    - Precision@{top_k}:                   {avg_precision * 100:.2f}%")
        print(f"    - Recall@{top_k}:                      {avg_recall * 100:.2f}%")
        print(f"{'='*70}\n")

        return {
            "rmse": rmse,
            "mae": mae,
            f"precision@{top_k}": avg_precision,
            f"recall@{top_k}": avg_recall,
        }


def print_user_history(
    user_id: int, df_ratings: pd.DataFrame, df_books: pd.DataFrame, top_n: int = 5
) -> None:
    """Hiển thị lịch sử các cuốn sách người dùng đã đánh giá cao nhất."""
    user_ratings = df_ratings[df_ratings["user_id"] == user_id]
    if user_ratings.empty:
        print(f"Không tìm thấy lịch sử đánh giá của user_id = {user_id}")
        return

    merged = user_ratings.merge(
        df_books[["book_id", "title", "authors", "rating_value", "genres"]],
        on="book_id",
        how="left",
    ).sort_values(by="rating", ascending=False)

    print(f"\n{'='*75}")
    print(f"LỊCH SỬ ĐỌC CỦA USER {user_id} (TOP {top_n} SÁCH ĐƯỢC CHẤM CAO NHẤT):")
    print(f"{'='*75}")
    for idx, row in enumerate(merged.head(top_n).itertuples(), start=1):
        authors_display = (
            ", ".join(row.authors) if isinstance(row.authors, list) else str(row.authors or "N/A")
        )
        genres_display = (
            ", ".join(row.genres[:4]) if isinstance(row.genres, list) else str(row.genres or "")
        )
        avg_r = getattr(row, "rating_value", "N/A")
        print(
            f"{idx}. {row.title}\n"
            f"   Tác giả: {authors_display} | Thể loại: [{genres_display}]\n"
            f"   Điểm User chấm: {row.rating} sao | Goodreads TB: {avg_r}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Hệ Thống Gợi Ý Sách Sử Dụng Thuật Toán Lọc Dựa Trên Nội Dung (Content-Based Filtering)"
    )
    parser.add_argument(
        "--user_id",
        type=int,
        default=None,
        help="Mã user_id muốn thực hiện gợi ý cá nhân hóa (mặc định: chọn user có nhiều đánh giá)",
    )
    parser.add_argument(
        "--book_id",
        type=int,
        default=None,
        help="Mã book_id muốn tìm các cuốn sách tương đồng (ví dụ: 1 cho The Hunger Games)",
    )
    parser.add_argument(
        "--book_title",
        type=str,
        default=None,
        help="Tên cuốn sách muốn tìm tương đồng (ví dụ: 'The Hobbit', 'Harry Potter')",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Số lượng sách gợi ý (mặc định: 10)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=1500,
        help="Số lượng người dùng tích cực để lấy mẫu huấn luyện (mặc định: 1500)",
    )
    parser.add_argument(
        "--books",
        type=int,
        default=2000,
        help="Số lượng sách phổ biến để lấy mẫu huấn luyện (mặc định: 2000)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Thực hiện đánh giá mô hình trên tập kiểm thử (RMSE, MAE, Precision@K)",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=3000,
        help="Số lượng mẫu test dùng để tính độ lỗi RMSE/MAE (mặc định: 3000)",
    )
    args = parser.parse_args()

    print("=" * 75)
    print(" HỆ THỐNG GỢI Ý SÁCH - CONTENT-BASED FILTERING (LỌC DỰA TRÊN NỘI DUNG)")
    print("=" * 75)

    # 1. Tải metadata sách (ưu tiên books_mapped.jsonl có đầy đủ genres & description)
    df_books = load_books_metadata()

    # 2. Tải và phân chia tập dữ liệu ratings
    train_df, test_df, df_ratings = load_ratings_dataset(
        num_users=args.users, num_books=args.books
    )

    # 3. Khởi tạo và huấn luyện mô hình Content-Based Filtering
    cb_model = ContentBasedFiltering(max_features=10000, ngram_range=(1, 2))
    cb_model.fit_items(df_books)
    cb_model.fit_users(train_df)

    # 4. Đánh giá nếu có cờ --eval hoặc chạy mặc định
    if args.eval:
        cb_model.evaluate(test_df, sample_size=args.test_samples, top_k=args.top_n)

    # 5. Thực nghiệm 1: Gợi ý sách tương tự (Item-to-Item Content-Based)
    query_id = args.book_id if args.book_id is not None else 1  # Mặc định The Hunger Games (id=1)
    target_book_row = df_books[df_books["book_id"] == query_id]
    book_title_display = target_book_row.iloc[0]["title"] if not target_book_row.empty else f"ID {query_id}"

    if args.book_title:
        book_title_display = f"từ khóa '{args.book_title}'"

    print(f"\n{'='*75}")
    print(f"DANH SÁCH TOP {args.top_n} SÁCH CÓ NỘI DUNG TƯƠNG ĐỒNG VỚI: {book_title_display}")
    print(f"{'='*75}")

    similar_books = cb_model.recommend_similar_books(
        book_id=args.book_id if args.book_id else (None if args.book_title else query_id),
        book_title=args.book_title,
        top_n=args.top_n,
    )

    if not similar_books.empty:
        for idx, row in enumerate(similar_books.itertuples(), start=1):
            authors_str = (
                ", ".join(row.authors) if isinstance(row.authors, list) else str(row.authors or "N/A")
            )
            genres_str = (
                ", ".join(row.genres[:4]) if isinstance(row.genres, list) else str(row.genres or "")
            )
            desc_short = (
                str(row.description)[:110] + "..." if row.description else "Không có mô tả."
            )
            print(
                f"{idx:2d}. {row.title} (ID: {row.book_id})\n"
                f"    - Tác giả: {authors_str} | Thể loại: [{genres_str}]\n"
                f"    - Độ tương đồng Cosine: {row.similarity_score:.4f}\n"
                f"    - Tóm tắt: {desc_short}"
            )
    else:
        print("[!] Không tìm thấy sách tương tự.")

    # 6. Thực nghiệm 2: Gợi ý cá nhân hóa cho một người dùng (User Profile Content-Based)
    if args.user_id is not None and args.user_id in cb_model.user_profiles:
        target_user = args.user_id
    else:
        # Chọn một user có nhiều tương tác trong tập train để kết quả sinh động
        target_user = int(train_df["user_id"].value_counts().index[2])

    print_user_history(target_user, df_ratings, df_books, top_n=5)

    print(f"\n{'='*75}")
    print(f"DANH SÁCH TOP {args.top_n} SÁCH GỢI Ý CÁ NHÂN HÓA CHO USER {target_user}:")
    print(f"{'='*75}")

    user_recommendations = cb_model.recommend_for_user(
        user_id=target_user, df_ratings_history=train_df, top_n=args.top_n
    )

    if not user_recommendations.empty:
        for idx, row in enumerate(user_recommendations.itertuples(), start=1):
            authors_str = (
                ", ".join(row.authors) if isinstance(row.authors, list) else str(row.authors or "N/A")
            )
            genres_str = (
                ", ".join(row.genres[:4]) if isinstance(row.genres, list) else str(row.genres or "")
            )
            avg_r = getattr(row, "rating_value", "N/A")
            print(
                f"{idx:2d}. {row.title} (ID: {row.book_id})\n"
                f"    - Tác giả: {authors_str} | Thể loại: [{genres_str}]\n"
                f"    - Điểm dự đoán: {row.predicted_rating:.2f} sao | Độ tương đồng sở thích: {row.similarity_score:.4f} | Điểm TB: {avg_r}"
            )
    else:
        print("[!] Không thể tạo danh sách gợi ý cho người dùng này.")

    print(f"\n{'='*75}")
    print("[+] Hoàn thành thực thi chương trình lọc dựa trên nội dung!")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
