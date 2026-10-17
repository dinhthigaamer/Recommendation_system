"""
Hệ Gợi Ý (Recommender Systems)
Thuật toán: User-Based Collaborative Filtering (User-User CF)
Dataset: Goodbooks-10k (ratings.csv & books.csv)
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def find_data_path(filename: str) -> str:
    """Tìm đường dẫn tới file dữ liệu một cách linh hoạt."""
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
        f"Không tìm thấy file '{filename}'. Vui lòng đảm bảo thư mục goodbooks-10k tồn tại!"
    )


def load_dataset(
    num_users: int = 1500, num_books: int = 2000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tải và lọc tập dữ liệu con (Active Users & Popular Books) để tối ưu hiệu năng tính toán.
    """
    ratings_file = find_data_path("ratings.csv")
    books_file = find_data_path("books.csv")

    print(f"[*] Đang tải dữ liệu ratings từ: {ratings_file}")
    df_ratings = pd.read_csv(ratings_file)

    print(f"[*] Đang tải thông tin sách từ:   {books_file}")
    df_books = pd.read_csv(books_file)

    print(
        f"    - Tổng bản ghi gốc: {len(df_ratings):,} ratings | {len(df_books):,} books"
    )

    # Lọc top người dùng tích cực và top sách có nhiều đánh giá nhất
    top_user_ids = (
        df_ratings["user_id"].value_counts().head(num_users).index
    )
    top_book_ids = (
        df_ratings["book_id"].value_counts().head(num_books).index
    )

    df_filtered = df_ratings[
        df_ratings["user_id"].isin(top_user_ids)
        & df_ratings["book_id"].isin(top_book_ids)
    ].copy()

    n_u = df_filtered["user_id"].nunique()
    n_b = df_filtered["book_id"].nunique()
    n_r = len(df_filtered)
    sparsity = 1.0 - (n_r / (n_u * n_b))

    print(f"[*] Dữ liệu sau khi lọc:")
    print(f"    - Số người dùng (Users): {n_u:,}")
    print(f"    - Số cuốn sách (Books):  {n_b:,}")
    print(f"    - Số lượt đánh giá:      {n_r:,}")
    print(f"    - Độ thưa (Sparsity):    {sparsity * 100:.2f}%")

    # Chia tập dữ liệu thành 80% Train và 20% Test
    train_df, test_df = train_test_split(
        df_filtered, test_size=0.2, random_state=42
    )
    print(
        f"[*] Chia tập dữ liệu: Train = {len(train_df):,} dòng | Test = {len(test_df):,} dòng"
    )

    return train_df, test_df, df_books, df_ratings


class UserBasedCollaborativeFiltering:
    """
    Mô hình Lọc cộng tác dựa trên người dùng (User-Based Collaborative Filtering).
    Bao gồm:
    - Chuẩn hóa điểm đánh giá (Mean-centering) để triệt tiêu định kiến người dùng.
    - Tính toán độ tương đồng Cosine Similarity (tương đương Pearson Correlation).
    - Dự đoán điểm dựa trên Top-K láng giềng có độ tương đồng cao nhất.
    """

    def __init__(self, k_neighbors: int = 25, min_similarity: float = 0.0):
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity

        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.book2idx: Dict[int, int] = {}
        self.idx2book: Dict[int, int] = {}

        self.user_means: np.ndarray = np.array([])
        self.global_mean: float = 3.0
        self.R: np.ndarray = np.array([])
        self.R_norm: np.ndarray = np.array([])
        self.user_similarity: np.ndarray = np.array([])

    def fit(self, df_train: pd.DataFrame) -> "UserBasedCollaborativeFiltering":
        """Huấn luyện mô hình từ DataFrame gồm: user_id, book_id, rating."""
        start_time = time.time()
        print("\n[*] Đang huấn luyện mô hình User-Based CF...")

        unique_users = sorted(df_train["user_id"].unique())
        unique_books = sorted(df_train["book_id"].unique())

        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for i, u in enumerate(unique_users)}
        self.book2idx = {b: i for i, b in enumerate(unique_books)}
        self.idx2book = {i: b for i, b in enumerate(unique_books)}

        n_u = len(unique_users)
        n_b = len(unique_books)
        self.global_mean = float(df_train["rating"].mean())

        # 1. Tạo ma trận tương tác User-Item ban đầu
        self.R = np.zeros((n_u, n_b), dtype=np.float32)
        for row in df_train.itertuples():
            if row.user_id in self.user2idx and row.book_id in self.book2idx:
                u = self.user2idx[row.user_id]
                b = self.book2idx[row.book_id]
                self.R[u, b] = row.rating

        # 2. Tính điểm trung bình của mỗi người dùng (Mean rating)
        counts = np.count_nonzero(self.R, axis=1)
        sums = np.sum(self.R, axis=1)
        self.user_means = np.divide(
            sums,
            counts,
            out=np.full(n_u, self.global_mean, dtype=float),
            where=counts > 0,
        )

        # 3. Chuẩn hóa ma trận (Mean-centered)
        # Điểm chuẩn hóa = r_{u, i} - mean_u
        self.R_norm = np.zeros_like(self.R)
        for u in range(n_u):
            rated_indices = np.where(self.R[u, :] > 0)[0]
            self.R_norm[u, rated_indices] = (
                self.R[u, rated_indices] - self.user_means[u]
            )

        # 4. Tính toán ma trận độ tương đồng Cosine giữa các người dùng
        self.user_similarity = cosine_similarity(self.R_norm)
        # Đặt độ tương đồng với chính mình bằng 0 để không tự làm láng giềng
        np.fill_diagonal(self.user_similarity, 0.0)

        elapsed = time.time() - start_time
        print(f"    - Hoàn tất trong {elapsed:.2f} giây.")
        print(
            f"    - Kích thước ma trận tương đồng người dùng: {self.user_similarity.shape}"
        )
        print(f"    - Điểm trung bình toàn cục (Global mean): {self.global_mean:.2f}")
        return self

    def predict_rating(self, user_id: int, book_id: int) -> float:
        """
        Dự đoán điểm số mà user_id sẽ dành cho book_id.
        Công thức:
            r_hat(u, i) = mean_u + sum(sim(u, v) * (r_{v, i} - mean_v)) / sum(|sim(u, v)|)
        """
        # Nếu user chưa xuất hiện trong tập huấn luyện (Cold-start user)
        if user_id not in self.user2idx:
            return self.global_mean

        u = self.user2idx[user_id]
        u_mean = self.user_means[u]

        # Nếu sách chưa xuất hiện trong tập huấn luyện (Cold-start book)
        if book_id not in self.book2idx:
            return float(u_mean)

        b = self.book2idx[book_id]

        # Tìm những người dùng khác đã từng đánh giá sách b
        other_users = np.where(self.R[:, b] > 0)[0]
        other_users = other_users[other_users != u]

        if len(other_users) == 0:
            return float(u_mean)

        # Lấy độ tương đồng giữa u và những người dùng này
        sims = self.user_similarity[u, other_users]

        # Chỉ lọc những người có độ tương đồng dương (> min_similarity)
        valid_mask = sims > self.min_similarity
        if not np.any(valid_mask):
            return float(u_mean)

        filtered_users = other_users[valid_mask]
        filtered_sims = sims[valid_mask]

        # Chọn Top-K láng giềng có độ tương đồng lớn nhất
        if len(filtered_sims) > self.k_neighbors:
            top_k_idx = np.argsort(filtered_sims)[-self.k_neighbors :]
            filtered_users = filtered_users[top_k_idx]
            filtered_sims = filtered_sims[top_k_idx]

        # Độ lệch chuẩn hóa của các láng giềng: (r_{v, i} - mean_v)
        norm_ratings = self.R_norm[filtered_users, b]

        sim_sum = np.sum(np.abs(filtered_sims))
        if sim_sum == 0:
            return float(u_mean)

        # Dự đoán theo công thức trung bình có trọng số
        pred = u_mean + (np.sum(filtered_sims * norm_ratings) / sim_sum)

        # Giới hạn trong khoảng thang điểm hợp lệ [1.0, 5.0]
        return float(np.clip(pred, 1.0, 5.0))

    def evaluate(
        self, test_df: pd.DataFrame, sample_size: Optional[int] = 3000
    ) -> Tuple[float, float]:
        """Đo lường sai số dự đoán trên tập test bằng RMSE và MAE."""
        if sample_size and sample_size < len(test_df):
            test_subset = test_df.sample(n=sample_size, random_state=42)
        else:
            test_subset = test_df

        actuals = []
        predictions = []

        start_time = time.time()
        for row in test_subset.itertuples():
            pred = self.predict_rating(row.user_id, row.book_id)
            actuals.append(row.rating)
            predictions.append(pred)

        eval_time = time.time() - start_time
        rmse = math.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        print(f"\n[*] Kết quả đánh giá trên {len(test_subset):,} mẫu test:")
        print(f"    - Thời gian đánh giá: {eval_time:.2f} giây")
        print(f"    - RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"    - MAE  (Mean Absolute Error):     {mae:.4f}")
        return rmse, mae

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        df_books: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Gợi ý Top-N cuốn sách phù hợp nhất mà user_id chưa từng đọc.
        """
        if user_id not in self.user2idx:
            print(f"[!] User {user_id} không tồn tại trong dữ liệu huấn luyện!")
            return pd.DataFrame()

        u = self.user2idx[user_id]
        # Tập các sách user đã đọc/đánh giá
        rated_book_indices = set(np.where(self.R[u, :] > 0)[0])

        # Dự đoán điểm cho tất cả các sách chưa đọc
        unrated_indices = [
            b for b in range(len(self.idx2book)) if b not in rated_book_indices
        ]

        scores = []
        for b in unrated_indices:
            book_id = self.idx2book[b]
            pred = self.predict_rating(user_id, book_id)
            scores.append((book_id, pred))

        # Sắp xếp giảm dần theo điểm dự đoán
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_n]

        rec_df = pd.DataFrame(
            top_scores, columns=["book_id", "predicted_rating"]
        )

        if df_books is not None:
            book_info_cols = [
                "book_id",
                "title",
                "authors",
                "original_publication_year",
                "average_rating",
            ]
            valid_cols = [c for c in book_info_cols if c in df_books.columns]
            rec_df = rec_df.merge(df_books[valid_cols], on="book_id", how="left")

        return rec_df


def print_user_history(
    user_id: int,
    df_ratings: pd.DataFrame,
    df_books: pd.DataFrame,
    top_n: int = 5,
):
    """Hiển thị các cuốn sách người dùng đã đánh giá cao nhất."""
    user_ratings = df_ratings[df_ratings["user_id"] == user_id]
    if user_ratings.empty:
        print(f"Không tìm thấy lịch sử của user_id = {user_id}")
        return

    merged = user_ratings.merge(
        df_books[
            [
                "book_id",
                "title",
                "authors",
                "original_publication_year",
                "average_rating",
            ]
        ],
        on="book_id",
    ).sort_values(by="rating", ascending=False)

    print(f"\n{'='*70}")
    print(f"LỊCH SỬ ĐỌC CỦA USER {user_id} (TOP {top_n} SÁCH ĐƯỢC CHẤM CAO NHẤT):")
    print(f"{'='*70}")
    for idx, row in enumerate(merged.head(top_n).itertuples(), start=1):
        year = (
            int(row.original_publication_year)
            if pd.notna(row.original_publication_year)
            else "N/A"
        )
        print(
            f"{idx}. {row.title} ({year})\n"
            f"   Tác giả: {row.authors} | Điểm User chấm: {row.rating} sao | Goodreads TB: {row.average_rating}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="User-Based Collaborative Filtering Recommender System"
    )
    parser.add_argument(
        "--users",
        type=int,
        default=28,
        help="Số lượng người dùng tích cực để lấy mẫu (mặc định: 1500)",
    )
    parser.add_argument(
        "--books",
        type=int,
        default=2000,
        help="Số lượng sách phổ biến để lấy mẫu (mặc định: 2000)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=25,
        help="Số lượng láng giềng tương đồng Top-K (mặc định: 25)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Số lượng sách gợi ý (mặc định: 10)",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=3000,
        help="Số mẫu test dùng để tính RMSE/MAE (mặc định: 3000)",
    )
    parser.add_argument(
        "--user_id",
        type=int,
        default=None,
        help="Mã user_id muốn thực hiện gợi ý (nếu không truyền sẽ chọn mẫu ngẫu nhiên)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("HỆ THỐNG GỢI Ý SÁCH (USER-BASED COLLABORATIVE FILTERING)")
    print("=" * 70)

    # 1. Tải và tiền xử lý dữ liệu
    train_df, test_df, df_books, df_ratings = load_dataset(
        num_users=args.users, num_books=args.books
    )

    # 2. Khởi tạo và huấn luyện mô hình
    model = UserBasedCollaborativeFiltering(k_neighbors=args.k)
    model.fit(train_df)

    # 3. Đánh giá mô hình trên tập test
    model.evaluate(test_df, sample_size=args.test_samples)

    # 4. Thực nghiệm gợi ý sách cho một người dùng cụ thể
    if args.user_id is not None and args.user_id in model.user2idx:
        target_user = args.user_id
    else:
        # Chọn một user có nhiều đánh giá trong tập train để kết quả gợi ý phong phú
        target_user = int(train_df["user_id"].value_counts().index[5])

    # In lịch sử đọc của người dùng
    print_user_history(target_user, df_ratings, df_books, top_n=5)

    # Gợi ý Top-N cuốn sách phù hợp nhất
    print(f"\n{'='*70}")
    print(f"DANH SÁCH TOP {args.top_n} SÁCH GỢI Ý CHO USER {target_user}:")
    print(f"{'='*70}")
    recommendations = model.recommend(
        user_id=target_user, top_n=args.top_n, df_books=df_books
    )

    if not recommendations.empty:
        for idx, row in enumerate(recommendations.itertuples(), start=1):
            year = (
                int(row.original_publication_year)
                if pd.notna(row.original_publication_year)
                else "N/A"
            )
            print(
                f"{idx}. {row.title} ({year})\n"
                f"   Tác giả: {row.authors}\n"
                f"   Điểm dự đoán: {row.predicted_rating:.2f} sao | Điểm TB hệ thống: {row.average_rating}"
            )
    else:
        print("Không thể tạo danh sách gợi ý cho người dùng này.")

    print("\n[+] Hoàn thành thực thi chương trình!")


if __name__ == "__main__":
    main()
