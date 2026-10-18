"""
Hệ Gợi Ý (Recommender Systems)
Các thuật toán Lọc cộng tác (Collaborative Filtering):
  1. User-Based CF  – Lọc cộng tác dựa trên người dùng
  2. Item-Based CF  – Lọc cộng tác dựa trên sản phẩm (item)
  3. Matrix Factorization (SVD) – Phân tích ma trận nhân tử
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
from scipy.sparse.linalg import svds
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
    num_users: int = -1, num_books: int = -1
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
        print(
            f"    - Điểm trung bình toàn cục (Global mean): {self.global_mean:.2f}")
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
            top_k_idx = np.argsort(filtered_sims)[-self.k_neighbors:]
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
            print(
                f"[!] User {user_id} không tồn tại trong dữ liệu huấn luyện!")
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
            rec_df = rec_df.merge(
                df_books[valid_cols], on="book_id", how="left")

        return rec_df


# =============================================================================
# THUẬT TOÁN 2: ITEM-BASED COLLABORATIVE FILTERING
# =============================================================================

class ItemBasedCollaborativeFiltering:
    """
    Mô hình Lọc cộng tác dựa trên sản phẩm (Item-Based Collaborative Filtering).
    Bao gồm:
    - Tính độ tương đồng Cosine giữa các cuốn sách dựa trên vector đánh giá của người dùng.
    - Dự đoán điểm dựa trên Top-K sách tương tự nhất mà user đã đánh giá.
    - Chuẩn hóa Mean-centering theo user để giảm độ lệch cá nhân.
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
        self.item_similarity: np.ndarray = np.array([])

    def fit(self, df_train: pd.DataFrame) -> "ItemBasedCollaborativeFiltering":
        """Huấn luyện mô hình từ DataFrame gồm: user_id, book_id, rating."""
        start_time = time.time()
        print("\n[*] Đang huấn luyện mô hình Item-Based CF...")

        unique_users = sorted(df_train["user_id"].unique())
        unique_books = sorted(df_train["book_id"].unique())

        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for i, u in enumerate(unique_users)}
        self.book2idx = {b: i for i, b in enumerate(unique_books)}
        self.idx2book = {i: b for i, b in enumerate(unique_books)}

        n_u = len(unique_users)
        n_b = len(unique_books)
        self.global_mean = float(df_train["rating"].mean())

        # 1. Xây dựng ma trận User-Item
        self.R = np.zeros((n_u, n_b), dtype=np.float32)
        for row in df_train.itertuples():
            if row.user_id in self.user2idx and row.book_id in self.book2idx:
                u = self.user2idx[row.user_id]
                b = self.book2idx[row.book_id]
                self.R[u, b] = row.rating

        # 2. Tính trung bình điểm của mỗi user
        counts = np.count_nonzero(self.R, axis=1)
        sums = np.sum(self.R, axis=1)
        self.user_means = np.divide(
            sums,
            counts,
            out=np.full(n_u, self.global_mean, dtype=float),
            where=counts > 0,
        )

        # 3. Chuẩn hóa Mean-centered
        self.R_norm = np.zeros_like(self.R)
        for u in range(n_u):
            rated_indices = np.where(self.R[u, :] > 0)[0]
            self.R_norm[u, rated_indices] = (
                self.R[u, rated_indices] - self.user_means[u]
            )

        # 4. Tính ma trận độ tương đồng giữa các item (sách)
        #    Chuyển vị: mỗi hàng = 1 item, mỗi cột = 1 user
        self.item_similarity = cosine_similarity(self.R_norm.T)
        np.fill_diagonal(self.item_similarity, 0.0)

        elapsed = time.time() - start_time
        print(f"    - Hoàn tất trong {elapsed:.2f} giây.")
        print(
            f"    - Kích thước ma trận tương đồng item: {self.item_similarity.shape}"
        )
        print(
            f"    - Điểm trung bình toàn cục (Global mean): {self.global_mean:.2f}")
        return self

    def predict_rating(self, user_id: int, book_id: int) -> float:
        """
        Dự đoán điểm số mà user_id sẽ dành cho book_id.
        Công thức:
            r_hat(u, i) = mean_u + sum(sim(i, j) * (r_{u, j} - mean_u)) / sum(|sim(i, j)|)
        """
        if user_id not in self.user2idx:
            return self.global_mean

        u = self.user2idx[user_id]
        u_mean = self.user_means[u]

        if book_id not in self.book2idx:
            return float(u_mean)

        b = self.book2idx[book_id]

        # Tìm các sách user đã đánh giá (trừ sách b)
        rated_by_user = np.where(self.R[u, :] > 0)[0]
        rated_by_user = rated_by_user[rated_by_user != b]

        if len(rated_by_user) == 0:
            return float(u_mean)

        # Độ tương đồng giữa sách b và các sách user đã đọc
        sims = self.item_similarity[b, rated_by_user]

        valid_mask = sims > self.min_similarity
        if not np.any(valid_mask):
            return float(u_mean)

        filtered_books = rated_by_user[valid_mask]
        filtered_sims = sims[valid_mask]

        # Chọn Top-K sách tương tự nhất
        if len(filtered_sims) > self.k_neighbors:
            top_k_idx = np.argsort(filtered_sims)[-self.k_neighbors:]
            filtered_books = filtered_books[top_k_idx]
            filtered_sims = filtered_sims[top_k_idx]

        # Độ lệch chuẩn hóa: (r_{u,j} - mean_u)
        norm_ratings = self.R_norm[u, filtered_books]

        sim_sum = np.sum(np.abs(filtered_sims))
        if sim_sum == 0:
            return float(u_mean)

        pred = u_mean + (np.sum(filtered_sims * norm_ratings) / sim_sum)
        return float(np.clip(pred, 1.0, 5.0))

    def evaluate(
        self, test_df: pd.DataFrame, sample_size: Optional[int] = 3000
    ) -> Tuple[float, float]:
        """Đo lường sai số dự đoán trên tập test bằng RMSE và MAE."""
        if sample_size and sample_size < len(test_df):
            test_subset = test_df.sample(n=sample_size, random_state=42)
        else:
            test_subset = test_df

        actuals, predictions = [], []
        start_time = time.time()
        for row in test_subset.itertuples():
            pred = self.predict_rating(row.user_id, row.book_id)
            actuals.append(row.rating)
            predictions.append(pred)

        eval_time = time.time() - start_time
        rmse = math.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        print(
            f"\n[*] Kết quả đánh giá (Item-Based CF) trên {len(test_subset):,} mẫu test:")
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
        """Gợi ý Top-N cuốn sách phù hợp nhất mà user_id chưa từng đọc."""
        if user_id not in self.user2idx:
            print(
                f"[!] User {user_id} không tồn tại trong dữ liệu huấn luyện!")
            return pd.DataFrame()

        u = self.user2idx[user_id]
        rated_book_indices = set(np.where(self.R[u, :] > 0)[0])
        unrated_indices = [
            b for b in range(len(self.idx2book)) if b not in rated_book_indices
        ]

        scores = []
        for b in unrated_indices:
            book_id = self.idx2book[b]
            pred = self.predict_rating(user_id, book_id)
            scores.append((book_id, pred))

        scores.sort(key=lambda x: x[1], reverse=True)
        rec_df = pd.DataFrame(scores[:top_n], columns=[
                              "book_id", "predicted_rating"])

        if df_books is not None:
            book_info_cols = [
                "book_id", "title", "authors",
                "original_publication_year", "average_rating",
            ]
            valid_cols = [c for c in book_info_cols if c in df_books.columns]
            rec_df = rec_df.merge(
                df_books[valid_cols], on="book_id", how="left")

        return rec_df


# =============================================================================
# THUẬT TOÁN 3: Matrix embedding
# =============================================================================

class MatrixEmbedding:
    def __init__(self, n_factors: int = 50):
        """
        Matrix Factorization dùng SGD.

        Args:
            n_factors: Số chiều latent factors.
        """
        self.n_factors = n_factors

        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.book2idx: Dict[int, int] = {}
        self.idx2book: Dict[int, int] = {}

        self.R = np.array([])
        self.Wu = np.array([])
        self.Hi = np.array([])

    def fit(
        self,
        df_train: pd.DataFrame,
        learning_rate: float = 0.005,
        num_epochs: int = 1000,
        reg: float = 0.02,
        random_state: int = 42,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
    ) -> "MatrixEmbedding":

        start_time = time.time()

        print("\n[*] Đang huấn luyện Matrix Factorization...")

        # 1. Tính rating trung bình toàn bộ tập train
        self.global_mean = float(df_train["rating"].mean())

        # 2. Ánh xạ user_id và book_id sang index liên tục
        unique_users = sorted(df_train["user_id"].unique())
        unique_books = sorted(df_train["book_id"].unique())

        self.user2idx = {
            u: idx for idx, u in enumerate(unique_users)
        }
        self.idx2user = {
            idx: u for u, idx in self.user2idx.items()
        }

        self.book2idx = {
            b: idx for idx, b in enumerate(unique_books)
        }
        self.idx2book = {
            idx: b for b, idx in self.book2idx.items()
        }

        n_users = len(unique_users)
        n_books = len(unique_books)

        # 3. Khởi tạo ma trận rating
        self.R = np.zeros((n_users, n_books), dtype=np.float32)

        # Chỉ lấy 3 cột cần thiết, đúng thứ tự
        train_samples = df_train[
            ["user_id", "book_id", "rating"]
        ].copy()

        # Chuyển ID thực thành index trong ma trận
        train_samples["u_idx"] = train_samples["user_id"].map(
            self.user2idx
        )
        train_samples["i_idx"] = train_samples["book_id"].map(
            self.book2idx
        )

        user_indices = train_samples["u_idx"].to_numpy(dtype=int)
        book_indices = train_samples["i_idx"].to_numpy(dtype=int)
        ratings = train_samples["rating"].to_numpy(dtype=float)

        # Lưu rating thực vào R
        self.R[user_indices, book_indices] = ratings

        # 4. Khởi tạo embedding
        rng = np.random.default_rng(random_state)

        self.Wu = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(self.n_factors, n_users),
        )

        self.Hi = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(self.n_factors, n_books),
        )

        # 5. Stochastic Gradient Descent
        n_samples = len(train_samples)

        for epoch in range(num_epochs):

            # Xáo trộn thứ tự các mẫu trong epoch
            order = rng.permutation(n_samples)

            for idx in order:
                u = user_indices[idx]
                i = book_indices[idx]
                rating = ratings[idx]

                # Lấy embedding của user và book
                w_u = self.Wu[:, u]
                h_i = self.Hi[:, i]

                # Dự đoán rating
                prediction = w_u @ h_i

                # Sai số dự đoán
                error = rating - prediction

                # Sao chép embedding trước khi cập nhật
                w_old = w_u.copy()
                h_old = h_i.copy()

                # Cập nhật embedding bằng SGD
                self.Wu[:, u] += learning_rate * (
                    error * h_old - reg * w_old
                )

                self.Hi[:, i] += learning_rate * (
                    error * w_old - reg * h_old
                )

            # Tính RMSE trên tập train sau mỗi epoch
            train_predictions = np.sum(
                self.Wu[:, user_indices] *
                self.Hi[:, book_indices],
                axis=0,
            )

            train_rmse = math.sqrt(
                mean_squared_error(ratings, train_predictions)
            )

            if epoch % 100 == 0 or epoch == num_epochs-1:
                print(
                    f"Epoch {epoch + 1:02d}/{num_epochs} "
                    f"| Train RMSE: {train_rmse:.4f}"
                )

        self.min_rating = min_rating
        self.max_rating = max_rating

        print(
            f"[*] Huấn luyện hoàn tất trong "
            f"{time.time() - start_time:.2f} giây."
        )

        return self

    def predict_rating(
        self,
        user_id: int,
        book_id: int,
    ) -> float:

        # Fallback nếu user hoặc book chưa xuất hiện trong train
        if (
            user_id not in self.user2idx
            or book_id not in self.book2idx
        ):
            return float(
                np.clip(
                    self.global_mean,
                    self.min_rating,
                    self.max_rating,
                )
            )

        u = self.user2idx[user_id]
        i = self.book2idx[book_id]

        # Rating dự đoán = tích vô hướng của hai embedding
        prediction = self.Wu[:, u] @ self.Hi[:, i]

        # Giới hạn rating trong khoảng hợp lệ
        return float(
            np.clip(
                prediction,
                self.min_rating,
                self.max_rating,
            )
        )

    def evaluate(
        self,
        test_df: pd.DataFrame,
        sample_size: Optional[int] = 3000,
    ) -> Tuple[float, float]:

        if sample_size and sample_size < len(test_df):
            test_subset = test_df.sample(
                n=sample_size,
                random_state=42,
            )
        else:
            test_subset = test_df

        start_time = time.time()

        actuals = test_subset["rating"].to_numpy(dtype=float)

        predictions = [
            self.predict_rating(row.user_id, row.book_id)
            for row in test_subset.itertuples(index=False)
        ]

        rmse = math.sqrt(
            mean_squared_error(actuals, predictions)
        )
        mae = mean_absolute_error(actuals, predictions)

        print(
            f"\n[*] Kết quả Matrix Factorization "
            f"trên {len(test_subset):,} mẫu test:"
        )
        print(f"    - Thời gian: {time.time() - start_time:.2f} giây")
        print(f"    - RMSE: {rmse:.4f}")
        print(f"    - MAE:  {mae:.4f}")

        return rmse, mae

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        df_books: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:

        if user_id not in self.user2idx:
            print(f"[!] User {user_id} không tồn tại trong tập train.")
            return pd.DataFrame()

        u = self.user2idx[user_id]

        # Tìm những cuốn sách user đã đánh giá
        rated_book_indices = set(
            np.where(self.R[u, :] > 0)[0]
        )

        # Dự đoán rating cho tất cả sách
        predictions = self.Wu[:, u] @ self.Hi

        # Chỉ giữ những sách user chưa đánh giá
        candidate_indices = [
            i
            for i in range(len(self.idx2book))
            if i not in rated_book_indices
        ]

        scores = [
            (
                self.idx2book[i],
                float(
                    np.clip(
                        predictions[i],
                        self.min_rating,
                        self.max_rating,
                    )
                ),
            )
            for i in candidate_indices
        ]

        scores.sort(key=lambda x: x[1], reverse=True)

        rec_df = pd.DataFrame(
            scores[:top_n],
            columns=["book_id", "predicted_rating"],
        )

        if df_books is not None and not rec_df.empty:
            book_info_cols = [
                "book_id",
                "title",
                "authors",
                "original_publication_year",
                "average_rating",
            ]

            valid_cols = [
                col for col in book_info_cols
                if col in df_books.columns
            ]

            if "book_id" in valid_cols:
                rec_df = rec_df.merge(
                    df_books[valid_cols].drop_duplicates("book_id"),
                    on="book_id",
                    how="left",
                )

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
    print(
        f"LỊCH SỬ ĐỌC CỦA USER {user_id} (TOP {top_n} SÁCH ĐƯỢC CHẤM CAO NHẤT):")
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


def _print_recommendations(recommendations: pd.DataFrame, top_n: int, user_id: int):
    """In danh sách gợi ý sách theo định dạng chuẩn."""
    print(f"\n{'='*70}")
    print(f"DANH SÁCH TOP {top_n} SÁCH GỢI Ý CHO USER {user_id}:")
    print(f"{'='*70}")
    if recommendations.empty:
        print("Không thể tạo danh sách gợi ý cho người dùng này.")
        return
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


def main():
    parser = argparse.ArgumentParser(
        description="Collaborative Filtering Recommender System (User-CF / Item-CF / SVD)"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="all",
        choices=["user", "item", "matrix", "all"],
        help="Thuật toán muốn chạy: user | item | svd | all (mặc định: all)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=28,
        help="Số lượng người dùng tích cực để lấy mẫu (mặc định: 28)",
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
        help="Số lượng láng giềng Top-K dùng cho User/Item-CF (mặc định: 25)",
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=50,
        help="Số chiều ẩn (latent factors) dùng cho SVD (mặc định: 50)",
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
    print("HỆ THỐNG GỢI Ý SÁCH – COLLABORATIVE FILTERING")
    print(f"Thuật toán được chọn: {args.algo.upper()}")
    print("=" * 70)

    # 1. Tải và tiền xử lý dữ liệu
    train_df, test_df, df_books, df_ratings = load_dataset(
        num_users=args.users, num_books=args.books
    )

    run_user = args.algo in ("user", "all")
    run_item = args.algo in ("item", "all")
    run_svd = args.algo in ("matrix",  "all")

    results: Dict[str, Tuple[float, float]] = {}  # algo -> (rmse, mae)

    # ── 2a. User-Based CF ──────────────────────────────────────────────────
    if run_user:
        print("\n" + "─" * 70)
        print("[ALGO 1/3] USER-BASED COLLABORATIVE FILTERING")
        print("─" * 70)
        user_model = UserBasedCollaborativeFiltering(k_neighbors=args.k)
        user_model.fit(train_df)
        rmse_u, mae_u = user_model.evaluate(
            test_df, sample_size=args.test_samples)
        results["User-CF"] = (rmse_u, mae_u)

    # ── 2b. Item-Based CF ──────────────────────────────────────────────────
    if run_item:
        print("\n" + "─" * 70)
        print("[ALGO 2/3] ITEM-BASED COLLABORATIVE FILTERING")
        print("─" * 70)
        item_model = ItemBasedCollaborativeFiltering(k_neighbors=args.k)
        item_model.fit(train_df)
        rmse_i, mae_i = item_model.evaluate(
            test_df, sample_size=args.test_samples)
        results["Item-CF"] = (rmse_i, mae_i)

    # ── 2c. Matrix Factorization SVD ───────────────────────────────────────
    if run_svd:
        print("\n" + "─" * 70)
        print("[ALGO 3/3] MATRIX FACTORIZATION (SVD)")
        print("─" * 70)
        svd_model = MatrixEmbedding(n_factors=args.factors)
        svd_model.fit(train_df)
        rmse_s, mae_s = svd_model.evaluate(
            test_df, sample_size=args.test_samples)
        results["SVD"] = (rmse_s, mae_s)

    # ── 3. Bảng so sánh kết quả ────────────────────────────────────────────
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("BẢNG SO SÁNH KẾT QUẢ CÁC THUẬT TOÁN")
        print("=" * 70)
        print(f"{'Thuật toán':<25} {'RMSE':>10} {'MAE':>10}")
        print("-" * 45)
        for algo_name, (rmse, mae) in results.items():
            print(f"{algo_name:<25} {rmse:>10.4f} {mae:>10.4f}")
        print("=" * 70)

    # ── 4. Gợi ý sách cho người dùng cụ thể ───────────────────────────────
    # Chọn model ưu tiên: svd > user > item
    if run_svd:
        primary_model = svd_model          # type: ignore[possibly-undefined]
    elif run_user:
        primary_model = user_model         # type: ignore[possibly-undefined]
    else:
        primary_model = item_model         # type: ignore[possibly-undefined]

    # Xác định target user
    if args.user_id is not None and args.user_id in primary_model.user2idx:
        target_user = args.user_id
    else:
        target_user = int(train_df["user_id"].value_counts().index[5])

    print_user_history(target_user, df_ratings, df_books, top_n=5)

    # Gợi ý từ từng model đã chạy
    if run_user:
        print(f"\n>>> [User-Based CF] Gợi ý cho User {target_user}:")
        _print_recommendations(
            user_model.recommend(user_id=target_user,
                                 top_n=args.top_n, df_books=df_books),
            args.top_n, target_user,
        )

    if run_item:
        print(f"\n>>> [Item-Based CF] Gợi ý cho User {target_user}:")
        _print_recommendations(
            item_model.recommend(user_id=target_user,
                                 top_n=args.top_n, df_books=df_books),
            args.top_n, target_user,
        )

    if run_svd:
        print(
            f"\n>>> [Matrix Factorization SVD] Gợi ý cho User {target_user}:")
        _print_recommendations(
            svd_model.recommend(user_id=target_user,
                                top_n=args.top_n, df_books=df_books),
            args.top_n, target_user,
        )

    print("\n[+] Hoàn thành thực thi chương trình!")


if __name__ == "__main__":
    main()
