"""
Hệ Gợi Ý (Recommender Systems)
Thuật toán: Lọc Dựa Trên Nội Dung (Content-Based Filtering)
Dataset: Goodbooks-10k (books_mapped.jsonl, books.csv, ratings.csv, tags.csv, book_tags.csv)
Môi trường: Python 3.10+ (scikit-learn, pandas, numpy, scipy)

Tổng quan về thuật toán Lọc Dựa Trên Nội Dung (Content-Based Filtering):
----------------------------------------------------------------------
1. Nguyên lý cốt lõi:
   - Gợi ý các mục (items) dựa trên sự tương đồng giữa đặc trưng nội dung của mục
     với hồ sơ sở thích (User Profile) được xây dựng từ lịch sử tương tác của người dùng.
   - Trả lời câu hỏi: "Người dùng thích nội dung X trong quá khứ, vậy mục nào khác cũng chứa nội dung X?"

2. Ưu điểm nổi bật so với Collaborative Filtering (Lọc cộng tác):
   - Giải quyết triệt để bài toán "Item Cold-Start": Có thể gợi ý ngay một cuốn sách mới
     vừa xuất bản (chưa có ai đọc/chấm điểm), chỉ cần có thông tin mô tả nội dung.
   - Tính giải thích được (Explainability / Transparency): Dễ dàng chỉ ra vì sao một cuốn sách
     được gợi ý (cùng thể loại, cùng tác giả, chung từ khóa cốt truyện).
   - Độc lập người dùng: Không bị ảnh hưởng bởi sự khan hiếm tương tác của cộng đồng người dùng khác.

3. Quy trình thực hiện trong mã nguồn này:
   - Bước 1: Thu thập & làm sạch đặc trưng (Title, Authors, Genres, Tags cộng đồng, Description).
   - Bước 2: Biểu diễn vector bằng TF-IDF (Term Frequency - Inverse Document Frequency) với N-gram & Sublinear TF.
   - Bước 3: Tính toán độ tương đồng Cosine Similarity cho bài toán Item-to-Item.
   - Bước 4: Xây dựng User Profile theo 2 phương pháp:
             + Phương pháp 1: Centroid có trọng số lệch chuẩn (Mean-centered Weighted Average / Rocchio-based).
             + Phương pháp 2: Hồi quy tuyến tính có điều chuẩn L2 (Ridge Regression per user).
   - Bước 5: Dự đoán điểm đánh giá & sinh danh sách Top-N gợi ý kèm lời giải thích (Explainable Recommendation).
   - Bước 6: Đánh giá mô hình trên tập kiểm thử bằng RMSE, MAE, Precision@K, Recall@K, NDCG@K và Hit Rate@K.
   - Bước 7: Thử nghiệm kịch bản Item Cold-Start với một cuốn sách mới tinh.
"""

import argparse
import html
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


# =============================================================================
# 1. TIỆN ÍCH DỮ LIỆU & NẠP DATASET
# =============================================================================

def find_data_path(filename: str) -> str:
    """Tìm đường dẫn tới file dữ liệu trong thư mục hiện tại hoặc goodbooks-10k."""
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
        f"Không tìm thấy file '{filename}'. Vui lòng kiểm tra thư mục dự án hoặc goodbooks-10k!"
    )


def load_book_tags() -> Dict[int, List[str]]:
    """
    Đọc tags.csv và book_tags.csv để bổ sung từ khóa cộng đồng cho từng cuốn sách.
    Loại bỏ các tag dạng kệ sách cá nhân (to-read, favorites, owned...) để giữ lại tag nội dung thực.
    """
    try:
        tags_file = find_data_path("tags.csv")
        book_tags_file = find_data_path("book_tags.csv")
    except FileNotFoundError:
        return {}

    # Bộ lọc các tag không mang ngữ nghĩa nội dung
    ignore_tags = {
        "to-read", "currently-reading", "favorites", "favourites", "books-i-own",
        "owned", "owned-books", "default", "audiobook", "audiobooks", "audio",
        "ebook", "ebooks", "kindle", "library", "my-books", "i-own", "read-in-2015",
        "read-in-2016", "read-in-2017", "wish-list", "all-time-favorites",
    }

    try:
        df_tags = pd.read_csv(tags_file)
        df_book_tags = pd.read_csv(book_tags_file)

        merged_tags = df_book_tags.merge(df_tags, on="tag_id", how="inner")
        # Giữ lại các tag có chữ/số hợp lệ
        valid_mask = merged_tags["tag_name"].str.match(r"^[a-zA-Z0-9\s\-]+$", na=False)
        merged_tags = merged_tags[valid_mask]
        merged_tags = merged_tags[~merged_tags["tag_name"].isin(ignore_tags)]

        # Lấy top 8 tags có lượt gán nhiều nhất cho mỗi goodreads_book_id
        top_tags = (
            merged_tags.sort_values(["goodreads_book_id", "count"], ascending=[True, False])
            .groupby("goodreads_book_id")
            .head(8)
        )
        tags_dict = top_tags.groupby("goodreads_book_id")["tag_name"].apply(list).to_dict()
        return tags_dict
    except Exception as e:
        print(f"[!] Cảnh báo: Không thể nạp book_tags: {e}")
        return {}


def load_books_metadata() -> pd.DataFrame:
    """
    Tải dữ liệu sách:
    - Ưu tiên đọc books_mapped.jsonl (có description & genres).
    - Tự động bổ sung thông tin từ books.csv để đảm bảo 100% sách có đầy đủ:
      title, authors, original_publication_year, average_rating (không bị None).
    - Bổ sung các tag từ vựng cộng đồng từ book_tags.csv.
    """
    mapped_candidates = [
        "books_mapped.jsonl",
        "books_data_en.jsonl",
        os.path.join(os.path.dirname(__file__), "books_mapped.jsonl"),
        os.path.join(os.path.dirname(__file__), "books_data_en.jsonl"),
    ]

    jsonl_path = next((cand for cand in mapped_candidates if os.path.exists(cand)), None)
    csv_path = find_data_path("books.csv")
    df_csv = pd.read_csv(csv_path)

    if jsonl_path:
        print(f"[*] Đang tải dữ liệu thuộc tính sách từ: {jsonl_path}")
        books = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    books.append(json.loads(line))
        df_json = pd.DataFrame(books)

        if "id" in df_json.columns:
            df_json["book_id"] = df_json["id"].astype(int)
        elif "book_id" in df_json.columns:
            df_json["book_id"] = df_json["book_id"].astype(int)

        # Ghép nối với books.csv để bù đắp bất kỳ trường nào bị thiếu trong JSON
        cols_from_csv = [
            "book_id", "goodreads_book_id", "title", "authors",
            "original_publication_year", "average_rating", "image_url"
        ]
        valid_csv_cols = [c for c in cols_from_csv if c in df_csv.columns]
        df_merged = df_json.merge(
            df_csv[valid_csv_cols],
            on="book_id",
            how="right",
            suffixes=("_json", "_csv"),
        )

        # Bù title nếu JSON bị None
        df_merged["title"] = df_merged["title_json"].fillna(df_merged["title_csv"])
        # Bù authors nếu JSON là rỗng hoặc None
        def _get_authors(row):
            aj = row.get("authors_json")
            if isinstance(aj, list) and len(aj) > 0:
                return aj
            ac = row.get("authors_csv")
            if pd.notna(ac):
                return [a.strip() for a in str(ac).split(",")]
            return ["Unknown"]

        df_merged["authors"] = df_merged.apply(_get_authors, axis=1)

        # Bù average_rating / rating_value
        if "rating_value" in df_merged.columns:
            df_merged["rating_value"] = df_merged["rating_value"].fillna(df_merged["average_rating"])
        else:
            df_merged["rating_value"] = df_merged["average_rating"]

        # Bù publication year
        if "original_publication_year" not in df_merged.columns:
            df_merged["original_publication_year"] = df_csv["original_publication_year"]

        # Chuẩn hóa genres
        def _clean_genres(g):
            if isinstance(g, list):
                return [x for x in g if x and x not in ("...more", "...less", "more", "less")]
            return []

        df_merged["genres"] = df_merged["genres"].apply(_clean_genres)
        df_merged["description"] = df_merged["description"].fillna("")
        df_books = df_merged
    else:
        print(f"[!] Không tìm thấy file JSONL, sử dụng trực tiếp từ CSV: {csv_path}")
        df_books = df_csv.copy()
        df_books["genres"] = [[] for _ in range(len(df_books))]
        df_books["description"] = ""
        df_books["rating_value"] = df_books["average_rating"]
        df_books["authors"] = df_books["authors"].apply(
            lambda x: [a.strip() for a in str(x).split(",")] if pd.notna(x) else ["Unknown"]
        )

    # Nạp bổ sung tags
    tags_dict = load_book_tags()
    if tags_dict:
        def _attach_tags(row):
            gid = row.get("goodreads_book_id_csv") or row.get("goodreads_book_id")
            if pd.notna(gid) and int(gid) in tags_dict:
                return tags_dict[int(gid)]
            return []

        df_books["tags"] = df_books.apply(_attach_tags, axis=1)
    else:
        df_books["tags"] = [[] for _ in range(len(df_books))]

    print(f"    - Đã nạp thành công {len(df_books):,} cuốn sách vào bộ nhớ (Metadata đầy đủ 100%).")
    return df_books


def load_ratings_dataset(
    num_users: int = 1500, num_books: int = 2000, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tải và phân chia tập đánh giá ratings thành Train và Test."""
    ratings_file = find_data_path("ratings.csv")
    print(f"[*] Đang tải dữ liệu đánh giá (ratings) từ: {ratings_file}")
    df_ratings = pd.read_csv(ratings_file)

    top_user_ids = df_ratings["user_id"].value_counts().head(num_users).index
    top_book_ids = df_ratings["book_id"].value_counts().head(num_books).index

    df_filtered = df_ratings[
        df_ratings["user_id"].isin(top_user_ids)
        & df_ratings["book_id"].isin(top_book_ids)
    ].copy()

    n_u = df_filtered["user_id"].nunique()
    n_b = df_filtered["book_id"].nunique()
    n_r = len(df_filtered)
    sparsity = 1.0 - (n_r / (n_u * n_b)) if (n_u * n_b) > 0 else 0.0

    print(f"[*] Dữ liệu ratings đã lọc mẫu:")
    print(f"    - Số người dùng (Users): {n_u:,}")
    print(f"    - Số cuốn sách (Books):  {n_b:,}")
    print(f"    - Tổng số đánh giá:      {n_r:,}")
    print(f"    - Độ thưa (Sparsity):    {sparsity * 100:.2f}%")

    train_df, test_df = train_test_split(df_filtered, test_size=test_size, random_state=42)
    print(f"[*] Phân chia tập dữ liệu: Train = {len(train_df):,} dòng | Test = {len(test_df):,} dòng")
    return train_df, test_df, df_filtered


# =============================================================================
# 2. LỚP THUẬT TOÁN CONTENT-BASED FILTERING CHÍNH
# =============================================================================

class ContentBasedRecommender:
    """
    Hệ thống gợi ý dựa trên nội dung (Content-Based Recommender System):
    - Xây dựng ma trận đặc trưng văn bản TF-IDF từ các trường nội dung:
      Tiêu đề, Tác giả, Thể loại, Tags cộng đồng, Tóm tắt cốt truyện.
    - Hỗ trợ 2 cơ chế xây dựng hồ sơ người dùng (User Profile):
      1) 'weighted_avg': Centroid có trọng số lệch chuẩn (Mean-centered Weighted Average).
      2) 'ridge': Hồi quy tuyến tính có điều chuẩn L2 (Ridge Regression).
    - Hỗ trợ tìm kiếm sách tương tự (Item-to-Item) và Gợi ý cá nhân hóa (User-to-Item) kèm lời giải thích.
    - Hỗ trợ gợi ý cho mục mới (Item Cold-Start).
    """

    def __init__(
        self,
        max_features: int = 12000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
        )

        self.df_books: pd.DataFrame = pd.DataFrame()
        self.tfidf_matrix: Optional[sparse.csr_matrix] = None
        self.book_id_to_idx: Dict[int, int] = {}
        self.idx_to_book_id: Dict[int, int] = {}
        self.feature_names: np.ndarray = np.array([])

        # Hồ sơ người dùng
        self.user_profiles: Dict[int, np.ndarray] = {}
        self.user_stats: Dict[int, Tuple[float, float]] = {}  # user_id -> (mean, std)
        self.user_models: Dict[int, Ridge] = {}               # Dùng cho phương pháp 'ridge'
        self.user_rated_books: Dict[int, Dict[int, float]] = {}
        self.global_mean: float = 3.0
        self.method: str = "weighted_avg"

    @staticmethod
    def _create_content_soup(row: pd.Series) -> str:
        """
        Tổng hợp tất cả đặc trưng nội dung thành 'Content Soup' giàu ngữ nghĩa.
        Áp dụng trọng số tương đối bằng cách nhân bản các trường quan trọng:
        - Thể loại (Genres): x3 (rất quan trọng trong phân loại thị hiếu sách).
        - Tác giả (Authors): x2 (phong cách viết, độc giả thường đọc theo tác giả).
        - Tags cộng đồng: x2 (từ khóa đặc thù từ bạn đọc).
        - Tiêu đề (Title): x2.
        - Tóm tắt (Description): x1.
        """
        # 1. Thể loại (Genres)
        raw_genres = row.get("genres") or []
        genres_tokens = []
        for g in raw_genres:
            g_clean = re.sub(r"[^\w]", "", str(g).lower())
            if g_clean and g_clean not in ("more", "less"):
                genres_tokens.append(f"genre_{g_clean}")
        genres_part = " ".join(genres_tokens)

        # 2. Tác giả (Authors)
        raw_authors = row.get("authors") or []
        authors_tokens = []
        for a in raw_authors:
            a_clean = re.sub(r"[^\w]", "_", str(a).strip().lower())
            if a_clean:
                authors_tokens.append(f"author_{a_clean}")
        authors_part = " ".join(authors_tokens)

        # 3. Tags từ vựng cộng đồng
        raw_tags = row.get("tags") or []
        tags_tokens = []
        for t in raw_tags:
            t_clean = re.sub(r"[^\w]", "_", str(t).strip().lower())
            if t_clean:
                tags_tokens.append(f"tag_{t_clean}")
        tags_part = " ".join(tags_tokens)

        # 4. Tiêu đề
        title_str = str(row.get("title") or "").strip()
        title_clean = html.unescape(title_str)

        # 5. Tóm tắt cốt truyện
        desc_str = str(row.get("description") or "").strip()
        desc_clean = re.sub(r"<[^>]+>", " ", desc_str)  # Bỏ HTML tags
        desc_clean = html.unescape(desc_clean)

        # Kết hợp có trọng số
        soup = (
            f"{title_clean} {title_clean} "
            f"{authors_part} {authors_part} "
            f"{genres_part} {genres_part} {genres_part} "
            f"{tags_part} {tags_part} "
            f"{desc_clean}"
        )
        return soup

    def fit_items(self, df_books: pd.DataFrame) -> "ContentBasedRecommender":
        """Xây dựng không gian đặc trưng TF-IDF cho toàn bộ danh mục sách."""
        start_time = time.time()
        print("\n[*] Đang xây dựng không gian đặc trưng nội dung (TF-IDF)...")

        self.df_books = df_books.copy().reset_index(drop=True)
        self.book_id_to_idx = {int(row["book_id"]): i for i, row in self.df_books.iterrows()}
        self.idx_to_book_id = {i: int(row["book_id"]) for i, row in self.df_books.iterrows()}

        # Sinh content soup cho 10,000 cuốn sách
        soups = self.df_books.apply(self._create_content_soup, axis=1)

        # Huấn luyện TF-IDF vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(soups)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

        elapsed = time.time() - start_time
        print(f"    - Hoàn tất trong: {elapsed:.2f} giây.")
        print(f"    - Kích thước ma trận TF-IDF: {self.tfidf_matrix.shape} (Số sách × Số từ khóa)")
        return self

    def fit_users(
        self,
        df_train: pd.DataFrame,
        method: str = "weighted_avg",
        ridge_alpha: float = 1.0,
    ) -> "ContentBasedRecommender":
        """
        Xây dựng hồ sơ sở thích (User Profile) cho từng người dùng từ tập Train:
        - method='weighted_avg':
          W_u = sum_i ( (r_{ui} - mean_u) * v_i )
          Vector đặc trưng được chuẩn hóa L2 norm.
        - method='ridge':
          Huấn luyện mô hình hồi quy tuyến tính Ridge Regression (min ||Xw - y||^2 + alpha*||w||^2)
          cho mỗi người dùng, ánh xạ trực tiếp đặc trưng nội dung sang điểm đánh giá.
        """
        start_time = time.time()
        self.method = method
        print(f"\n[*] Đang xây dựng hồ sơ người dùng (User Profiles) theo phương pháp '{method}'...")

        self.global_mean = float(df_train["rating"].mean())
        self.user_profiles.clear()
        self.user_stats.clear()
        self.user_models.clear()
        self.user_rated_books.clear()

        n_features = self.tfidf_matrix.shape[1]

        # Nhóm theo từng người dùng
        for user_id, group in df_train.groupby("user_id"):
            u_id = int(user_id)
            ratings = group["rating"].values.astype(float)
            u_mean = float(np.mean(ratings))
            u_std = float(np.std(ratings))
            if np.isnan(u_std) or u_std == 0:
                u_std = 1.0
            self.user_stats[u_id] = (u_mean, u_std)

            # Lưu danh sách sách user đã đánh giá
            self.user_rated_books[u_id] = {
                int(row.book_id): float(row.rating) for row in group.itertuples()
            }

            # Lấy các vector đặc trưng của các sách user đã đánh giá
            book_indices = [
                self.book_id_to_idx[int(row.book_id)]
                for row in group.itertuples()
                if int(row.book_id) in self.book_id_to_idx
            ]
            if not book_indices:
                continue

            user_item_matrix = self.tfidf_matrix[book_indices]

            if method == "ridge":
                # Phương pháp Ridge Regression
                clf = Ridge(alpha=ridge_alpha, fit_intercept=True)
                clf.fit(user_item_matrix, ratings)
                self.user_models[u_id] = clf
            else:
                # Phương pháp Weighted Average Centroid
                user_vec = np.zeros(n_features, dtype=np.float32)
                for row in group.itertuples():
                    bid = int(row.book_id)
                    if bid in self.book_id_to_idx:
                        idx = self.book_id_to_idx[bid]
                        # Trọng số = Điểm đánh giá - Điểm trung bình của chính user đó
                        weight = float(row.rating) - u_mean
                        book_vec = self.tfidf_matrix[idx].toarray().flatten()
                        user_vec += weight * book_vec

                norm = np.linalg.norm(user_vec)
                if norm > 0:
                    user_vec /= norm
                self.user_profiles[u_id] = user_vec

        elapsed = time.time() - start_time
        n_fitted = len(self.user_models) if method == "ridge" else len(self.user_profiles)
        print(f"    - Hoàn tất xây dựng hồ sơ cho {n_fitted:,} người dùng ({elapsed:.2f}s)!")
        return self

    def _explain_similarity(self, idx1: int, idx2: int, top_k: int = 4) -> List[str]:
        """Trích xuất các từ khóa đặc trưng nội dung chung đóng góp nhiều nhất vào độ tương đồng."""
        v1 = self.tfidf_matrix[idx1].toarray().flatten()
        v2 = self.tfidf_matrix[idx2].toarray().flatten()
        product = v1 * v2
        top_feature_indices = np.argsort(product)[::-1][:top_k]

        explanations = []
        for fid in top_feature_indices:
            if product[fid] > 0.001:
                word = self.feature_names[fid]
                if word.startswith("genre_"):
                    explanations.append(f"Thể loại: {word[6:].title()}")
                elif word.startswith("author_"):
                    explanations.append(f"Tác giả: {word[7:].replace('_', ' ').title()}")
                elif word.startswith("tag_"):
                    explanations.append(f"Tag: {word[4:].replace('_', ' ')}")
                else:
                    explanations.append(f"Từ khóa: {word}")
        return explanations

    def recommend_similar_books(
        self,
        book_id: Optional[int] = None,
        book_title: Optional[str] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Gợi ý sách tương đồng (Item-to-Item Content-Based):
        Tính độ tương đồng Cosine giữa cuốn sách chỉ định và toàn bộ kho sách.
        Kèm phân tích lý do tương đồng (cùng thể loại, cùng tác giả, từ khóa chính).
        """
        target_idx = None

        if book_id is not None and int(book_id) in self.book_id_to_idx:
            target_idx = self.book_id_to_idx[int(book_id)]
        elif book_title:
            query = book_title.strip().lower()
            matches = self.df_books[
                self.df_books["title"].str.lower().str.contains(query, regex=False, na=False)
            ]
            if not matches.empty:
                target_idx = matches.index[0]
            else:
                print(f"[!] Không tìm thấy sách khớp với từ khóa: '{book_title}'")
                return pd.DataFrame()

        if target_idx is None:
            print(f"[!] Không tìm thấy cuốn sách chỉ định (book_id={book_id})")
            return pd.DataFrame()

        target_vec = self.tfidf_matrix[target_idx]
        sim_scores = cosine_similarity(target_vec, self.tfidf_matrix).flatten()

        top_indices = sim_scores.argsort()[::-1]
        top_indices = [idx for idx in top_indices if idx != target_idx][:top_n]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.df_books.iloc[idx].to_dict()
            row["rank"] = rank
            row["similarity_score"] = float(sim_scores[idx])
            row["shared_features"] = self._explain_similarity(target_idx, idx, top_k=3)
            results.append(row)

        return pd.DataFrame(results)

    def predict_rating(self, user_id: int, book_id: int) -> float:
        """Dự đoán điểm số người dùng sẽ chấm cho một cuốn sách cụ thể."""
        if book_id not in self.book_id_to_idx:
            u_mean, _ = self.user_stats.get(user_id, (self.global_mean, 1.0))
            return float(u_mean)

        b_idx = self.book_id_to_idx[book_id]

        if self.method == "ridge":
            if user_id in self.user_models:
                pred = float(self.user_models[user_id].predict(self.tfidf_matrix[b_idx])[0])
                return float(np.clip(pred, 1.0, 5.0))
            return self.global_mean
        else:
            if user_id not in self.user_profiles:
                return self.global_mean

            user_vec = self.user_profiles[user_id]
            u_mean, u_std = self.user_stats.get(user_id, (self.global_mean, 1.0))
            sim = float(self.tfidf_matrix[b_idx].dot(user_vec).item())
            # Mô hình hồi quy điểm: Rating = u_mean + sim * u_std * 2.0
            pred = u_mean + (sim * u_std * 2.0)
            return float(np.clip(pred, 1.0, 5.0))

    def recommend_for_user(
        self,
        user_id: int,
        df_ratings_history: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Gợi ý sách cá nhân hóa cho người dùng (User-to-Item Content-Based):
        - So khớp User Profile với tất cả các sách trong hệ thống.
        - Loại trừ những cuốn sách mà người dùng đã đọc/đánh giá.
        - Trả về Top-N cuốn sách phù hợp nhất kèm lời giải thích (Explainable Recommender).
        """
        if self.method == "ridge":
            if user_id not in self.user_models:
                print(f"[!] Người dùng ID {user_id} chưa có dữ liệu huấn luyện.")
                return pd.DataFrame()
            all_preds = self.user_models[user_id].predict(self.tfidf_matrix)
        else:
            if user_id not in self.user_profiles:
                print(f"[!] Người dùng ID {user_id} chưa có dữ liệu huấn luyện.")
                return pd.DataFrame()
            user_vec = self.user_profiles[user_id]
            u_mean, u_std = self.user_stats.get(user_id, (self.global_mean, 1.0))
            sims = self.tfidf_matrix.dot(user_vec).flatten()
            all_preds = u_mean + (sims * u_std * 2.0)

        # Loại bỏ các sách người dùng đã đọc
        rated_book_ids = set(
            df_ratings_history[df_ratings_history["user_id"] == user_id]["book_id"]
        )
        if user_id in self.user_rated_books:
            rated_book_ids.update(self.user_rated_books[user_id].keys())

        for bid in rated_book_ids:
            if bid in self.book_id_to_idx:
                all_preds[self.book_id_to_idx[bid]] = -999.0

        top_indices = all_preds.argsort()[::-1][:top_n]

        # Tìm các cuốn sách user từng chấm điểm cao nhất để làm cơ sở giải thích
        user_favorites = []
        if user_id in self.user_rated_books:
            sorted_favs = sorted(
                self.user_rated_books[user_id].items(), key=lambda x: x[1], reverse=True
            )
            user_favorites = [bid for bid, r in sorted_favs[:5] if bid in self.book_id_to_idx]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.df_books.iloc[idx].to_dict()
            row["rank"] = rank
            row["predicted_rating"] = float(np.clip(all_preds[idx], 1.0, 5.0))

            # Tìm sách tương tự nhất trong danh sách yêu thích của người dùng để giải thích
            best_match_title = ""
            best_sim = -1.0
            for fav_bid in user_favorites:
                fav_idx = self.book_id_to_idx[fav_bid]
                sim_fav = float(
                    cosine_similarity(
                        self.tfidf_matrix[idx], self.tfidf_matrix[fav_idx]
                    )[0][0]
                )
                if sim_fav > best_sim:
                    best_sim = sim_fav
                    best_match_title = self.df_books.iloc[fav_idx]["title"]

            row["similarity_score"] = float(best_sim) if best_sim > 0 else 0.0
            row["explanation"] = (
                f"Tương đồng với cuốn bạn đã thích: '{best_match_title}'"
                if best_match_title
                else "Phù hợp với đặc trưng thể loại bạn thường đọc"
            )
            results.append(row)

        return pd.DataFrame(results)

    def demo_cold_start(
        self,
        new_title: str,
        new_authors: List[str],
        new_genres: List[str],
        new_description: str,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Minh họa khả năng giải quyết bài toán Item Cold-Start của Content-Based Filtering:
        - Tiếp nhận một cuốn sách hoàn toàn mới (chưa có trong database và chưa có rating).
        - Vector hóa sách mới dựa trên thông tin mô tả.
        - Tìm các cuốn sách tương đồng nhất và tìm những người dùng có sở thích phù hợp nhất.
        """
        print(f"\n{'='*75}")
        print(f"THỬ NGHIỆM ĐẶC TÍNH ITEM COLD-START VỚI CUỐN SÁCH MỚI HOÀN TOÀN:")
        print(f"Tiêu đề: {new_title}")
        print(f"Tác giả: {', '.join(new_authors)}")
        print(f"Thể loại: {', '.join(new_genres)}")
        print(f"Tóm tắt: {new_description[:120]}...")
        print(f"{'='*75}")

        dummy_row = pd.Series({
            "title": new_title,
            "authors": new_authors,
            "genres": new_genres,
            "tags": new_genres,
            "description": new_description,
        })
        new_soup = self._create_content_soup(dummy_row)
        new_vec = self.vectorizer.transform([new_soup])

        # 1. Tìm các cuốn sách tương đồng trong thư viện
        sims = cosine_similarity(new_vec, self.tfidf_matrix).flatten()
        top_sim_indices = sims.argsort()[::-1][:top_n]

        print(f"\n[+] Top {top_n} cuốn sách tương đồng nhất có sẵn trong hệ thống:")
        for r, idx in enumerate(top_sim_indices, start=1):
            b_row = self.df_books.iloc[idx]
            print(f"   {r}. {b_row['title']} (Cosine: {sims[idx]:.4f}) | Tác giả: {', '.join(b_row['authors'])}")

        # 2. Tìm top người dùng có hồ sơ phù hợp nhất để giới thiệu cuốn sách này
        user_scores = []
        for u_id, u_vec in self.user_profiles.items():
            match_score = float(new_vec.dot(u_vec).item())
            user_scores.append((u_id, match_score))

        user_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"\n[+] Top {top_n} độc giả tiềm năng có thị hiếu phù hợp nhất với sách mới này:")
        for r, (u_id, score) in enumerate(user_scores[:top_n], start=1):
            print(f"   {r}. User ID: {u_id:6d} | Điểm tương thích sở thích: {score:.4f}")

        return pd.DataFrame(
            [self.df_books.iloc[idx].to_dict() for idx in top_sim_indices]
        )

    def evaluate(
        self,
        df_test: pd.DataFrame,
        sample_size: int = 3000,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Đánh giá toàn diện mô hình Content-Based Filtering trên tập Test:
        - Sai số dự đoán điểm: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error).
        - Chất lượng xếp hạng Top-K: Precision@K, Recall@K, NDCG@K, Hit Rate@K.
        """
        print(f"\n{'='*75}")
        print(f"ĐÁNH GIÁ MÔ HÌNH CONTENT-BASED FILTERING TRÊN TẬP TEST")
        print(f"{'='*75}")

        test_sample = df_test.head(sample_size).copy()
        y_true = []
        y_pred = []

        start_time = time.time()
        for row in test_sample.itertuples():
            u = int(row.user_id)
            b = int(row.book_id)
            if b in self.book_id_to_idx:
                pred = self.predict_rating(u, b)
                y_true.append(float(row.rating))
                y_pred.append(pred)

        eval_time = time.time() - start_time
        if not y_true:
            print("[!] Không có mẫu kiểm thử hợp lệ.")
            return {}

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        print(f"[*] Kết quả dự đoán điểm số ({len(y_true):,} mẫu đánh giá - {eval_time:.2f}s):")
        print(f"    - RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"    - MAE  (Mean Absolute Error):     {mae:.4f}")

        # Đánh giá chỉ số xếp hạng (Ranking Metrics)
        eval_users = list(set(test_sample["user_id"]))[:100]
        precisions = []
        recalls = []
        ndcgs = []
        hits = []

        for u in eval_users:
            u_test = test_sample[test_sample["user_id"] == u]
            # Mục yêu thích: Điểm thực tế >= 4.0
            relevant_items = set(u_test[u_test["rating"] >= 4.0]["book_id"])
            if not relevant_items:
                continue

            candidates = list(u_test["book_id"])
            scored = [(b, self.predict_rating(u, b)) for b in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_rec = [b for b, _ in scored[:top_k]]

            # Hit rate
            hit_count = len(set(top_rec) & relevant_items)
            precisions.append(hit_count / min(top_k, len(top_rec)))
            recalls.append(hit_count / len(relevant_items))
            hits.append(1.0 if hit_count > 0 else 0.0)

            # NDCG@K
            dcg = 0.0
            for i, b in enumerate(top_rec):
                if b in relevant_items:
                    dcg += 1.0 / math.log2(i + 2)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(top_k, len(relevant_items))))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        p_k = float(np.mean(precisions)) if precisions else 0.0
        r_k = float(np.mean(recalls)) if recalls else 0.0
        ndcg_k = float(np.mean(ndcgs)) if ndcgs else 0.0
        hr_k = float(np.mean(hits)) if hits else 0.0

        print(f"\n[*] Kết quả chỉ số xếp hạng Top-{top_k} (Ranking Metrics trên sách được thích >= 4 sao):")
        print(f"    - Precision@{top_k}: {p_k * 100:.2f}%")
        print(f"    - Recall@{top_k}:    {r_k * 100:.2f}%")
        print(f"    - NDCG@{top_k}:      {ndcg_k * 100:.2f}%")
        print(f"    - Hit Rate@{top_k}:  {hr_k * 100:.2f}%")
        print(f"{'='*75}\n")

        return {
            "rmse": rmse,
            "mae": mae,
            f"precision@{top_k}": p_k,
            f"recall@{top_k}": r_k,
            f"ndcg@{top_k}": ndcg_k,
            f"hit_rate@{top_k}": hr_k,
        }


# =============================================================================
# 3. HIỂN THỊ LỊCH SỬ VÀ KẾT QUẢ GỢI Ý
# =============================================================================

def print_user_history(
    user_id: int,
    df_ratings: pd.DataFrame,
    df_books: pd.DataFrame,
    top_n: int = 5,
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
            ", ".join(row.genres[:4]) if isinstance(row.genres, list) and row.genres else "N/A"
        )
        avg_r = getattr(row, "rating_value", "N/A")
        print(
            f"{idx}. {row.title}\n"
            f"   Tác giả: {authors_display} | Thể loại: [{genres_display}]\n"
            f"   Điểm User chấm: {row.rating} sao | Goodreads TB: {avg_r}"
        )


# =============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH (MAIN ENTRYPOINT)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hệ Thống Gợi Ý Sách Sử Dụng Thuật Toán Lọc Dựa Trên Nội Dung (Content-Based Filtering)"
    )
    parser.add_argument(
        "--user_id",
        type=int,
        default=None,
        help="Mã user_id muốn thực hiện gợi ý cá nhân hóa (mặc định: tự động chọn user tích cực)",
    )
    parser.add_argument(
        "--book_id",
        type=int,
        default=None,
        help="Mã book_id muốn tìm các cuốn sách tương đồng nội dung (ví dụ: 1 cho The Hunger Games)",
    )
    parser.add_argument(
        "--book_title",
        type=str,
        default=None,
        help="Tên cuốn sách muốn tìm tương đồng (ví dụ: 'The Hobbit', 'Harry Potter')",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="weighted_avg",
        choices=["weighted_avg", "ridge"],
        help="Phương pháp xây dựng hồ sơ người dùng: 'weighted_avg' (Centroid có trọng số) hoặc 'ridge' (Hồi quy Ridge)",
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
        help="Thực hiện đánh giá mô hình trên tập kiểm thử (RMSE, MAE, Precision@K, Recall@K, NDCG@K)",
    )
    parser.add_argument(
        "--cold_start",
        action="store_true",
        help="Thực hiện demo khả năng xử lý Item Cold-Start (gợi ý sách mới xuất bản)",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=3000,
        help="Số lượng mẫu test dùng để đánh giá mô hình (mặc định: 3000)",
    )
    args = parser.parse_args()

    print("=" * 75)
    print(" HỆ THỐNG GỢI Ý SÁCH - CONTENT-BASED FILTERING (LỌC DỰA TRÊN NỘI DUNG)")
    print(f" Phương pháp hồ sơ người dùng: {args.method.upper()}")
    print("=" * 75)

    # 1. Tải metadata sách (ghép nối books_mapped.jsonl và books.csv)
    df_books = load_books_metadata()

    # 2. Tải và phân chia tập dữ liệu ratings
    train_df, test_df, df_ratings = load_ratings_dataset(
        num_users=args.users, num_books=args.books
    )

    # 3. Khởi tạo và huấn luyện mô hình Content-Based Filtering
    cb_model = ContentBasedRecommender(max_features=12000, ngram_range=(1, 2))
    cb_model.fit_items(df_books)
    cb_model.fit_users(train_df, method=args.method)

    # 4. Đánh giá nếu có cờ --eval
    if args.eval:
        cb_model.evaluate(test_df, sample_size=args.test_samples, top_k=args.top_n)

    # 5. Thực nghiệm 1: Gợi ý sách tương tự nội dung (Item-to-Item Content-Based)
    query_id = args.book_id if args.book_id is not None else 1  # Mặc định The Hunger Games (id=1)
    target_book_row = df_books[df_books["book_id"] == query_id]
    book_title_display = (
        target_book_row.iloc[0]["title"] if not target_book_row.empty else f"ID {query_id}"
    )
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
                ", ".join(row.genres[:4]) if isinstance(row.genres, list) and row.genres else "N/A"
            )
            desc_short = (
                str(row.description)[:100] + "..." if row.description else "Không có mô tả."
            )
            shared_str = " | ".join(row.shared_features) if row.shared_features else "Chung đặc trưng nội dung"
            print(
                f"{idx:2d}. {row.title} (ID: {row.book_id})\n"
                f"    - Tác giả: {authors_str} | Thể loại: [{genres_str}]\n"
                f"    - Độ tương đồng Cosine: {row.similarity_score:.4f}\n"
                f"    - Đặc trưng chung: [{shared_str}]\n"
                f"    - Tóm tắt: {desc_short}"
            )
    else:
        print("[!] Không tìm thấy sách tương tự.")

    # 6. Thực nghiệm 2: Gợi ý cá nhân hóa cho một người dùng (User-to-Item Content-Based)
    if args.user_id is not None and args.user_id in (
        cb_model.user_models if args.method == "ridge" else cb_model.user_profiles
    ):
        target_user = args.user_id
    else:
        # Chọn user có nhiều tương tác trong tập train
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
                ", ".join(row.genres[:4]) if isinstance(row.genres, list) and row.genres else "N/A"
            )
            avg_r = getattr(row, "rating_value", "N/A")
            print(
                f"{idx:2d}. {row.title} (ID: {row.book_id})\n"
                f"    - Tác giả: {authors_str} | Thể loại: [{genres_str}]\n"
                f"    - Điểm dự đoán: {row.predicted_rating:.2f} sao | Điểm TB hệ thống: {avg_r}\n"
                f"    - Lý do gợi ý: {row.explanation}"
            )
    else:
        print("[!] Không thể tạo danh sách gợi ý cho người dùng này.")

    # 7. Thực nghiệm 3: Demo Item Cold-Start nếu có cờ --cold_start
    if args.cold_start:
        cb_model.demo_cold_start(
            new_title="Dune: Part Three - The Prophet",
            new_authors=["Frank Herbert"],
            new_genres=["Science Fiction", "Dystopia", "Space Opera", "Fantasy"],
            new_description=(
                "A sweeping epic set in a desert world of Arrakis, focusing on politics, "
                "religion, ecology, and the destiny of Paul Atreides as a messiah."
            ),
            top_n=5,
        )

    print(f"\n{'='*75}")
    print("[+] Hoàn tất thực thi thành công chương trình Lọc Dựa Trên Nội Dung!")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
