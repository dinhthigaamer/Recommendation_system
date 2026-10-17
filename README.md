# Recommendation_system

Hệ thống gợi ý sách (Recommender System) sử dụng dataset Goodbooks-10k.

## 1. Các thuật toán & Module chính
- `ucf.py`: User-Based Collaborative Filtering (Lọc cộng tác dựa trên người dùng).
- `content_filtering.py`: Content-Based Filtering (Lọc dựa trên nội dung - TF-IDF & Cosine Similarity).
- `crawl_books.py`: Bộ thu thập dữ liệu sách (Goodreads Scrapy-Playwright crawler).
- `mapping.py`: Ánh xạ và chuẩn hóa dữ liệu sách giữa `books_data_en.jsonl` và `books.csv`.

## 2. Dữ liệu
- `goodbooks-10k/`: Chứa `books.csv`, `ratings.csv`, `book_tags.csv`, `tags.csv`.
- `books_mapped.jsonl`: Dữ liệu 10.000 cuốn sách đã crawl đầy đủ `description` và `genres`, kèm `id` chuẩn hóa.

## 3. Cài đặt & Sử dụng
```bash
# Tạo môi trường và cài đặt thư viện
pip install -r requirements.txt

# Chạy Lọc cộng tác (UCF)
python ucf.py

# Chạy Lọc dựa trên nội dung (CBF)
python content_filtering.py
```
