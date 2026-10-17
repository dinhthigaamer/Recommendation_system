"""
Script ánh xạ dữ liệu sách đã crawl (books_data_en.jsonl) với danh sách gốc trong books.csv.

Nhiệm vụ:
    1. Đọc books.csv (Goodbooks-10k) để lấy bảng ánh xạ theo goodreads_book_id / book_id.
    2. Đọc books_data_en.jsonl:
       - Bổ sung trường `id` (chính là book_id từ 1 đến 10000 trong books.csv).
       - Loại bỏ các trường trung gian `input_title` và `input_author`.
    3. Lưu kết quả vào file JSONL mới (mặc định: `books_mapped.jsonl`), có tùy chọn sắp xếp theo id tăng dần.
    4. Tìm và in ra danh sách các cuốn sách có trong books.csv nhưng chưa có trong dữ liệu jsonl.

Cách chạy:
    python mapping.py
    python mapping.py -i books_data_en.jsonl -c goodbooks-10k/books.csv -o books_mapped.jsonl
    python mapping.py --save-missing missing_books.csv
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def find_file(candidates: List[str]) -> Optional[Path]:
    """Tìm file tồn tại đầu tiên trong danh sách ứng viên."""
    for cand in candidates:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def extract_goodreads_id(item: dict) -> Optional[str]:
    """Trích xuất goodreads_book_id từ trường dữ liệu hoặc từ URL."""
    gid = item.get("goodreads_book_id")
    if gid is not None and str(gid).strip():
        return str(gid).strip()

    url = item.get("url", "")
    match = re.search(r"/book/show/(\d+)", url)
    if match:
        return match.group(1)
    return None


def load_books_csv(csv_path: Path) -> Tuple[List[dict], Dict[str, dict], Dict[str, dict]]:
    """
    Đọc books.csv và tạo từ điển ánh xạ theo goodreads_book_id và book_id.
    """
    csv_rows = []
    by_gid = {}
    by_bid = {}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows.append(row)
            gid = str(row.get("goodreads_book_id", "")).strip()
            bid = str(row.get("book_id", "")).strip()
            if gid:
                by_gid[gid] = row
            if bid:
                by_bid[bid] = row

    return csv_rows, by_gid, by_bid


def map_and_clean_books(
    jsonl_path: Path, by_gid: Dict[str, dict], by_bid: Dict[str, dict]
) -> Tuple[List[dict], Set[int], int]:
    """
    Đọc file JSONL, bổ sung trường id, loại bỏ input_title / input_author.
    Trả về:
        mapped_items: danh sách các bản ghi đã ánh xạ và làm sạch.
        matched_book_ids: tập các book_id (int) đã được ánh xạ thành công.
        unmatched_count: số dòng trong JSONL không ánh xạ được sang books.csv.
    """
    mapped_items = []
    matched_book_ids: Set[int] = set()
    unmatched_count = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[!] Cảnh báo: Lỗi cú pháp JSON ở dòng {line_num}")
                continue

            # Bỏ input_title và input_author
            item.pop("input_title", None)
            item.pop("input_author", None)

            gid = extract_goodreads_id(item)
            bid = str(item.get("book_id") or "").strip()

            csv_row = by_gid.get(gid) if gid else None
            if not csv_row and bid:
                csv_row = by_bid.get(bid)

            if csv_row:
                book_id = int(csv_row["book_id"])
                matched_book_ids.add(book_id)

                # Bổ sung id lên đầu dictionary
                new_item = {
                    "id": book_id,
                    "book_id": book_id,
                    "goodreads_book_id": int(csv_row["goodreads_book_id"]),
                }

                # Đưa các trường còn lại vào
                for k, v in item.items():
                    if k not in new_item:
                        new_item[k] = v

                mapped_items.append(new_item)
            else:
                unmatched_count += 1
                mapped_items.append(item)

    return mapped_items, matched_book_ids, unmatched_count


def print_missing_books(missing_books: List[dict]) -> None:
    """In danh sách các sách có trong books.csv nhưng không có trong JSONL."""
    if not missing_books:
        print("\n" + "=" * 90)
        print(" [HOÀN HẢO] Toàn bộ 10,000 / 10,000 cuốn sách trong books.csv đều đã có trong JSONL!")
        print(" Không có cuốn sách nào bị thiếu.")
        print("=" * 90 + "\n")
        return

    print("\n" + "=" * 90)
    print(f" DANH SÁCH SÁCH CÓ TRONG BOOKS.CSV NHƯNG CHƯA CÓ TRONG JSONL ({len(missing_books)} cuốn)")
    print("=" * 90)
    header = f"{'STT':<4} | {'ID':<6} | {'Goodreads ID':<13} | {'Ngôn ngữ':<8} | {'Tên sách':<45} | {'Tác giả'}"
    print(header)
    print("-" * 90)

    for idx, row in enumerate(missing_books, start=1):
        bid = row.get("book_id", "")
        gid = row.get("goodreads_book_id", "")
        lang = row.get("language_code", "") or "N/A"
        title = (row.get("title", "")[:42] + "...") if len(row.get("title", "")) > 45 else row.get("title", "")
        authors = (row.get("authors", "")[:25] + "...") if len(row.get("authors", "")) > 28 else row.get("authors", "")
        print(f"{idx:<4} | {bid:<6} | {gid:<13} | {lang:<8} | {title:<45} | {authors}")

    print("=" * 90)
    print(f"[+] Gợi ý: Có thể xem chi tiết link sách bằng định dạng: https://www.goodreads.com/book/show/<Goodreads ID>\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ánh xạ books_data_en.jsonl với books.csv, bổ sung trường id và bỏ input_title/input_author."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="Đường dẫn file JSONL đầu vào (mặc định: tự tìm books_data_en.jsonl hoặc book_data_en.jsonl)",
    )
    parser.add_argument(
        "-c",
        "--csv",
        default=None,
        help="Đường dẫn file books.csv (mặc định: tự tìm goodbooks-10k/books.csv hoặc books.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="books_mapped.jsonl",
        help="Đường dẫn file JSONL đầu ra sau khi ánh xạ (mặc định: books_mapped.jsonl)",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Không sắp xếp danh sách kết quả theo id tăng dần (giữ nguyên thứ tự ban đầu)",
    )
    parser.add_argument(
        "--save-missing",
        default=None,
        help="Tùy chọn lưu danh sách sách bị thiếu ra file riêng (hỗ trợ .csv hoặc .jsonl)",
    )
    args = parser.parse_args()

    # 1. Tìm file CSV
    csv_file = args.csv or find_file(["goodbooks-10k/books.csv", "books.csv"])
    if not csv_file:
        raise SystemExit("[-] Lỗi: Không tìm thấy file books.csv! Vui lòng chỉ định bằng -c/--csv.")

    # 2. Tìm file JSONL
    jsonl_file = args.input or find_file(["books_data_en.jsonl", "book_data_en.jsonl"])
    if not jsonl_file:
        raise SystemExit("[-] Lỗi: Không tìm thấy file JSONL! Vui lòng chỉ định bằng -i/--input.")

    print(f"[*] File CSV nguồn : {csv_file}")
    print(f"[*] File JSONL nguồn: {jsonl_file}")
    print(f"[*] File JSONL đích : {args.output}")

    # 3. Đọc dữ liệu từ CSV
    csv_rows, by_gid, by_bid = load_books_csv(Path(csv_file))
    print(f"[*] Đã tải {len(csv_rows):,} cuốn sách từ {csv_file}.")

    # 4. Ánh xạ và xử lý dữ liệu JSONL
    mapped_items, matched_book_ids, unmatched_count = map_and_clean_books(
        Path(jsonl_file), by_gid, by_bid
    )
    print(f"[*] Đã xử lý {len(mapped_items):,} bản ghi từ {jsonl_file}.")
    print(f"[*] Số bản ghi ánh xạ thành công có id: {len(matched_book_ids):,}")
    if unmatched_count > 0:
        print(f"[!] Cảnh báo: Có {unmatched_count} bản ghi không tìm thấy trong books.csv!")

    # 5. Sắp xếp theo id tăng dần nếu không chọn --no-sort
    if not args.no_sort:
        mapped_items.sort(key=lambda x: x.get("id", float("inf")))
        print("[*] Đã sắp xếp các bản ghi theo trường 'id' tăng dần (1 -> 10000).")

    # 6. Ghi ra file JSONL mới
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in mapped_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[+] Đã lưu thành công dữ liệu mới vào: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")

    # 7. Tìm các sách có trong books.csv mà trong jsonl không có
    missing_books = [r for r in csv_rows if int(r["book_id"]) not in matched_book_ids]
    print_missing_books(missing_books)

    # 8. Lưu file sách thiếu nếu có yêu cầu
    if args.save_missing:
        missing_path = Path(args.save_missing)
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        if missing_path.suffix.lower() == ".csv":
            if missing_books:
                with missing_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(missing_books[0].keys()))
                    writer.writeheader()
                    writer.writerows(missing_books)
            print(f"[+] Đã lưu danh sách sách thiếu ra file CSV: {missing_path}")
        else:
            with missing_path.open("w", encoding="utf-8") as f:
                for row in missing_books:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[+] Đã lưu danh sách sách thiếu ra file JSONL: {missing_path}")


if __name__ == "__main__":
    main()
