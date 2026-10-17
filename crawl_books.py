"""
Goodreads spider (scrapy-playwright) để crawl thông tin sách tiếng Anh từ Goodbooks-10k.

Luồng chạy:
    1. Đọc file `books_data_en.jsonl` (hoặc `book_data_en.jsonl`) để lấy danh sách các sách đã crawl.
    2. Duyệt file `books.csv` (lọc sách tiếng Anh):
       - Nếu tên sách (hoặc id) đã có trong dữ liệu đã crawl thì bỏ qua ("thôi").
       - Nếu chưa có: tạo link chi tiết Goodreads với id tương ứng:
         https://www.goodreads.com/book/show/<goodreads_book_id>
    3. Mở trực tiếp link để crawl thông tin chi tiết qua JSON-LD / HTML (nếu lỗi hoặc không tìm thấy,
       có cơ chế dự phòng fallback tìm kiếm qua search).
    4. Ghi tiếp vào file output (mặc định nối tiếp vào `books_data_en.jsonl`).

Cài đặt:
    pip install scrapy scrapy-playwright
    playwright install chromium

Chạy:
    python crawl_books.py                                      # Tự động nhận diện books.csv & books_data_en.jsonl
    python crawl_books.py --limit 10                           # Crawl thử 10 cuốn chưa có
    python crawl_books.py -c goodbooks-10k/books.csv -o books_data_en.jsonl
    python crawl_books.py --headed                             # Chạy hiển thị cửa sổ trình duyệt để quan sát
    python crawl_books.py --use-search                         # Dùng luồng tìm kiếm từ khóa thay vì đi thẳng URL
"""
import argparse
import csv
import json
import logging
import re
from pathlib import Path
from urllib.parse import quote_plus

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy_playwright.page import PageMethod

BASE_URL = "https://www.goodreads.com"

# Timeout cấu hình (mili giây)
NAV_TIMEOUT_MS = 60_000
SELECTOR_TIMEOUT_MS = 25_000
GOTO_KWARGS = {"wait_until": "domcontentloaded", "timeout": NAV_TIMEOUT_MS}

# Các mã ngôn ngữ tiếng Anh trong goodbooks-10k
ENGLISH_LANG_CODES = {"", "en", "eng", "en-us", "en-gb", "en-ca"}

BOOK_LINK_SELECTOR = (
    "a.bookTitle::attr(href), "
    "[data-testid='book-item-title'] a::attr(href)"
)


def clean_text(value):
    """Gộp các khoảng trắng / xuống dòng liên tiếp và cắt 2 đầu chuỗi."""
    return " ".join(str(value or "").split())


def should_abort(request):
    """Bỏ qua tài nguyên ảnh / media / font để tăng tốc độ tải trang."""
    return request.resource_type in {"image", "media", "font"}


def find_file(candidates):
    """Tìm file tồn tại đầu tiên trong danh sách ứng viên."""
    for cand in candidates:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def load_crawled_books(file_path):
    """
    Đọc file JSONL chứa các sách đã crawl.
    Trả về:
        crawled_titles: set các tên sách đã chuẩn hóa (viết thường, gọn khoảng trắng).
        crawled_ids: set các goodreads_book_id đã crawl.
    """
    crawled_titles = set()
    crawled_ids = set()

    if not file_path or not Path(file_path).exists():
        return crawled_titles, crawled_ids

    with Path(file_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            # Lưu cả input_title và title đã crawl được để so khớp chính xác
            t1 = clean_text(data.get("input_title", "")).lower()
            t2 = clean_text(data.get("title", "")).lower()
            if t1:
                crawled_titles.add(t1)
            if t2:
                crawled_titles.add(t2)

            gid = str(data.get("goodreads_book_id", "")).strip()
            if gid:
                crawled_ids.add(gid)

            # Trích xuất ID từ url nếu có (dạng /book/show/<id>)
            url = data.get("url", "")
            m = re.search(r"/book/show/(\d+)", url)
            if m:
                crawled_ids.add(m.group(1))

    return crawled_titles, crawled_ids


def load_books_to_crawl(csv_path, crawled_titles, crawled_ids, filter_english=True, limit=0):
    """
    Duyệt danh sách sách trong books.csv:
    - Nếu tên sách hoặc id đã có thì thôi (bỏ qua).
    - Còn không thì tạo thông tin sách kèm link Goodreads tương ứng với id.
    """
    to_crawl = []
    if not csv_path or not Path(csv_path).exists():
        raise FileNotFoundError(f"Không tìm thấy file books.csv tại: {csv_path}")

    with Path(csv_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = clean_text(row.get("title", ""))
            author = clean_text(row.get("authors", ""))
            gid = str(row.get("goodreads_book_id", "")).strip()
            bid = str(row.get("book_id", "")).strip()
            lang_code = row.get("language_code", "").strip().lower()

            if not title:
                continue

            # Lọc sách tiếng Anh nếu được yêu cầu
            if filter_english:
                if lang_code not in ENGLISH_LANG_CODES and not lang_code.startswith("en"):
                    continue

            # Kiểm tra xem tên sách hoặc id đã crawl chưa
            norm_title = title.lower()
            # if norm_title in crawled_titles:
            #     continue
            if gid and gid in crawled_ids:
                continue

            # Tạo link với id tương ứng
            direct_id = gid or bid
            book_url = f"{BASE_URL}/book/show/{direct_id}" if direct_id else None

            item = {
                "title": title,
                "author": author,
                "goodreads_book_id": gid,
                "book_id": bid,
                "isbn": row.get("isbn", "").strip(),
                "image_url": row.get("image_url", "").strip(),
                "direct_url": book_url,
            }
            to_crawl.append(item)

            if limit > 0 and len(to_crawl) >= limit:
                break

    return to_crawl


class GoodreadsSpider(scrapy.Spider):
    name = "goodreads"

    custom_settings = {
        # Cấu hình scrapy-playwright
        "DOWNLOAD_HANDLERS": {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "PLAYWRIGHT_LAUNCH_OPTIONS": {
            "headless": True,
            "slow_mo": 200,
        },
        "PLAYWRIGHT_CONTEXTS": {
            "default": {
                "user_data_dir": "./chromium_profile",
                "headless": True,
                "slow_mo": 200,
            }
        },
        "PLAYWRIGHT_ABORT_REQUEST": should_abort,
        "PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT": NAV_TIMEOUT_MS,
        "DOWNLOAD_TIMEOUT": 180,
        # Lịch sự với server Goodreads
        "CONCURRENT_REQUESTS": 1,
        "DOWNLOAD_DELAY": 2,
        "RETRY_TIMES": 2,
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, books_to_crawl=None, use_search=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.books_to_crawl = books_to_crawl or []
        self.use_search = str(use_search).lower() in ("1", "true", "yes")

    def _get_start_requests(self):
        for book in self.books_to_crawl:
            if self.use_search or not book.get("direct_url"):
                yield self._build_search_request(book)
            else:
                yield self._build_direct_request(book)

    async def start(self):  # Scrapy >= 2.13
        self.logger.info("Bắt đầu crawl %d cuốn sách...", len(self.books_to_crawl))
        for request in self._get_start_requests():
            yield request

    def start_requests(self):  # Scrapy < 2.13
        self.logger.info("Bắt đầu crawl %d cuốn sách...", len(self.books_to_crawl))
        yield from self._get_start_requests()

    # ------------------------------------------------------------------
    # Luồng 1: Mở thẳng URL sách từ id trong books.csv
    # ------------------------------------------------------------------
    def _build_direct_request(self, book):
        url = book["direct_url"]
        return scrapy.Request(
            url=url,
            callback=self.parse_book,
            errback=self.on_error,
            cb_kwargs={"book": book, "from_search": False},
            meta={
                "playwright": True,
                "playwright_context": "default",
                "playwright_page_goto_kwargs": GOTO_KWARGS,
                "playwright_page_methods": [
                    PageMethod(
                        "wait_for_function",
                        "() => document.querySelector(\"script[type='application/ld+json']\") "
                        "|| document.body.innerText.length > 200",
                        timeout=SELECTOR_TIMEOUT_MS,
                    ),
                ],
                "handle_httpstatus_list": [404, 410],
            },
            dont_filter=True,
        )

    # ------------------------------------------------------------------
    # Luồng 2: Tìm kiếm theo từ khóa nếu được yêu cầu hoặc khi direct link lỗi
    # ------------------------------------------------------------------
    def _build_search_request(self, book, use_author=True):
        query = f'{book["title"]} {book["author"]}'.strip() if use_author and book.get("author") else book["title"]
        search_url = f"{BASE_URL}/search?q={quote_plus(query)}"

        return scrapy.Request(
            url=search_url,
            callback=self.parse_search,
            errback=self.on_error,
            cb_kwargs={"book": book, "used_author": use_author},
            meta={
                "playwright": True,
                "playwright_context": "default",
                "playwright_page_goto_kwargs": GOTO_KWARGS,
                "playwright_page_methods": [
                    PageMethod(
                        "wait_for_selector",
                        BOOK_LINK_SELECTOR,
                        timeout=SELECTOR_TIMEOUT_MS,
                    ),
                ],
            },
            dont_filter=True,
        )

    async def parse_search(self, response, book, used_author):
        href = response.css(BOOK_LINK_SELECTOR).get()

        if not href:
            if used_author and book.get("author"):
                self.logger.info("Không tìm thấy với tác giả, thử lại chỉ với tên sách: %s", book["title"])
                yield self._build_search_request(book, use_author=False)
                return

            self.logger.warning("Không tìm thấy kết quả tìm kiếm cho: %s", book["title"])
            yield {
                "input_title": book["title"],
                "input_author": book["author"],
                "goodreads_book_id": book.get("goodreads_book_id"),
                "book_id": book.get("book_id"),
                "found": False,
                "url": response.url,
            }
            return

        yield response.follow(
            href,
            callback=self.parse_book,
            errback=self.on_error,
            cb_kwargs={"book": book, "from_search": True},
            meta={
                "playwright": True,
                "playwright_context": "default",
                "playwright_page_goto_kwargs": GOTO_KWARGS,
                "playwright_page_methods": [
                    PageMethod(
                        "wait_for_function",
                        "() => document.querySelector(\"script[type='application/ld+json']\") "
                        "|| document.body.innerText.length > 200",
                        timeout=SELECTOR_TIMEOUT_MS,
                    ),
                ],
                "handle_httpstatus_list": [404, 410],
            },
        )

    # ------------------------------------------------------------------
    # Trích xuất dữ liệu từ trang chi tiết sách
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_book_ld(response):
        """Trích xuất đối tượng JSON-LD có @type là 'Book'."""
        for raw in response.css("script[type='application/ld+json']::text").getall():
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            candidates = data if isinstance(data, list) else [data]
            for item in candidates:
                if isinstance(item, dict) and item.get("@type") == "Book":
                    return item
        return {}

    def parse_book(self, response, book, from_search=False):
        # Nếu URL trực tiếp bị 404/410, tự động chuyển sang search
        if response.status in (404, 410):
            if not from_search:
                self.logger.warning("URL id 404 (%s) cho '%s', chuyển sang tìm kiếm...", response.url, book["title"])
                yield self._build_search_request(book)
            else:
                self.logger.warning("Không tìm thấy trang sách: %s", response.url)
                yield {
                    "input_title": book["title"],
                    "input_author": book["author"],
                    "goodreads_book_id": book.get("goodreads_book_id"),
                    "book_id": book.get("book_id"),
                    "found": False,
                    "url": response.url,
                }
            return

        ld = self._extract_book_ld(response)

        # Trích xuất tác giả
        authors = ld.get("author") or []
        if isinstance(authors, dict):
            authors = [authors]
        author_names = [a.get("name") for a in authors if isinstance(a, dict) and a.get("name")]
        if not author_names:
            author_names = [
                a.strip()
                for a in response.css("[data-testid='name']::text").getall()
                if a.strip()
            ]
        if not author_names and book.get("author"):
            author_names = [a.strip() for a in book["author"].split(",") if a.strip()]

        # Trích xuất điểm đánh giá
        rating = ld.get("aggregateRating") or {}
        rating_val = rating.get("ratingValue")
        if rating_val is None:
            rv_text = (response.css(".RatingStatistics__rating::text").get() or "").strip()
            try:
                rating_val = float(rv_text) if rv_text else None
            except ValueError:
                rating_val = rv_text or None

        # Trích xuất mô tả
        desc_parts = [
            t.strip()
            for t in response.css("[data-testid='description'] span.Formatted ::text").getall()
            if t.strip()
        ]
        description = " ".join(desc_parts) if desc_parts else ld.get("description")

        # Trích xuất thể loại
        genres = [
            g.strip()
            for g in response.css("[data-testid='genresList'] .Button__labelItem::text").getall()
            if g.strip()
        ]

        # Trích xuất số trang
        num_pages = ld.get("numberOfPages") or response.css("[data-testid='pagesFormat']::text").get()
        if isinstance(num_pages, str):
            page_match = re.search(r"(\d+)", num_pages)
            if page_match:
                try:
                    num_pages = int(page_match.group(1))
                except ValueError:
                    pass

        # Trích xuất tên sách
        title = ld.get("name") or response.css("h1[data-testid='bookTitle']::text").get() or book["title"]

        yield {
            "input_title": book["title"],
            "input_author": book["author"],
            "found": True,
            "url": response.url,
            "title": title,
            "authors": author_names,
            "isbn": ld.get("isbn") or book.get("isbn") or None,
            "language": ld.get("inLanguage") or None,
            "book_format": ld.get("bookFormat") or None,
            "num_pages": num_pages,
            "rating_value": rating_val,
            "ratings_count": rating.get("ratingCount"),
            "reviews_count": rating.get("reviewCount"),
            "genres": genres,
            "description": description or None,
            "publication_info": response.css("[data-testid='publicationInfo']::text").get() or None,
            "cover_image": ld.get("image") or response.css(".BookCover__image img::attr(src)").get() or book.get("image_url") or None,
            "goodreads_book_id": book.get("goodreads_book_id"),
            "book_id": book.get("book_id"),
        }

    def on_error(self, failure):
        self.logger.error("Lỗi request: %s - %r", failure.request.url, failure.value)


FORMAT_BY_EXTENSION = {
    ".json": "json",
    ".jsonl": "jsonlines",
    ".jl": "jsonlines",
    ".csv": "csv",
    ".xml": "xml",
}


def build_feeds(output_path, overwrite=False):
    """Tạo cấu hình FEEDS của Scrapy; tự nhận diện định dạng từ đuôi file."""
    path = Path(output_path)
    fmt = FORMAT_BY_EXTENSION.get(path.suffix.lower())
    if fmt is None:
        raise SystemExit(
            f"Đuôi file '{path.suffix}' không được hỗ trợ. "
            f"Dùng một trong: {', '.join(FORMAT_BY_EXTENSION)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"format": fmt, "encoding": "utf-8", "overwrite": overwrite}
    if fmt == "json":
        options["indent"] = 2
    return {str(path): options}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl thông tin sách tiếng Anh từ Goodreads dựa theo books.csv (bỏ qua sách đã có trong file jsonl).")
    parser.add_argument("-c", "--csv", default=None,
                        help="Đường dẫn tới file books.csv (mặc định: tự tìm goodbooks-10k/books.csv hoặc books.csv)")
    parser.add_argument("--crawled", default=None,
                        help="Đường dẫn file đã crawl (mặc định: tự tìm books_data_en.jsonl hoặc book_data_en.jsonl)")
    parser.add_argument("-o", "--output", default="books_data_en.jsonl",
                        help="Đường dẫn file lưu kết quả (mặc định: books_data_en.jsonl)")
    parser.add_argument("--append", action="store_true", default=None,
                        help="Ghi nối tiếp vào file output nếu đã có (mặc định: BẬT)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ghi đè file output thay vì ghi nối tiếp")
    parser.add_argument("--limit", type=int, default=0,
                        help="Giới hạn số lượng sách cần crawl trong lần chạy này (0 = toàn bộ)")
    parser.add_argument("--all-langs", action="store_true",
                        help="Crawl tất cả sách trong books.csv (mặc định: chỉ lọc sách tiếng Anh)")
    parser.add_argument("--use-search", action="store_true",
                        help="Tìm kiếm từ khóa thay vì đi thẳng tới URL theo id")
    parser.add_argument("--headed", action="store_true",
                        help="Mở cửa sổ trình duyệt (mặc định chạy headless)")
    args = parser.parse_args()

    # 1. Xác định file books.csv
    csv_file = args.csv or find_file(["goodbooks-10k/books.csv", "books.csv"])
    if not csv_file:
        raise SystemExit("Lỗi: Không tìm thấy file books.csv! Hãy chỉ định qua tham số -c/--csv.")

    # 2. Xác định file đã crawl
    crawled_file = args.crawled or find_file(["books_data_en.jsonl", "book_data_en.jsonl", args.output])

    # 3. Đọc dữ liệu sách đã crawl
    print(f"[*] Đang tải sách đã crawl từ: {crawled_file or 'Chưa có file đã crawl'}")
    crawled_titles, crawled_ids = load_crawled_books(crawled_file)
    print(f"[*] Đã nhận diện {len(crawled_titles)} tên sách và {len(crawled_ids)} ID đã crawl trước đó.")

    # 4. Đọc danh sách sách từ books.csv và loại bỏ các sách đã có
    filter_en = not args.all_langs
    books_to_crawl = load_books_to_crawl(
        csv_file,
        crawled_titles=crawled_titles,
        crawled_ids=crawled_ids,
        filter_english=filter_en,
        limit=args.limit,
    )
    print(f"[*] Tổng số sách cần crawl trong đợt này: {len(books_to_crawl)}")
    if not books_to_crawl:
        print("[+] Toàn bộ sách đã được crawl xong! Không còn sách mới.")
        raise SystemExit(0)

    # 5. Cấu hình chế độ hiển thị trình duyệt nếu dùng --headed
    if args.headed:
        GoodreadsSpider.custom_settings["PLAYWRIGHT_LAUNCH_OPTIONS"]["headless"] = False
        GoodreadsSpider.custom_settings["PLAYWRIGHT_CONTEXTS"]["default"]["headless"] = False

    # 6. Cấu hình chế độ ghi file (mặc định append nối tiếp để không mất dữ liệu cũ)
    overwrite_feed = args.overwrite if args.overwrite else (args.append is False)

    process = CrawlerProcess(
        settings={"FEEDS": build_feeds(args.output, overwrite=overwrite_feed)})
    process.crawl(
        GoodreadsSpider,
        books_to_crawl=books_to_crawl,
        use_search=args.use_search,
    )
    process.start()
