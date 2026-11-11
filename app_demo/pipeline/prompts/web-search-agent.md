# Agent tìm web (luồng 2)

Prompt đã dùng để tạo `pipeline/agent-data/webs.raw.json`, chạy ngày 11/11/2026, mỗi độc giả một lần. Agent có tool `WebSearch`. Bản dưới đây bỏ một chỉ dẫn nạp tool chỉ có ý nghĩa trong môi trường chạy lúc đó. `{reader_file}` là `agent-data/reader_<id>.txt` và `{hint}` là một dòng tóm tắt gu đọc do người chạy viết.

Lần chạy này, file lịch sử còn thiếu tên của 5 đến 12 cuốn mỗi độc giả, vì dữ liệu crawl không có. Lỗi này đã được sửa trong `personas.py` sau đó. Trang demo ghi rõ số dòng thiếu ở ô đầu vào của luồng 2.

## Prompt

```text
You are the WEB-SEARCH agent in a university demo of a book recommender. Your tool calls and their output are shown on screen as "luồng 2: gợi ý từ web", so everything must be real and traceable. Today is {today}.

Reader: read {reader_file} (every book this reader rated; format "rating | title | authors | genres"). Their taste in short: {hint}.

Task:
1. Load the WebSearch tool and run 3-4 web searches aimed at books published 2024-2026 that fit this reader (new releases, award lists, "best of 2025/2026" lists in their genres, new books by authors they love). Record each query string exactly as you sent it.
2. Pick 6 books. Each must come from your search results: source_url must be a URL that appeared in the results, snippet a short quote or close paraphrase (max 200 chars) of what that result says, source_name the site name. query_index = 0-based index into your queries list.
3. Exclude books the reader already rated. Prefer variety (not 6 books of one series).
4. why_vi: one Vietnamese sentence tying the book to this reader; related_history: 1-2 exact titles from the reader's file that justify it.
5. If you cannot verify a candidate's author/year from results, drop it and say so in "dropped".
Set reader_id = {reader_id}.
```

## Output schema

Mỗi lần chạy trả về một object như dưới đây. File trong `agent-data/` là mảng JSON gồm object của cả 3 độc giả.

```json
{
  "reader_id": 2043,
  "queries": [{"query": "best young adult fantasy books 2025", "purpose_vi": "…", "results_used": 2}],
  "books": [{
    "title": "Among Ghosts", "author": "Rachel Hartman", "year": 2025,
    "why_vi": "…", "related_history": ["Eragon (The Inheritance Cycle #1)"],
    "source_url": "https://…", "source_name": "Reactor", "snippet": "…", "query_index": 0
  }],
  "dropped": ["Tên sách: lý do loại"]
}
```

## Kiểm chứng nguồn

Sau khi agent chạy, một agent khác mở lại từng `source_url` để kiểm tên sách, tác giả, năm và snippet. 13 trên 18 nguồn đúng hoàn toàn. 5 cuốn đã được sửa trong `webs.json`, mỗi cuốn có trường `fix_note` ghi lý do:

- 3 trang nguồn chặn truy cập (HTTP 403), nên được thay bằng trang khác nói cùng thông tin.
- 2 cuốn ghi sai năm xuất bản.
- 1 snippet trích lệch tiêu đề danh sách.

`webs.raw.json` giữ nguyên output gốc của agent để so sánh.

Sau cùng, code trong `prep.py` (hàm `match_catalog`) so tên sách và họ tác giả với kho để gắn nhãn "Có trong kho" hoặc "Ngoài kho".
