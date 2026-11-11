# Agent trích sở thích (luồng 1)

Prompt đã dùng để tạo `pipeline/agent-data/prefs.json`, chạy ngày 11/11/2026, mỗi độc giả một lần. `{reader_file}` là `agent-data/reader_<id>.txt`, `{genres_file}` là `agent-data/genres.txt`. Output được ép theo JSON schema ở cuối file.

## Prompt

```text
You are the PREFERENCE-EXTRACTION agent in a university demo of a book recommender (dataset Goodbooks-10k). Your output is shown on screen as "luồng 1: gợi ý theo sở thích", so it must be faithful to the data.

Input: read {reader_file} (every book this reader rated; format "rating | title | authors | genres"). Some lines say "(no genres crawled)": they have a real title and author but no genre data. Genre vocabulary of the catalog with counts: {genres_file}. Book tags vocabulary is not available, so keep themes generic.

Produce a preference profile:
- genres: 4-6 genres the reader likes. "name" MUST be copied exactly from genres.txt. "weight" in 0..1 = share of the reader's 4-5 star books WITH crawled genres that carry this genre (count it with a script, round to 2 decimals). "evidence" = 2-3 exact titles from the file rated 4-5.
- authors: 2-4 authors the reader rated highly more than once, or rated 5 at least once with a series; "evidence" = exact titles. Copy the author name exactly as written in the file.
- themes: 4-6 short lowercase English keywords in Goodreads-tag style (e.g. "vampires", "mythology", "russian-literature") grounded in the history.
- avoid: 0-3 genres from genres.txt that appear mostly in this reader's 1-3 star books; evidence = exact titles with low ratings. Empty list if no clear signal.
- domain_vi: short Vietnamese label (max 6 words). summary_vi: 1-2 Vietnamese sentences, using correct Vietnamese.
Do not invent titles. Set reader_id = {reader_id}.
```

## Output schema

Mỗi lần chạy trả về một object như dưới đây. File trong `agent-data/` là mảng JSON gồm object của cả 3 độc giả.

```json
{
  "reader_id": 2043,
  "domain_vi": "Fantasy thiếu nhi, YA và manga",
  "summary_vi": "…",
  "genres": [{"name": "Fantasy", "weight": 0.92, "evidence": ["Eragon (The Inheritance Cycle #1)", "…"]}],
  "authors": [{"name": "Rick Riordan", "evidence": ["The Lightning Thief (Percy Jackson and the Olympians, #1)"]}],
  "themes": ["magic", "greek-mythology"],
  "avoid": [{"name": "Nonfiction", "evidence": ["The Prince"]}]
}
```

## Sau agent: lọc kho (code, `pipeline/prep.py`, hàm `pref_recs`)

Mỗi sách chưa đọc có ít nhất 1 thể loại khớp được chấm điểm:

```text
điểm = (Σ trọng số thể loại khớp + 0.3 × số chủ đề khớp tag + 1 nếu tác giả chính khớp − 1 × số thể loại cần tránh) × chất lượng
chất lượng = (điểm TB / 5) × min(1, log10(lượt chấm) / 6)
```

Giữ tối đa 2 cuốn mỗi tác giả để luồng không bị một series chiếm hết, rồi lấy top 10.
