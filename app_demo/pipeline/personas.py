"""Write each demo reader's full rating history and the catalog genre vocabulary for the agents.

Usage: python personas.py <goodbooks repo dir> <agent-data dir> <reader id> [<reader id> ...]
"""
import html, json, sys
from collections import Counter
from pathlib import Path
import pandas as pd

repo, out, reader_ids = Path(sys.argv[1]), Path(sys.argv[2]), [int(x) for x in sys.argv[3:]]
books = {}
for l in open(repo / "books_mapped.jsonl"):
    r = json.loads(l)
    books[r["book_id"]] = r
R = pd.read_csv(repo / "data/ratings.csv")
csv = pd.read_csv(repo / "data/books.csv").set_index("book_id")
for uid in reader_ids:
    ru = R[R.user_id == uid].sort_values("rating", ascending=False, kind="stable")
    lines = [f"Reader #{uid} ({len(ru)} ratings, mean {ru.rating.mean():.2f}). Format: rating | title | authors | genres"]
    for x in ru.itertuples():
        b, c = books[x.book_id], csv.loc[x.book_id]
        g = [g for g in (b.get("genres") or []) if g != "...more"][:6]
        # books.csv fallback, as prep.py does, for books the crawl did not find
        title = html.unescape(b.get("title") or c.title)
        authors = ", ".join(b.get("authors") or [c.authors])
        lines.append(f"{x.rating} | {title} | {html.unescape(authors)} | {', '.join(g) or '(no genres crawled)'}")
    (out / f"reader_{uid}.txt").write_text("\n".join(lines))
c = Counter(g for b in books.values() for g in (b.get("genres") or []) if g != "...more")
(out / "genres.txt").write_text("\n".join(f"{g}\t{n}" for g, n in c.most_common(200)))
print(sorted(p.name for p in out.iterdir()))
