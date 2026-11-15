"""Build data/demo-data.js for the book-recommender demo from the Goodbooks-10k repo.

Usage: python prep.py <goodbooks repo dir> <out .js or .json> <agent-data dir>

Stream 3 (ML) mirrors the repo's methods in reduced form: user-based CF and item-based CF with
k=25 neighbours (ucf.py), SVD with 50 factors standing in for the SGD MatrixEmbedding (ucf.py),
TF-IDF content filtering (content_filtering.py). Streams 1 and 2 read agent output from
<agent-data>/prefs.json and <agent-data>/webs.json.
"""
import html
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

repo, out, inputs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
K = 25
MAX_PER_AUTHOR = 2  # keeps the preference stream from becoming one series
MIN_SUPPORT = 0.5  # Σsim floor: without it, books reached only through near-zero similarities top the list
IGNORE = {
    "to-read", "currently-reading", "favorites", "favourites", "books-i-own", "owned", "owned-books",
    "default", "audiobook", "audiobooks", "audio", "ebook", "ebooks", "kindle", "library", "my-books",
    "i-own", "read-in-2015", "read-in-2016", "read-in-2017", "wish-list", "all-time-favorites",
}
def unesc(o):
    """Agent output quotes titles from the raw crawl, which is HTML-escaped up to twice."""
    if isinstance(o, str):
        return html.unescape(html.unescape(o))
    if isinstance(o, list):
        return [unesc(x) for x in o]
    return {k: unesc(v) for k, v in o.items()} if isinstance(o, dict) else o


PREFS = {p["reader_id"]: p for p in unesc(json.load(open(inputs / "prefs.json")))}
WEBS = {w["reader_id"]: w for w in unesc(json.load(open(inputs / "webs.json")))}

# --- books
rows = [json.loads(l) for l in open(repo / "books_mapped.jsonl")]
csv = pd.read_csv(repo / "data/books.csv").set_index("book_id")
tags = pd.read_csv(repo / "data/tags.csv")
bt = pd.read_csv(repo / "data/book_tags.csv").merge(tags, on="tag_id")
bt = bt[bt.tag_name.str.match(r"^[a-zA-Z0-9\s\-]+$", na=False) & ~bt.tag_name.isin(IGNORE)]
top_tags = bt.sort_values(["goodreads_book_id", "count"], ascending=[True, False]).groupby("goodreads_book_id").head(8)
tag_map = top_tags.groupby("goodreads_book_id").tag_name.apply(list).to_dict()

books = {}
for r in rows:
    bid = r["book_id"]
    c = csv.loc[bid]
    year = c.original_publication_year
    books[bid] = {
        "t": html.unescape(r.get("title") or c.title),
        "a": html.unescape(", ".join(" ".join(a.split()) for a in (r.get("authors") or [c.authors]))),
        "y": None if pd.isna(year) else int(year),
        "r": round(float(r.get("rating_value") or c.average_rating), 2),
        "n": int(r.get("ratings_count") or c.ratings_count),
        "p": int(r["num_pages"]) if str(r.get("num_pages") or "").isdigit() else None,  # some rows hold a format
        "g": [g for g in (r.get("genres") or []) if g != "...more"][:6],
        "tags": tag_map.get(int(c.goodreads_book_id), [])[:6],
        "d": re.sub(r"\s+", " ", r.get("description") or "").strip(),
    }
ids = sorted(books)
crawled = {r["book_id"]: bool(r.get("title")) for r in rows}
idx = {b: i for i, b in enumerate(ids)}


def author_token(a):
    """ASCII-fold so the token pattern keeps names like García Márquez or Saint-Exupéry."""
    folded = unicodedata.normalize("NFKD", a).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"[^a-z0-9]+", "_", folded).strip("_")
    return "author_" + name if name else ""  # non-Latin scripts fold to nothing; one shared token would fake similarity


def soup(b):
    x = books[b]
    toks = [f"genre_{g.lower().replace(' ', '_')}" for g in x["g"]] * 2
    toks += [t for t in map(author_token, x["a"].split(",")) if t] * 2
    toks += [f"tag_{t.replace('-', '_')}" for t in x["tags"]]
    return " ".join(toks) + " " + x["d"].lower()


vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_features=40000,
                      token_pattern=r"(?u)\b[a-z_][a-z0-9_]{2,}\b")
X = normalize(vec.fit_transform([soup(b) for b in ids]))
feat = vec.get_feature_names_out()


def label(w):
    for p, name in (("genre_", "Thể loại"), ("author_", "Tác giả"), ("tag_", "Tag")):
        if w.startswith(p):
            return f"{name}: {w[len(p):].replace('_', ' ').title() if p != 'tag_' else w[len(p):].replace('_', ' ')}"
    return f"Từ khoá: {w}"


def dense(v):
    return np.asarray(v.todense() if hasattr(v, "todense") else v).ravel()


def why(v1, v2, k=3):
    prod = dense(v1) * dense(v2)
    top = np.argsort(prod)[::-1][:k]
    return [label(feat[i]) for i in top if prod[i] > 0.001]


def similar(b, k=6, pool=None):
    s = (X @ X[idx[b]].T).toarray().ravel()
    s[idx[b]] = -1
    if pool is not None:
        s[~pool] = -1
    top = np.argsort(s)[::-1][:k]
    return [{"b": ids[i], "s": round(float(s[i]), 3), "why": why(X[idx[b]], X[i])} for i in top]


def ms(t0):
    return int((time.perf_counter() - t0) * 1000)


# --- ratings matrix (user-mean-centered) shared by item-CF and SVD
R = pd.read_csv(repo / "data/ratings.csv")
uids = np.sort(R.user_id.unique())
uix = {u: i for i, u in enumerate(uids)}
umean = R.groupby("user_id").rating.mean()
M = csr_matrix(((R.rating - R.user_id.map(umean)).values, (R.user_id.map(uix).values, R.book_id.map(idx).values)),
               shape=(len(uids), len(ids)))
t0 = time.perf_counter()
Mn = normalize(M.tocsc(), axis=0)
svd = TruncatedSVD(50, random_state=42)
P = svd.fit_transform(M)
Q = svd.components_
svd_ms = ms(t0)

# --- pick three contrasting real readers
counts = R.groupby("user_id").size()
cand = counts[(counts >= 90) & (counts <= 160)].index[:3000]
liked = R[R.user_id.isin(cand) & (R.rating == 5)]
liked = liked.assign(g=liked.book_id.map(lambda b: books[b]["g"]))
picked = []
for genre, persona in (("Fantasy", "Mê fantasy"), ("Classics", "Thích văn học kinh điển"), ("Romance", "Đọc romance")):
    grp = liked.groupby("user_id").g
    share = grp.apply(lambda s: sum(genre in g for g in s) / len(s))[grp.size() >= 20]
    share = share[~share.index.isin([p[0] for p in picked])]
    picked.append((int(share.idxmax()), persona, genre))


def ucf_recs(uid, ru, mu):
    """Mean-centered cosine on co-rated books with shrinkage, k=25 positive neighbours, >=3 neighbours per book."""
    co = R[R.book_id.isin(ru.index) & (R.user_id != uid)]
    co = co.assign(cv=co.rating - co.user_id.map(umean), cu=co.book_id.map(ru - mu))
    g = co.assign(dp=co.cv * co.cu, n2=co.cv ** 2, nu2=co.cu ** 2).groupby("user_id").agg(
        dp=("dp", "sum"), n2=("n2", "sum"), nu2=("nu2", "sum"), m=("dp", "size"))
    g = g[g.m >= 8]
    # cosine over co-rated books, shrunk by min(m, 30)/30 so neighbours with few shared books count less
    sim = g.dp / (np.sqrt(g.n2) * np.sqrt(g.nu2) + 1e-9) * np.minimum(g.m, 30) / 30
    nb = sim[sim > 0].nlargest(K)
    nr = R[R.user_id.isin(nb.index) & ~R.book_id.isin(ru.index)]
    nr = nr.assign(sim=nr.user_id.map(nb), dev=nr.rating - nr.user_id.map(umean))
    agg = nr.groupby("book_id").apply(lambda d: pd.Series({
        "pred": mu + (d.sim * d.dev).sum() / d.sim.abs().sum(), "cnt": len(d)}), include_groups=False)
    agg = agg[agg.cnt >= 3].sort_values(["pred", "cnt"], ascending=False).head(10)
    recs = []
    for b, a in agg.iterrows():
        d = nr[nr.book_id == b].sort_values("sim", ascending=False).head(3)
        recs.append({"b": int(b), "score": round(float(min(a.pred, 5)), 2), "cnt": int(a.cnt),
                     "nb": [{"u": int(x.user_id), "sim": round(float(x.sim), 2), "r": int(x.rating)} for x in d.itertuples()]})
    return recs, [{"u": int(u), "sim": round(float(v), 2)} for u, v in nb.head(5).items()], len(g)


def icf_recs(ru, mu):
    """ucf.py ItemBasedCollaborativeFiltering formula: r̂ = r̄ᵤ + Σ sim·(r − r̄ᵤ) / Σ|sim| over top-25 rated items."""
    rated = np.array([idx[b] for b in ru.index])
    S = (Mn[:, rated].T @ Mn).toarray()
    S[S <= 0] = 0
    dev = (ru - mu).values
    top = np.argsort(-S, axis=0)[:K]
    St = np.take_along_axis(S, top, 0)
    num, den, cnt = (St * dev[top]).sum(0), St.sum(0), (St > 0).sum(0)
    pred = mu + np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    pred[rated] = -np.inf
    pred[cnt < 3] = -np.inf
    pred[den < MIN_SUPPORT] = -np.inf
    recs = []
    for j in np.lexsort((-den, -pred))[:10]:
        best = np.argsort(-S[:, j])[:2]
        recs.append({"b": ids[j], "score": round(float(min(pred[j], 5)), 2), "support": round(float(den[j]), 2),
                     "via": [{"b": ids[rated[i]], "sim": round(float(S[i, j]), 2), "r": int(ru.iloc[i])} for i in best]})
    return recs


def mf_recs(uid, rated):
    pu = P[uix[uid]]
    sc = pu @ Q
    sc[rated] = -np.inf
    recs = []
    for j in np.argsort(-sc)[:10]:
        contrib = pu * Q[:, j]
        f = int(np.argmax(contrib))
        reps = [i for i in np.argsort(-(Q[f] * np.sign(pu[f]))) if i != j and i not in rated][:2]
        recs.append({"b": ids[j], "score": round(float(sc[j]), 3), "f": f, "term": round(float(contrib[f]), 3),
                     "reps": [ids[i] for i in reps]})
    return recs


def cbf_recs(ru, mu):
    """Mean-centered weighted centroid (Rocchio) of TF-IDF vectors, explained by shared features + anchor book."""
    rows_u = [idx[b] for b in ru.index]
    prof = normalize(np.asarray(X[rows_u].T @ (ru - mu).values).reshape(1, -1))
    s = (X @ prof.T).ravel()
    s[rows_u] = -1
    pos = [b for b in ru.index if ru[b] >= 4]
    recs = []
    for i in np.argsort(s)[::-1][:10]:
        anchor = max(pos, key=lambda b: (X[idx[b]] @ X[i].T).toarray()[0, 0])
        recs.append({"b": ids[i], "score": round(float(s[i]), 3), "why": why(prof, X[i]), "anchor": int(anchor)})
    return recs


def pref_recs(p, ru):
    """Rank the catalog by the agent's preference profile, weighted by book quality."""
    gw = {g["name"]: g["weight"] for g in p["genres"]}
    authors = {" ".join(a["name"].split()).lower() for a in p["authors"]}
    themes = {t.lower().replace(" ", "-") for t in p["themes"]}
    avoid = {a["name"] for a in p["avoid"]}
    res, n_match = [], 0
    for b in ids:
        if b in ru.index:
            continue
        x = books[b]
        hit_g = [g for g in x["g"] if g in gw]
        if not hit_g:
            continue
        n_match += 1
        s = sum(gw[g] for g in hit_g)
        hit_t = [t for t in x["tags"] if t in themes]
        first = x["a"].split(", ")[0]
        hit_a = [first] if first.lower() in authors else []  # later names are translators, illustrators, introductions
        hit_x = [g for g in x["g"] if g in avoid]
        s += 0.3 * len(hit_t) + (1.0 if hit_a else 0) - 1.0 * len(hit_x)
        q = x["r"] / 5 * min(1.0, np.log10(x["n"]) / 6)
        if s * q > 0:
            res.append((s * q, b, hit_g, hit_t, hit_a, hit_x))
    res.sort(key=lambda r: -r[0])
    per_author, recs = {}, []
    for sc, b, hg, ht, ha, hx in res:
        first = books[b]["a"].split(", ")[0]
        if per_author.get(first, 0) >= MAX_PER_AUTHOR:
            continue
        per_author[first] = per_author.get(first, 0) + 1
        recs.append({"b": b, "score": round(float(sc), 3), "genres": hg, "themes": ht, "authors": ha, "avoid": hx})
        if len(recs) == 10:
            break
    return recs, n_match, len(res)


def norm_title(t):
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t.lower())
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t).split())


title_ix = {}
for b in ids:
    title_ix.setdefault(norm_title(books[b]["t"]), []).append(b)


def match_catalog(title, author):
    last = author.split()[-1].lower() if author.split() else ""
    return next((b for b in title_ix.get(norm_title(title), []) if last in books[b]["a"].lower()), None)


users = []
for uid, persona, genre in picked:
    ru = R[R.user_id == uid].set_index("book_id").rating
    mu = ru.mean()
    rated = np.array([idx[b] for b in ru.index])
    timing = {}
    t0 = time.perf_counter(); ucf, nbs, n_users = ucf_recs(uid, ru, mu); timing["ucf"] = ms(t0)
    t0 = time.perf_counter(); icf = icf_recs(ru, mu); timing["icf"] = ms(t0)
    t0 = time.perf_counter(); mf = mf_recs(uid, rated); timing["mf"] = ms(t0)
    t0 = time.perf_counter(); cbf = cbf_recs(ru, mu); timing["cbf"] = ms(t0)
    t0 = time.perf_counter(); pref, n_match, pool = pref_recs(PREFS[uid], ru); timing["pref"] = ms(t0)
    web = WEBS[uid]
    for wb in web["books"]:
        wb["b"] = match_catalog(wb["title"], wb["author"])
    hist = ru.sort_values(ascending=False, kind="stable")  # stable: many 5-star ties
    users.append({"id": uid, "persona": persona, "genre": genre, "n": int(len(ru)), "mean": round(float(mu), 2),
                  "dist": [int((ru == k).sum()) for k in range(1, 6)],
                  "history": [{"b": int(b), "r": int(r)} for b, r in hist.head(12).items()],
                  "nb": nbs, "n_users": int(n_users), "timing": timing,
                  "ml": {"ucf": ucf, "icf": icf, "mf": mf, "cbf": cbf},
                  "pref": {"profile": PREFS[uid], "recs": pref, "match": n_match, "pool": pool}, "web": web,
                  "missing": int(sum(not crawled[b] for b in ru.index))})

# --- search set + similar lists for every shown book
explore = [b for b in sorted(ids, key=lambda b: -books[b]["n"])][:120]
shown = set(explore)
for u in users:
    shown |= {h["b"] for h in u["history"]}
    for algo in u["ml"].values():
        for x in algo:
            shown.add(x["b"])
            shown |= {v["b"] for v in x.get("via", [])} | set(x.get("reps", [])) | ({x["anchor"]} if "anchor" in x else set())
    shown |= {x["b"] for x in u["pref"]["recs"]} | {x["b"] for x in u["web"]["books"] if x["b"]}
sims = {b: similar(b) for b in shown}
shown |= {x["b"] for v in sims.values() for x in v}
# Outer books get neighbours from the demo set only, so no drawer is empty; UI flags these lists.
pool = np.isin(ids, list(shown))
local = sorted(shown - set(sims))
for b in local:
    sims[b] = similar(b, pool=pool)
out_books = {b: {**books[b], "d": books[b]["d"][:700]} for b in shown}
meta = {"n_books": len(ids), "n_users": len(uids), "n_ratings": int(len(R)), "svd_ms": svd_ms,
        "svd_var": round(float(svd.explained_variance_ratio_.sum()), 3)}
text = json.dumps({"books": out_books, "users": users, "explore": explore, "similar": sims, "local": local,
                   "meta": meta}, ensure_ascii=False, separators=(",", ":"))
# A .js file sets a global so index.html works when opened straight from disk (fetch() is blocked on file://).
out.write_text(f"window.DEMO_DATA = {text};\n" if out.suffix == ".js" else text)
print(len(out_books), "books;", [(u["id"], u["persona"], u["timing"]) for u in users], meta, out.stat().st_size, "bytes")
