// Data is precomputed by pipeline/prep.py into data/demo-data.js (a script, so the page also works from file://).
const D = window.DEMO_DATA;
const B = D.books, LOCAL = new Set(D.local);
const nf = new Intl.NumberFormat('vi-VN');
const view = document.getElementById('view'), drawer = document.getElementById('drawer');
const state = { user: null, page: 'login', algo: 'ucf', more: {}, q: '', genre: '', sort: 'n' };
const U = () => D.users[state.user];

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const short = t => t.replace(/\s*\([^)]*\)\s*$/, '');
const series = t => (t.match(/\(([^)]*)\)\s*$/) || [])[1] || '';
const hash = s => { let h = 7; for (const ch of s) h = (h * 31 + ch.codePointAt(0)) | 0; return Math.abs(h); };
const pct = (v, max) => Math.max(2, Math.round(v / max * 100));
const first = a => a.split(',')[0].trim();
const stars = n => '★'.repeat(n);
const safeUrl = u => /^https:\/\/[^\s"'<>]+$/.test(u) ? u : '';

function cover(title, author, size) {
  const who = first(author).split(' ').pop();
  return `<div class="cover c${hash(title) % 8}${size ? ' ' + size : ''}" aria-hidden="true"><span class="cv-t" lang="en">${esc(short(title))}</span><span class="cv-a">${esc(who)}</span></div>`;
}
const bookCover = (id, size) => cover(B[id].t, B[id].a, size);
const coverBtn = (id, size) => `<button class="cover-btn" data-go="book-${id}" aria-label="Mở ${esc(B[id].t)}">${bookCover(id, size)}</button>`;
const titleBtn = id => `<button class="title-btn" data-go="book-${id}">${esc(short(B[id].t))}</button>`;
const row = (id, body) => `<div class="row">${coverBtn(id, 'sm')}<div class="row-body">${titleBtn(id)}<span class="author">${esc(B[id].a)}${B[id].y ? ` · ${B[id].y}` : ''}</span>${body}</div></div>`;
const tile = (id, extra) => `<div class="tile">${coverBtn(id)}<span class="t">${esc(short(B[id].t))}</span><span class="m">${extra || `<span class="star">★</span> <span class="num">${B[id].r.toFixed(2)}</span> · ${esc(first(B[id].a))}`}</span></div>`;
const step = (n, title, tags, body) => `<li class="step"><span class="dot">${n}</span><div class="step-body"><div class="step-title"><b>${title}</b>${tags}</div>${body}</div></li>`;
const moreBtn = (key, total) => total > 5 ? `<button class="more" data-more="${key}">${state.more[key] ? 'Thu gọn' : `Xem cả ${total} cuốn`}</button>` : '';
const take = (list, key) => state.more[key] ? list : list.slice(0, 5);
const AVATAR = ['var(--c4)', 'var(--c1)', 'var(--c0)'];

/* ---------- login ---------- */
function renderLogin() {
  view.innerHTML = `<div class="login">
    <div class="head"><div><span class="eyebrow">Bản demo · dữ liệu Goodbooks-10k + mô tả đã crawl</span><h1>Đăng nhập</h1>
      <p>Chọn một tài khoản demo. Mỗi tài khoản là một độc giả thật trong <span class="num">ratings.csv</span>; lịch sử chấm điểm của họ là đầu vào cho cả 3 luồng gợi ý.</p></div></div>
    <div class="accounts">${D.users.map((u, i) => `<button class="account" data-login="${i}">
      <span class="avatar" style="background:${AVATAR[i]}">#${String(u.id).slice(-2)}</span>
      <span><b>${esc(u.pref.profile.domain_vi)}</b><span class="num note">Độc giả #${u.id} · ${u.n} lượt chấm · TB ${u.mean.toFixed(2)}★</span></span>
      <span class="go">Đăng nhập</span></button>`).join('')}</div>
    <section><span class="eyebrow">Sau khi đăng nhập</span>
      <div class="flow"><span class="pill">Lịch sử chấm điểm</span><span class="arrow">→</span>
        <span class="pill p1">1 · Theo sở thích</span><span class="pill p2">2 · Từ web</span><span class="pill p3">3 · Thuật toán ML</span>
        <span class="arrow">→</span><span class="pill">Trang chủ: 3 cột gợi ý</span><span class="arrow">→</span><span class="pill">Tab Tìm kiếm</span></div>
      <p class="note">Ba luồng chạy độc lập. Kết quả trong bản demo được tính sẵn ngày 11/11/2026: luồng 1 và 2 do LLM agent chạy thật, luồng 3 chạy thuật toán trên toàn bộ ${nf.format(D.meta.n_ratings)} lượt chấm.</p>
    </section></div>`;
}

/* ---------- lane 1: preferences ---------- */
function lanePref(u) {
  const p = u.pref.profile, maxW = Math.max(...p.genres.map(g => g.weight));
  const recs = take(u.pref.recs, 'pref').map(x => {
    const chips = [...x.genres.slice(0, 3).map(g => `<span class="chip">${esc(g)}</span>`),
      ...x.authors.map(a => `<span class="chip">Tác giả: ${esc(a)}</span>`),
      ...x.themes.map(t => `<span class="chip k">#${esc(t)}</span>`),
      ...x.avoid.map(a => `<span class="chip x">${esc(a)}</span>`)].join('');
    return row(x.b, `<div class="score"><span class="num">điểm ${x.score.toFixed(3)}</span></div><div class="chips">${chips}</div>`);
  }).join('');
  return `<article class="lane l1" aria-labelledby="lane1">
    <div class="lane-head"><span class="lane-no">1</span><div><h2 id="lane1">Theo sở thích</h2><span class="kind">LLM agent trích sở thích → lọc kho theo domain</span></div></div>
    <div class="block"><span class="eyebrow">Đầu vào</span><div class="io"><span><b class="num">${u.n}</b> lượt chấm của độc giả #${u.id}</span><span class="note">Danh mục ${nf.format(D.meta.n_books)} sách, mỗi sách có thể loại, tag, tác giả.</span></div></div>
    <div class="block"><span class="eyebrow">Các bước</span><ol class="steps">
      ${step(1, 'Agent trích sở thích', '<span class="tag">LLM</span>', `
        <p class="summary"><b>${esc(p.domain_vi)}.</b> ${esc(p.summary_vi)}</p>
        <div class="wrows">${p.genres.map(g => `<div class="wrow" title="Bằng chứng: ${esc(g.evidence.join('; '))}"><span>${esc(g.name)}</span><span class="track"><i style="width:${pct(g.weight, maxW)}%"></i></span><span class="num">${g.weight.toFixed(2)}</span></div>`).join('')}</div>
        <div class="chips">${p.authors.map(a => `<span class="chip" title="${esc(a.evidence.join('; '))}">${esc(a.name)}</span>`).join('')}${p.themes.map(t => `<span class="chip k">#${esc(t)}</span>`).join('')}</div>
        ${p.avoid.length ? `<div class="chips"><span class="note">Tránh:</span>${p.avoid.map(a => `<span class="chip x" title="${esc(a.evidence.join('; '))}">${esc(a.name)}</span>`).join('')}</div>` : ''}
        <details><summary>Bằng chứng agent trích dẫn</summary><ul>${p.genres.map(g => `<li><b>${esc(g.name)}</b>: ${esc(g.evidence.join('; '))}</li>`).join('')}</ul></details>`)}
      ${step(2, 'Lọc kho theo domain', `<span class="tag">code</span><span class="tag plain">${u.timing.pref} ms</span>`, `
        <p><b class="num">${nf.format(u.pref.match)}</b> / ${nf.format(D.meta.n_books)} sách chưa đọc khớp ít nhất 1 thể loại; <b class="num">${nf.format(u.pref.pool)}</b> cuốn còn điểm dương sau khi trừ thể loại tránh.</p>
        <div class="code">điểm = (Σ trọng số thể loại + 0.3·#chủ đề + 1·tác giả − 1·thể loại tránh) × chất lượng; tối đa 2 cuốn/tác giả</div>`)}
    </ol></div>
    <div class="block"><span class="eyebrow">Đầu ra · top ${u.pref.recs.length}</span><div class="rows">${recs}</div>${moreBtn('pref', u.pref.recs.length)}</div>
  </article>`;
}

/* ---------- lane 2: web search ---------- */
function laneWeb(u) {
  const w = u.web;
  const recs = take(w.books.map((x, i) => [x, i]), 'web').map(([x, i]) => {
    const href = safeUrl(x.source_url);
    const go = `web-${state.user}-${i}`;
    const head = x.b ? `${coverBtn(x.b, 'sm')}` : `<button class="cover-btn" data-go="${go}" aria-label="Mở ${esc(x.title)}">${cover(x.title, x.author, 'sm')}</button>`;
    return `<div class="row">${head}<div class="row-body">
      <button class="title-btn" data-go="${x.b ? 'book-' + x.b : go}">${esc(x.title)}</button>
      <span class="author">${esc(x.author)} · ${x.year}</span>
      <div class="score">${x.b ? '<span class="badge">Có trong kho</span>' : '<span class="badge">Ngoài kho</span>'}<span class="note">nguồn: ${href ? `<a href="${esc(href)}" target="_blank" rel="noopener noreferrer">${esc(x.source_name)}</a>` : esc(x.source_name)} · truy vấn ${x.query_index + 1}</span></div>
      <p class="why">${esc(x.why_vi)}</p></div></div>`;
  }).join('');
  return `<article class="lane l2" aria-labelledby="lane2">
    <div class="lane-head"><span class="lane-no">2</span><div><h2 id="lane2">Từ web</h2><span class="kind">LLM agent gọi tool web_search</span></div></div>
    <div class="block"><span class="eyebrow">Đầu vào</span><div class="io"><span>Lịch sử chấm + gu đọc của độc giả #${u.id}</span><span class="note">Ngày chạy 11/11/2026. Tìm sách xuất bản 2024–2026, phần kho Goodbooks-10k không có.</span>${u.missing ? `<span class="note">Lúc chạy, ${u.missing} dòng lịch sử thiếu tên sách vì dữ liệu crawl không có.</span>` : ''}</div></div>
    <div class="block"><span class="eyebrow">Các bước · ${w.queries.length} lần gọi tool</span><ol class="steps">
      ${w.queries.map((q, i) => step(i + 1, 'web_search', `<span class="tag">tool</span><span class="tag plain">dùng ${q.results_used} kết quả</span>`,
        `<div class="call"><div class="code">"${esc(q.query)}"</div><span class="note">${esc(q.purpose_vi)}</span></div>`)).join('')}
      ${step(w.queries.length + 1, 'Agent chọn và kiểm chứng', '<span class="tag">LLM</span>', `
        <p>Giữ <b>${w.books.length}</b> cuốn có nguồn cụ thể, loại <b>${w.dropped.length}</b> ứng viên.</p>
        <details><summary>Ứng viên bị loại và lý do</summary><ul>${w.dropped.map(d => `<li>${esc(d)}</li>`).join('')}</ul></details>`)}
      ${step(w.queries.length + 2, 'Khớp với kho', '<span class="tag">code</span>', `<p><b>${w.books.filter(x => x.b).length}</b> / ${w.books.length} cuốn có trong Goodbooks-10k (kho chỉ có sách tới 2017).</p>`)}
    </ol></div>
    <div class="block"><span class="eyebrow">Đầu ra · ${w.books.length} cuốn</span><div class="rows">${recs}</div>${moreBtn('web', w.books.length)}</div>
  </article>`;
}

/* ---------- lane 3: classic ML ---------- */
const ALGO = {
  ucf: { name: 'UCF', full: 'Lọc cộng tác theo người dùng', method: 'ucf.py · UserBasedCollaborativeFiltering · sim = cosine lệch trung bình trên sách cùng chấm (≥ 8 cuốn) × min(m, 30)/30 · 25 láng giềng · r̂ = r̄ᵤ + Σ sim·(r − r̄ᵥ) / Σ|sim| · cần ≥ 3 láng giềng đã chấm' },
  icf: { name: 'Item-CF', full: 'Lọc cộng tác theo sách', method: 'ucf.py · ItemBasedCollaborativeFiltering · adjusted cosine giữa các cột sách · r̂ = r̄ᵤ + Σ sim·(r − r̄ᵤ) / Σ|sim| trên 25 sách đã chấm giống nhất · bỏ sách có Σsim < 0.5' },
  mf: { name: 'MF', full: 'Phân rã ma trận', method: '' },
  cbf: { name: 'CBF', full: 'Lọc theo nội dung', method: 'content_filtering.py · TF-IDF trên thể loại + tác giả + tag + mô tả · hồ sơ = trọng tâm có trọng số (r − r̄ᵤ) · xếp theo cosine · giải thích = đặc trưng chung đóng góp nhiều nhất' },
};
ALGO.mf.method = `SVD 50 yếu tố trên ma trận đã trừ trung bình, giữ ${Math.round(D.meta.svd_var * 100)}% phương sai (thay cho MatrixEmbedding dùng SGD trong ucf.py) · điểm = pᵤ · qᵢ · giải thích = yếu tố f có pᵤ[f]·qᵢ[f] lớn nhất`;

function mlBody(a, x) {
  if (a === 'ucf') return `<div class="score"><span class="star">★</span><span class="num">${x.score.toFixed(2)}</span><span class="note">dự đoán · ${x.cnt} láng giềng đã chấm</span></div>
    <div class="why">${x.nb.slice(0, 2).map(n => `Độc giả <b class="num">#${n.u}</b> <span class="num">(sim ${n.sim.toFixed(2)})</span> chấm <span class="star">${stars(n.r)}</span>`).join('<br>')}</div>`;
  if (a === 'icf') return `<div class="score"><span class="star">★</span><span class="num">${x.score.toFixed(2)}</span><span class="note">dự đoán · Σsim ${x.support.toFixed(2)}</span></div>
    <div class="why">${x.via.map(v => `Giống <b>${esc(short(B[v.b].t))}</b> <span class="num">(${v.sim.toFixed(2)})</span> bạn chấm <span class="star">${stars(v.r)}</span>`).join('<br>')}</div>`;
  if (a === 'mf') return `<div class="score"><span class="num">pᵤ·qᵢ = ${x.score.toFixed(3)}</span><span class="note">yếu tố #${x.f} đóng góp ${x.term >= 0 ? '+' : ''}${x.term.toFixed(3)}, lớn nhất</span></div>
    <div class="why">Yếu tố #${x.f} gần với: ${x.reps.map(r => `<b>${esc(short(B[r].t))}</b>`).join(', ')}</div>`;
  return `<div class="score"><span class="num">cos ${x.score.toFixed(3)}</span></div>
    <div class="why">Gần <b>${esc(short(B[x.anchor].t))}</b> bạn chấm cao</div><div class="chips">${x.why.map(w => `<span class="chip">${esc(w)}</span>`).join('')}</div>`;
}

function laneML(u) {
  const a = state.algo, list = u.ml[a];
  const extra = a === 'ucf' ? `<div class="wrows">${u.nb.map(n => `<div class="wrow"><span class="num">#${n.u}</span><span class="track"><i style="width:${pct(n.sim, u.nb[0].sim)}%"></i></span><span class="num">${n.sim.toFixed(2)}</span></div>`).join('')}</div>
    <span class="note">5 láng giềng gần nhất trong ${nf.format(u.n_users)} người có ≥ 8 sách chung.</span>` : '';
  return `<article class="lane l3" aria-labelledby="lane3">
    <div class="lane-head"><span class="lane-no">3</span><div><h2 id="lane3">Thuật toán ML</h2><span class="kind">Thuật toán truyền thống, không dùng LLM</span></div></div>
    <div class="block"><span class="eyebrow">Đầu vào</span><div class="io"><span>Ma trận <b class="num">${nf.format(D.meta.n_users)} × ${nf.format(D.meta.n_books)}</b>, ${nf.format(D.meta.n_ratings)} lượt chấm</span><span class="note">Lịch sử của độc giả #${u.id} là một hàng trong ma trận.</span></div></div>
    <div class="block"><span class="eyebrow">Thuật toán</span>
      <div class="tabs" role="tablist">${Object.entries(ALGO).map(([k, x]) => `<button role="tab" data-algo="${k}" aria-selected="${k === a}" title="${x.full}">${x.name}</button>`).join('')}</div>
      <ol class="steps">${step(1, esc(ALGO[a].full), `<span class="tag">code</span><span class="tag plain">${u.timing[a]} ms${a === 'mf' ? ` + huấn luyện ${D.meta.svd_ms} ms` : ''}</span>`, `<div class="code">${ALGO[a].method}</div>${extra}`)}</ol></div>
    <div class="block"><span class="eyebrow">Đầu ra · top ${list.length}</span><div class="rows">${take(list, 'ml').map(x => row(x.b, mlBody(a, x))).join('')}</div>${moreBtn('ml', list.length)}</div>
  </article>`;
}

/* ---------- home ---------- */
function renderHome() {
  const u = U();
  const sets = { 1: new Set(u.pref.recs.map(x => x.b)), 2: new Set(u.web.books.filter(x => x.b).map(x => x.b)), 3: new Set(Object.values(u.ml).flat().map(x => x.b)) };
  const all = [...new Set([...sets[1], ...sets[2], ...sets[3]])];
  const shared = all.map(b => [b, [1, 2, 3].filter(k => sets[k].has(b))]).filter(([, ks]) => ks.length > 1);
  const maxD = Math.max(...u.dist);
  view.innerHTML = `
  <section><div class="head"><div><span class="eyebrow">Độc giả #${u.id} · ${esc(u.pref.profile.domain_vi)}</span><h1>Gợi ý cho bạn</h1>
    <p>Ba luồng chạy độc lập từ cùng một lịch sử chấm điểm. Mỗi cột hiện đầu vào, các bước đã chạy và sách mà luồng đó trả về.</p></div></div></section>
  <div class="lanes">${lanePref(u)}${laneWeb(u)}${laneML(u)}</div>
  <section class="panel"><span class="eyebrow">Giao nhau giữa các luồng</span>
    ${shared.length ? `<div class="overlap">${shared.map(([b, ks]) => `<div class="ov">${ks.map(k => `<span class="pill p${k}">${k}</span>`).join('')}<button class="title-btn" data-go="book-${b}">${esc(short(B[b].t))}</button></div>`).join('')}</div>`
      : '<p class="muted">Không cuốn nào xuất hiện ở hai luồng trở lên.</p>'}
    <span class="note">Luồng 3 tính gộp top 10 của cả 4 thuật toán (${sets[3].size} cuốn khác nhau). Luồng 2 chỉ tính sách có trong kho.</span></section>
  <section><div class="head"><h2>Dữ liệu đầu vào: lịch sử chấm</h2><span class="muted">12 / ${u.n} cuốn chấm cao nhất</span></div>
    <div class="profile">
      <div class="panel"><div class="stat"><div><span class="num">${u.n}</span><span class="muted">sách đã chấm</span></div><div><span class="num">${u.mean.toFixed(2)}</span><span class="muted">điểm trung bình</span></div></div></div>
      <div class="panel"><div class="dist">${[5, 4, 3, 2, 1].map(k => `<div class="dist-row"><span class="num">${k}★</span><span class="track"><i style="width:${pct(u.dist[k - 1], maxD)}%"></i></span><span class="num">${u.dist[k - 1]}</span></div>`).join('')}</div></div>
    </div>
    <div class="shelf">${u.history.map(h => tile(h.b, `<span class="star">${stars(h.r)}</span>`)).join('')}</div></section>`;
}

/* ---------- search ---------- */
const ALL = Object.keys(B).map(Number);
const GENRES = (() => {
  const c = {};
  ALL.forEach(id => B[id].g.forEach(x => { if (x !== 'Fiction' && x !== 'Audiobook') c[x] = (c[x] || 0) + 1; }));
  return Object.entries(c).sort((a, b) => b[1] - a[1]).slice(0, 24).map(e => e[0]);
})();

function renderSearch() {
  view.innerHTML = `<section>
    <div class="head"><div><span class="eyebrow">Tìm trong ${nf.format(ALL.length)} sách của bản demo</span><h1>Tìm kiếm</h1>
      <p>Khi 3 luồng chưa đủ, độc giả tự tìm ở đây. Mở một cuốn để xem sách tương tự theo nội dung.</p></div></div>
    <div class="controls">
      <input id="q" type="search" placeholder="Tên sách hoặc tác giả" value="${esc(state.q)}" aria-label="Tìm sách">
      <select id="genre" aria-label="Thể loại"><option value="">Mọi thể loại</option>${GENRES.map(g => `<option${g === state.genre ? ' selected' : ''}>${esc(g)}</option>`).join('')}</select>
      <select id="sort" aria-label="Sắp xếp">${[['n', 'Nhiều lượt chấm'], ['r', 'Điểm cao'], ['y', 'Mới xuất bản']].map(([v, l]) => `<option value="${v}"${v === state.sort ? ' selected' : ''}>${l}</option>`).join('')}</select>
    </div>
    <p class="note" id="count"></p>
    <div class="grid" id="grid"></div></section>`;
  drawGrid();
}

function drawGrid() {
  const q = state.q.trim().toLowerCase();
  const key = { n: b => -b.n, r: b => -b.r, y: b => -(b.y ?? -9999) }[state.sort];
  const ids = ALL.filter(id => {
    const b = B[id];
    return (!q || b.t.toLowerCase().includes(q) || b.a.toLowerCase().includes(q)) && (!state.genre || b.g.includes(state.genre));
  }).sort((a, b) => key(B[a]) - key(B[b]));
  document.getElementById('count').textContent = `${nf.format(ids.length)} kết quả${ids.length > 120 ? ', hiện 120 cuốn đầu' : ''}`;
  document.getElementById('grid').innerHTML = ids.length ? ids.slice(0, 120).map(id => tile(id)).join('')
    : `<div class="empty">Không có sách khớp “${esc(state.q)}”${state.genre ? ` trong ${esc(state.genre)}` : ''}. Thử bỏ bớt bộ lọc.</div>`;
}

/* ---------- drawers ---------- */
let opener = null;
function showDrawer(html) {
  if (drawer.hidden) opener = document.activeElement;
  drawer.innerHTML = `<button class="backdrop" data-close aria-label="Đóng"></button><div class="sheet" role="dialog" aria-modal="true" aria-labelledby="bk-title"><button class="btn close" data-close>Đóng</button>${html}</div>`;
  drawer.hidden = false;
  drawer.querySelector('.sheet').scrollTop = 0;
  drawer.querySelector('.close').focus();
  document.body.style.overflow = 'hidden';
}

function openBook(id) {
  const b = B[id], sims = D.similar[id] || [];
  const maxS = Math.max(...sims.map(s => s.s), 0.001);
  showDrawer(`<div class="book-head">${bookCover(id, 'lg')}<div style="display:grid;gap:10px;align-content:start;min-width:0">
      <h2 id="bk-title">${esc(short(b.t))}</h2>${series(b.t) ? `<span class="muted">${esc(series(b.t))}</span>` : ''}
      <dl class="meta"><dt>Tác giả</dt><dd>${esc(b.a)}</dd>${b.y ? `<dt>Xuất bản</dt><dd class="num">${b.y}</dd>` : ''}${b.p ? `<dt>Số trang</dt><dd class="num">${b.p}</dd>` : ''}
        <dt>Điểm TB</dt><dd><span class="star">★</span> <span class="num">${b.r.toFixed(2)}</span></dd><dt>Lượt chấm</dt><dd class="num">${nf.format(b.n)}</dd></dl></div></div>
    <div class="chips">${b.g.map(g => `<span class="chip">${esc(g)}</span>`).join('')}${b.tags.map(t => `<span class="chip k">#${esc(t)}</span>`).join('')}</div>
    ${b.d ? `<p class="desc">${esc(b.d)}${b.d.length >= 700 ? '…' : ''}</p>` : ''}
    <section><h3>Sách tương tự theo nội dung</h3>
      ${LOCAL.has(+id) ? `<span class="note">Bản demo chỉ chứa ${nf.format(ALL.length)} sách; danh sách này xếp hạng trong số đó.</span>` : ''}
      <div class="sims">${sims.map(s => `<div class="sim">${coverBtn(s.b, 'sm')}<div class="row-body">${titleBtn(s.b)}<span class="author">${esc(B[s.b].a)}</span>
        <div class="score"><span class="num">cos ${s.s.toFixed(3)}</span><span class="track" style="flex:1;max-width:120px"><i style="width:${pct(s.s, maxS)}%"></i></span></div>
        <div class="chips">${s.why.map(w => `<span class="chip">${esc(w)}</span>`).join('')}</div></div></div>`).join('')}</div></section>`);
}

function openWeb(ui, i) {
  const x = D.users[ui].web.books[i], q = D.users[ui].web.queries[x.query_index], href = safeUrl(x.source_url);
  showDrawer(`<div class="book-head">${cover(x.title, x.author, 'lg')}<div style="display:grid;gap:10px;align-content:start;min-width:0">
      <h2 id="bk-title">${esc(x.title)}</h2>
      <dl class="meta"><dt>Tác giả</dt><dd>${esc(x.author)}</dd><dt>Năm</dt><dd class="num">${x.year}</dd>
        <dt>Nguồn</dt><dd>${href ? `<a href="${esc(href)}" target="_blank" rel="noopener noreferrer">${esc(x.source_name)}</a>` : esc(x.source_name)}</dd>
        <dt>Trong kho</dt><dd>Không. Goodbooks-10k chỉ có sách tới 2017 nên chưa có điểm cộng đồng.</dd></dl></div></div>
    <section><span class="eyebrow">Trích từ kết quả tìm kiếm</span><blockquote>${esc(x.snippet)}</blockquote>
      ${href ? `<span class="note num" style="overflow-wrap:anywhere">${esc(href)}</span>` : ''}
      ${x.fix_note ? `<p class="note"><b>Đã sửa sau kiểm chứng nguồn:</b> ${esc(x.fix_note)}</p>` : ''}</section>
    <section><span class="eyebrow">Vì sao agent chọn</span><p>${esc(x.why_vi)}</p>
      ${x.related_history.length ? `<p class="note">Liên quan tới sách độc giả đã chấm: ${x.related_history.map(t => `<b>${esc(t)}</b>`).join('; ')}</p>` : ''}</section>
    ${q ? `<section><span class="eyebrow">Tool call</span><div class="code">web_search("${esc(q.query)}")</div><span class="note">${esc(q.purpose_vi)}</span></section>` : ''}`);
}

function closeDrawer() {
  if (drawer.hidden) return;
  drawer.hidden = true; drawer.innerHTML = '';
  document.body.style.overflow = '';
  if (opener && opener.isConnected) opener.focus();
  opener = null;
}
// Replace the drawer's history entry so Back does not reopen a closed book.
const leaveDrawer = () => { location.replace('#' + state.page); route(); };

/* ---------- routing ---------- */
function renderTop() {
  const u = state.user === null ? null : U();
  document.getElementById('nav').hidden = !u;
  document.getElementById('who').innerHTML = u ? `<span class="num note">#${u.id}</span><button class="btn" data-logout>Đăng xuất</button>` : '';
  document.querySelectorAll('[data-nav]').forEach(a => a.setAttribute('aria-current', a.dataset.nav === state.page ? 'page' : 'false'));
}

function renderPage() {
  ({ login: renderLogin, home: renderHome, search: renderSearch })[state.page]();
  renderTop();
}

let last = null;
function route() {
  const h = location.hash.slice(1);
  let m;
  const book = (m = h.match(/^book-(\d+)$/)) && B[m[1]] ? m[1] : null;
  const web = (m = h.match(/^web-(\d+)-(\d+)$/)) && D.users[m[1]]?.web.books[m[2]] ? [+m[1], +m[2]] : null;
  if (book || web) {
    if (!last) { last = state.page; renderPage(); }
    return book ? openBook(book) : openWeb(...web);
  }
  closeDrawer();
  let page = h === 'search' ? 'search' : 'home';
  if (state.user === null) page = 'login';
  if (page !== last) { state.page = last = page; renderPage(); window.scrollTo(0, 0); }
}

document.addEventListener('click', e => {
  const t = e.target.closest('[data-go],[data-login],[data-logout],[data-algo],[data-more],[data-close]');
  if (!t) return;
  const d = t.dataset;
  if (d.go) {
    // Inside an open drawer, replace the entry so the drawer owns at most one history entry.
    if (drawer.hidden) location.hash = d.go; else { location.replace('#' + d.go); route(); }
  }
  else if (d.login !== undefined) {
    state.user = +d.login; state.more = {}; state.algo = 'ucf';
    // A logged-out visit to #home shows login without changing the hash, so no hashchange would fire.
    if (location.hash === '#home') route(); else location.hash = 'home';
  }
  else if (d.logout !== undefined) { state.user = null; location.hash = 'login'; route(); }
  else if (d.algo) { state.algo = d.algo; renderHome(); view.querySelector(`[data-algo="${d.algo}"]`)?.focus(); }
  else if (d.more) { state.more[d.more] = !state.more[d.more]; renderHome(); view.querySelector(`[data-more="${d.more}"]`)?.focus(); }
  else leaveDrawer();
});
document.addEventListener('input', e => { if (e.target.id === 'q') { state.q = e.target.value; drawGrid(); } });
document.addEventListener('change', e => {
  if (e.target.id === 'genre') { state.genre = e.target.value; drawGrid(); }
  if (e.target.id === 'sort') { state.sort = e.target.value; drawGrid(); }
});
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !drawer.hidden) leaveDrawer(); });
window.addEventListener('hashchange', route);
route();
