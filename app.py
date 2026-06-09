import streamlit as st
from fastembed import TextEmbedding
import yt_dlp
import numpy as np
import json, os, time

from streamlit_gtag import st_gtag

# Replace with your actual ID!
st_gtag(
    gtag_id="G-KZGDGQ975Y", 
    config={
        "send_page_view": True 
    }
)

# ... the rest of your app code starts here ...
st.set_page_config(page_title="IITM Neural Search", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.result-card {
    background: #f0f8ff;
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 10px;
    border-left: 5px solid #007bff;
}
.result-card a {
    text-decoration: none;
    color: #000;
    font-weight: bold;
    font-size: 16px;
}
.result-card a:hover { color: #007bff; }
.score { font-size: 12px; color: #666; margin-top: 4px; }
.course-tag { font-size: 12px; color: #007bff; margin-top: 2px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

CACHE_DIR    = os.path.dirname(__file__)
VIDEOS_CACHE = os.path.join(CACHE_DIR, "all_videos.json")
EMBED_CACHE  = os.path.join(CACHE_DIR, "all_embeddings.npy")

CHANNEL_URL = "https://www.youtube.com/@IITMadrasBSDegreeProgramme/courses"

# Extra playlists not listed under /courses but part of the programme
EXTRA_PLAYLISTS = [
    {"url": "https://www.youtube.com/playlist?list=PLZ2ps__7DhBa9hqi20allqocTSUUt3nWX",
     "name": "Introduction to Deep Learning & AI"},
]


# ── Load embedding model (local, no API key) ──────────────
@st.cache_resource
def load_model():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

model = load_model()

# ── Fetch course list from channel ────────────────────────
@st.cache_data(ttl=3600)
def fetch_course_catalog():
    ydl_opts = {"quiet": True, "extract_flat": True, "ignoreerrors": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(CHANNEL_URL, download=False)
    catalog = {}
    junk = ["shorts", "testimonial", "webinar", "event", "hackathon",
            "promo", "teaser", "live session", "live class"]
    for entry in (info.get("entries") or []):
        title = (entry.get("title") or "").strip()
        url   = entry.get("url") or entry.get("webpage_url", "")
        if title and url and not any(j in title.lower() for j in junk):
            catalog[title] = url
    return dict(sorted(catalog.items()))

# ── Fetch videos from one playlist URL ───────────────────
def fetch_playlist_videos(url: str, course_name: str) -> list:
    ydl_opts = {"quiet": True, "extract_flat": True, "ignoreerrors": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    videos = []
    for v in (info.get("entries") or []):
        if not v:
            continue
        title  = (v.get("title") or "").strip()
        vid_id = v.get("id", "")
        if not title or not vid_id:
            continue
        if title in ("[Private video]", "[Deleted video]"):
            continue
        videos.append({
            "title":  title,
            "id":     vid_id,
            "url":    f"https://www.youtube.com/watch?v={vid_id}",
            "course": course_name,
        })
    return videos

# ── Build ALL courses index ────────────────────────────────
def build_all_index(catalog: dict, progress_bar=None, status=None):
    all_videos = []
    total = len(catalog)
    for i, (name, url) in enumerate(catalog.items()):
        if status:
            status.caption(f"Fetching: {name} ({i+1}/{total})")
        if progress_bar:
            progress_bar.progress((i + 1) / total * 0.7)
        try:
            all_videos.extend(fetch_playlist_videos(url, name))
        except Exception:
            pass
        time.sleep(0.2)

    # Add extra playlists not in /courses
    for extra in EXTRA_PLAYLISTS:
        if status:
            status.caption(f"Fetching: {extra['name']}")
        try:
            all_videos.extend(fetch_playlist_videos(extra["url"], extra["name"]))
        except Exception:
            pass

    if status:
        status.caption("Building embeddings for all videos…")
    titles     = [v["title"] for v in all_videos]
    embeddings = np.array(list(model.embed(titles)))
    if progress_bar:
        progress_bar.progress(1.0)

    with open(VIDEOS_CACHE, "w", encoding="utf-8") as f:
        json.dump(all_videos, f, indent=2, ensure_ascii=False)
    np.save(EMBED_CACHE, embeddings)
    return all_videos, embeddings

def load_all_cache():
    if os.path.exists(VIDEOS_CACHE) and os.path.exists(EMBED_CACHE):
        with open(VIDEOS_CACHE, encoding="utf-8") as f:
            videos = json.load(f)
        return videos, np.load(EMBED_CACHE)
    return None, None

# ── Cosine similarity search ──────────────────────────────
def do_search(query, videos, embeddings, top_k=6, threshold=0.35):
    query_vec = np.array(list(model.embed([query]))[0])
    scores    = np.dot(embeddings, query_vec)
    top_idx   = np.argsort(scores)[::-1]
    results   = []
    for i in top_idx:
        if scores[i] >= threshold:
            results.append({**videos[i], "score": float(scores[i])})
        if len(results) >= top_k:
            break
    return results

# ─────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────
st.title("🎓 IITM BS — Neural Video Search")
st.markdown("Search lecture videos · local AI · no API keys needed")
st.divider()

# Load catalog
if "catalog" not in st.session_state:
    with st.spinner("Loading course list from YouTube…"):
        st.session_state.catalog = fetch_course_catalog()

catalog = st.session_state.catalog

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Options")

    # ALL COURSES + individual courses in one dropdown
    extra_names    = [e["name"] for e in EXTRA_PLAYLISTS]
    course_options = ["ALL COURSES"] + list(catalog.keys()) + extra_names
    selected = st.selectbox("Select Course", course_options)

    top_k     = st.selectbox("Max results", [3, 4, 5, 6, 7, 8, 9, 10], index=3)
    threshold = st.selectbox("Min relevance", [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50], index=3)

    st.divider()
    refresh_btn = st.button("🔄 Refresh from YouTube", use_container_width=True,
                             help="Only needed when new lectures are added to YouTube")

    if refresh_btn:
        for f in [VIDEOS_CACHE, EMBED_CACHE]:
            if os.path.exists(f):
                os.remove(f)
        for k in ["videos", "embeddings", "loaded_scope"]:
            st.session_state.pop(k, None)
        st.rerun()

    if "loaded_scope" in st.session_state:
        st.divider()
        st.caption(f"✅ **{st.session_state['loaded_scope']}**")
        st.caption(f"{len(st.session_state.get('videos', []))} videos ready")

# ── Auto-load on startup ──────────────────────────────────
# Runs automatically when app opens — no button click needed.
# If cache exists on disk → loads instantly.
# If no cache yet → fetches from YouTube (first time only).

def load_scope(scope: str):
    """Load videos for the selected scope into session state."""
    if scope == "ALL COURSES":
        videos, embeddings = load_all_cache()
        if videos is not None:
            st.session_state.update({"videos": videos, "embeddings": embeddings,
                                     "loaded_scope": "ALL COURSES"})
        else:
            st.info("⏳ First time setup: fetching all courses from YouTube (~5–10 min). Cached forever after.")
            bar    = st.progress(0)
            status = st.empty()
            videos, embeddings = build_all_index(catalog, bar, status)
            bar.empty(); status.empty()
            st.session_state.update({"videos": videos, "embeddings": embeddings,
                                     "loaded_scope": "ALL COURSES"})
            st.success(f"✅ {len(videos)} videos indexed and cached!")
    else:
        # Check catalog first, then extra playlists
        url = catalog.get(scope, "")
        if not url:
            extra = next((e for e in EXTRA_PLAYLISTS if e["name"] == scope), None)
            if extra:
                url = extra["url"]
        if not url:
            return
        with st.spinner(f"Loading {scope}…"):
            videos = fetch_playlist_videos(url, scope)
        if videos:
            titles     = [v["title"] for v in videos]
            embeddings = np.array(list(model.embed(titles)))
            st.session_state.update({"videos": videos, "embeddings": embeddings,
                                     "loaded_scope": scope})

# Auto-load: trigger when scope changes or nothing is loaded yet
if "loaded_scope" not in st.session_state or st.session_state["loaded_scope"] != selected:
    load_scope(selected)

# ── Search ────────────────────────────────────────────────
if "videos" in st.session_state:
    videos     = st.session_state["videos"]
    embeddings = st.session_state["embeddings"]
    scope      = st.session_state["loaded_scope"]

    st.markdown(f"**Scope:** {scope} &nbsp;·&nbsp; {len(videos)} videos")

    query = st.text_input("", placeholder="e.g.  gradient descent  ·  backpropagation  ·  SQL joins")

    SUGGESTIONS = ["gradient descent", "backpropagation", "PCA", "hypothesis testing",
                   "SQL joins", "neural network", "decision tree", "recursion",
                   "eigenvalues", "Flask API", "LSTM", "overfitting"]
    for row in range(0, 12, 6):
        cols = st.columns(6)
        for i, sug in enumerate(SUGGESTIONS[row:row+6]):
            if cols[i].button(sug, key=f"s{row+i}", use_container_width=True):
                query = sug

    st.divider()

    if query:
        results = do_search(query, videos, embeddings, top_k, threshold)
        if not results:
            st.warning("No matches found. Try lowering Min relevance or different keywords.")
        else:
            st.markdown(f"### Results for *\"{query}\"*")
            for r in results:
                course_tag = f'<div class="course-tag">📚 {r["course"]}</div>' \
                             if scope == "ALL COURSES" else ""
                st.markdown(f"""
<div class="result-card">
  <a href="{r['url']}" target="_blank">🎥 {r['title']}</a>
  {course_tag}
  <div class="score">Relevance: {r['score']:.2f}</div>
</div>""", unsafe_allow_html=True)
else:
    st.info("👈 Select a course from the sidebar to begin.")
    

