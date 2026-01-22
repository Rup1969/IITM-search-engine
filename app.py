import streamlit as st
from fastembed import TextEmbedding
import yt_dlp
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="IITM Neural Search", page_icon="⚡", layout="wide")

# --- 1. LOAD LIGHTWEIGHT AI ---
@st.cache_resource
def load_model():
    # This loads a quantized, lightweight version of the same neural network
    # It is fast and uses very little RAM.
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading AI model: {e}")
    st.stop()

# --- 2. BACKEND LOGIC ---
@st.cache_data(ttl=3600)
def fetch_course_catalog():
    """Scrapes the live YouTube playlists tab"""
    ydl_opts = {
        'quiet': True, 
        'extract_flat': True, 
        'playlistend': 500,
        'ignoreerrors': True
    }
    channel_url = "https://www.youtube.com/@IITMadrasBSDegreeProgramme/playlists"
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(channel_url, download=False)
            raw_entries = info.get('entries', [])
        except Exception:
            return {}

    clean_catalog = {}
    junk_terms = ["shorts", "testimonial", "webinar", "event", "hackathon", "promo", "teaser", "live session"]
    
    for entry in raw_entries:
        title = entry.get('title', 'Unknown')
        url = entry.get('url')
        if title and url and not any(term in title.lower() for term in junk_terms):
            clean_catalog[title] = url
            
    return dict(sorted(clean_catalog.items()))

def index_course(playlist_url):
    """Downloads titles and creates AI Embeddings"""
    ydl_opts = {'quiet': True, 'extract_flat': True, 'ignoreerrors': True}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        videos = info.get('entries', [])

    titles = []
    metadata = []
    
    for vid in videos:
        if vid and vid.get('title'):
            title = vid['title']
            vid_id = vid['id']
            if title not in ["[Private video]", "[Deleted video]"]:
                titles.append(title)
                metadata.append({"id": vid_id, "title": title})

    if titles:
        # Generate Embeddings using FastEmbed
        # It returns a generator, so we convert to list -> numpy array
        embeddings = list(model.embed(titles))
        return len(titles), np.array(embeddings), metadata
        
    return 0, None, []

# --- 3. FRONTEND UI ---
st.sidebar.title("⚡ IITM Fast-Search")
st.sidebar.markdown("---")

# Load Catalog
if 'catalog' not in st.session_state:
    with st.spinner("Connecting to IIT Madras Channel..."):
        st.session_state.catalog = fetch_course_catalog()
    st.sidebar.success(f"Loaded {len(st.session_state.catalog)} courses")

# Select Course
selected_course = st.sidebar.selectbox("1. Select Course:", list(st.session_state.catalog.keys()))

# Load Button
if st.sidebar.button("2. Load Course Videos"):
    url = st.session_state.catalog[selected_course]
    with st.spinner(f"Reading {selected_course}..."):
        count, embeddings, meta = index_course(url)
        if count > 0:
            st.session_state.active_embeddings = embeddings
            st.session_state.active_meta = meta
            st.session_state.course_name = selected_course
            st.success(f"✅ Ready! Indexed {count} lectures.")
        else:
            st.error("No videos found in this playlist.")

st.sidebar.markdown("---")

# Main Search Area
st.title("Neural Search Engine (FastEmbed)")

if 'active_embeddings' in st.session_state:
    st.caption(f"Searching inside: **{st.session_state.course_name}**")
    query = st.text_input("Enter Topic:", placeholder="e.g. Gradient Descent, Hypothesis Testing...")
    
    if query:
        # 1. Embed Query
        query_vec = list(model.embed([query]))[0]
        
        # 2. Cosine Similarity (Manual Numpy Calculation)
        # (dot product of query and all video vectors)
        scores = np.dot(st.session_state.active_embeddings, query_vec)
        
        # 3. Get Top 10 Indices
        top_indices = np.argsort(scores)[-10:][::-1]
        
        st.markdown("### Results")
        
        found = False
        for idx in top_indices:
            score = scores[idx]
            if score > 0.4: # Threshold for "good match"
                found = True
                meta = st.session_state.active_meta[idx]
                url = f"https://www.youtube.com/watch?v={meta['id']}"
                
                st.markdown(f"""
                <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #007bff;">
                    <a href="{url}" target="_blank" style="text-decoration: none; color: #000; font-weight: bold; font-size: 18px;">
                        🎥 {meta['title']}
                    </a>
                    <div style="font-size: 12px; color: #666; margin-top: 5px;">Relevance Score: {score:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        if not found:
            st.warning("No close matches found. Try a different term.")

else:
    st.info("👈 Please select a course and click 'Load Course Videos' to start.")
