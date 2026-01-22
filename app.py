import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import yt_dlp
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="IITM Search Engine", page_icon="🎓", layout="wide")

# --- AI SETUP ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- BACKEND LOGIC ---
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
    """Downloads and Indexes a specific course using FAISS"""
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
        # Convert titles to vectors
        embeddings = model.encode(titles)
        # FAISS requires float32 format
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS Index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return len(titles), index, metadata, titles
        
    return 0, None, [], []

# --- FRONTEND UI ---
st.sidebar.title("🎓 IITM Search")
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
    with st.spinner(f"Indexing {selected_course}..."):
        count, index, meta, titles = index_course(url)
        if count > 0:
            st.session_state.active_index = index
            st.session_state.active_meta = meta
            st.session_state.course_name = selected_course
            st.success(f"✅ Ready! Indexed {count} lectures.")
        else:
            st.error("No videos found in this playlist.")

st.sidebar.markdown("---")
st.sidebar.info("Tip: Select a course, click Load, then search.")

# Main Search Area
st.title("Search Engine")

if 'active_index' in st.session_state:
    st.caption(f"Searching inside: **{st.session_state.course_name}**")
    query = st.text_input("Enter Topic:", placeholder="e.g. Gradient Descent, Hypothesis Testing...")
    
    if query:
        # Search FAISS
        query_vec = model.encode([query])
        query_vec = np.array(query_vec).astype('float32')
        
        # Get top 10 matches
        k = 10 
        distances, indices = st.session_state.active_index.search(query_vec, k)
        
        st.markdown("### Results")
        found = False
        for i in range(k):
            idx = indices[0][i]
            if idx < len(st.session_state.active_meta): # Safety check
                found = True
                meta = st.session_state.active_meta[idx]
                url = f"https://www.youtube.com/watch?v={meta['id']}"
                
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #ff4b4b;">
                    <a href="{url}" target="_blank" style="text-decoration: none; color: #000; font-weight: bold; font-size: 18px;">
                        🎥 {meta['title']}
                    </a>
                </div>
                """, unsafe_allow_html=True)
        
        if not found:
            st.warning("No matches found.")
else:
    st.info("👈 Please select a course and click 'Load Course Videos' to start.")

