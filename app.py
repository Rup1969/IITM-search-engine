import streamlit as st
from sentence_transformers import SentenceTransformer, util
import yt_dlp
import torch

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
    """Downloads and Indexes a specific course"""
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
        # Create Embeddings
        embeddings = model.encode(titles, convert_to_tensor=True)
        return len(titles), embeddings, metadata
        
    return 0, None, []

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
        count, embeddings, meta = index_course(url)
        if count > 0:
            st.session_state.active_embeddings = embeddings
            st.session_state.active_meta = meta
            st.session_state.course_name = selected_course
            st.success(f"✅ Ready! Indexed {count} lectures.")
        else:
            st.error("No videos found in this playlist.")

st.sidebar.markdown("---")
st.sidebar.info("Tip: Select a course, click Load, then search.")

# Main Search Area
st.title("Search Engine")

if 'active_embeddings' in st.session_state:
    st.caption(f"Searching inside: **{st.session_state.course_name}**")
    query = st.text_input("Enter Topic:", placeholder="e.g. Gradient Descent, Hypothesis Testing...")
    
    if query:
        # 1. Convert Query to Vector
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # 2. Pure Math Search (Cosine Similarity)
        # We calculate the score against ALL video titles instantly
        cos_scores = util.cos_sim(query_embedding, st.session_state.active_embeddings)[0]
        
        # 3. Find Top 10 matches
        top_results = torch.topk(cos_scores, k=min(10, len(st.session_state.active_meta)))
        
        st.markdown("### Results")
        
        if top_results.values[0] < 0.2: # If the best match is very low score
             st.warning("No close matches found. Try a different keyword.")
        else:
            for score, idx in zip(top_results.values, top_results.indices):
                meta = st.session_state.active_meta[idx]
                url = f"https://www.youtube.com/watch?v={meta['id']}"
                
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #28a745;">
                    <a href="{url}" target="_blank" style="text-decoration: none; color: #000; font-weight: bold; font-size: 18px;">
                        🎥 {meta['title']}
                    </a>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("👈 Please select a course and click 'Load Course Videos' to start.")

