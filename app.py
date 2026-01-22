import sys
import os

# --- SQLITE FIX FOR STREAMLIT CLOUD ---
# We wrap this in a try-except block so it doesn't crash the app immediately.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If this fails, we just continue. 
    # ChromaDB might complain later about an old SQLite version, 
    # but at least the app will load.
    pass
# ---------------------------------------

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import yt_dlp

# ... rest of your code ...

import streamlit as st
import chromadb
# ... rest of your imports
from sentence_transformers import SentenceTransformer
import yt_dlp
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="IITM Search Engine", page_icon="🎓", layout="wide")

# --- AI & CACHING SETUP ---
# We use @st.cache_resource so the AI model loads ONLY once (fast).
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

model = load_model()
client = get_chroma_client()

# --- BACKEND LOGIC ---
@st.cache_data(ttl=3600) # Cache the catalog for 1 hour
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

    # Filter junk
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
    # Create a unique collection for the current session
    collection_name = "active_course"
    try: client.delete_collection(collection_name)
    except: pass
    collection = client.create_collection(collection_name)
    
    ydl_opts = {'quiet': True, 'extract_flat': True, 'ignoreerrors': True}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        videos = info.get('entries', [])

    documents = []
    metadatas = []
    ids = []
    
    for vid in videos:
        if vid and vid.get('title'):
            title = vid['title']
            vid_id = vid['id']
            # Skip private/deleted
            if title not in ["[Private video]", "[Deleted video]"]:
                documents.append(title)
                metadatas.append({"video_id": vid_id, "title": title})
                ids.append(vid_id)

    if documents:
        embeddings = model.encode(documents).tolist()
        collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
        return len(documents), collection
    return 0, None

# --- FRONTEND UI (Sidebar) ---
st.sidebar.title("🎓 IITM Search")
st.sidebar.markdown("---")

# State Management (Remembering the loaded course)
if 'catalog' not in st.session_state:
    with st.spinner("Connecting to IIT Madras Channel..."):
        st.session_state.catalog = fetch_course_catalog()
    st.sidebar.success(f"Loaded {len(st.session_state.catalog)} courses")

# Course Selector
selected_course = st.sidebar.selectbox("1. Select Course:", list(st.session_state.catalog.keys()))

# Load Button
if st.sidebar.button("2. Load Course Videos"):
    url = st.session_state.catalog[selected_course]
    with st.spinner(f"Indexing {selected_course}..."):
        count, collection = index_course(url)
        if collection:
            st.session_state.active_collection = collection
            st.session_state.course_name = selected_course
            st.success(f"✅ Ready! Indexed {count} lectures from {selected_course}")
        else:
            st.error("No videos found in this playlist.")

st.sidebar.markdown("---")
st.sidebar.info("Tip: Select a course, click Load, then search topics on the right.")

# --- FRONTEND UI (Main Area) ---
st.title("Search Engine")

if 'active_collection' in st.session_state:
    st.caption(f"Searching inside: **{st.session_state.course_name}**")
    
    query = st.text_input("Enter Topic:", placeholder="e.g. Gradient Descent, Hypothesis Testing...")
    
    if query:
        # Search
        query_vec = model.encode(query).tolist()
        results = st.session_state.active_collection.query(query_embeddings=[query_vec], n_results=10)
        
        st.markdown("### Results")
        if results['documents']:
            for i in range(len(results['documents'][0])):
                title = results['documents'][0][i]
                vid_id = results['metadatas'][0][i]['video_id']
                url = f"https://www.youtube.com/watch?v={vid_id}"
                
                # Visual Card
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #ff4b4b;">
                    <a href="{url}" target="_blank" style="text-decoration: none; color: #000; font-weight: bold; font-size: 18px;">
                        🎥 {title}
                    </a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No matches found.")
else:
    st.info("👈 Please select a course and click 'Load Course Videos' in the sidebar to start.")

