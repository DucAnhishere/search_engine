import streamlit as st
import os
from text_processing import chunking
from vector_search import create_embedding, store_data, search_resumes
from scoring import compute_weighted_score
import mimetypes


# Streamlit UI Configuration
st.set_page_config(page_title="AI Resume Search", page_icon="🔍", layout="wide")

# Custom CSS 
st.markdown("""
    <style>
        .big-text { font-size:24px !important; font-weight: bold; }
        .medium-text { font-size:20px !important; }
        .score-text { font-size:22px !important; font-weight: bold; color: #FF4B4B; }
        .chunk-text { font-size:22px !important; font-style: italic; color: #FFFFFF; background-color: #333333; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🔍 AI Resume Search")
st.markdown("<p class='big-text'>Search through resumes using AI-powered embeddings!</p>", unsafe_allow_html=True)

# Processing parameters
model_name = "sentence-transformers/msmarco-MiniLM-L12-v3"
# model_name = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
collection_name = "CV1_collection"
chunk_name = "RecursiveCharacterTextSplitter"
chunk_size = 500
chunk_overlap = 100
dimension = 384
alpha = 0.9

# Sidebar: Search Query
st.sidebar.header("🔍 Search Resumes")
query = st.sidebar.text_input("Enter your search query:")

# Adjustable k parameter
st.sidebar.header("⚙️ Search Settings")
k = st.sidebar.slider("Number of top chunks", min_value=1, max_value=50, value=20, step=1)
st.sidebar.write(f"Selected top results: {k}")

# Perform Search
if query:
    with st.spinner("🔍 Searching..."):
        # Generate query embedding
        _, embeddings2 = create_embedding(model_name, [], [query])

        # Perform search
        top_k_similarity, top_k_chunks, resume_ids, resume_paths = search_resumes(embeddings2, collection_name, k)

        # Compute final weighted scores
        final_scores, matched_chunks_dict = compute_weighted_score(resume_ids, top_k_chunks, top_k_similarity, k, alpha)

        # Sort results by score
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    # Display Results
    st.markdown("<p class='big-text'>🎯 Search Results</p>", unsafe_allow_html=True)
    if sorted_results:
        for i, (resume_id, score) in enumerate(sorted_results):
            # Get the corresponding resume path
            resume_index = resume_ids.index(resume_id)
            matched_chunk = top_k_chunks[resume_index]
            resume_path = resume_paths[resume_index]

            # Display information with Bigger Text
            st.markdown(f"<p class='big-text'>📄 Resume {i+1}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='score-text'>💡 Similarity: {score:.4f}</p>", unsafe_allow_html=True)

            # Title for matched chunks
            st.markdown("<p class='big-text'>🔹 Matched Chunks:</p>", unsafe_allow_html=True)
            
            with st.expander(f"🔹 {len(matched_chunks_dict[resume_id])} Matched Chunks (Click to expand)"):
                for chunk, similarity in matched_chunks_dict[resume_id]:
                    st.markdown(f"<div class='chunk-text'>📌 <b>Chunk:</b> {chunk} </div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chunk-text'><b>Similarity:</b> {similarity:.4f}</div>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)


            # Add a button to download the CV file
            with open(resume_path, "rb") as file:
                file_bytes = file.read()
                mime_type, _ = mimetypes.guess_type(resume_path)
                st.download_button(
                    label=f"📥 Download CV {i+1}",
                    data=file_bytes,
                    file_name=os.path.basename(resume_path),
                    mime=mime_type or "application/octet-stream"
                )

            st.divider()
    else:
        st.warning("No matching CVs found.")

# Sidebar: Folder Picker
st.sidebar.header("📂 Add new CVs to the database")
directory_path = st.sidebar.text_input("Enter the folder path:")

# Display the selected folder path
if directory_path:
    st.sidebar.success(f"📂 Selected: {directory_path}")
    
# Button: Process CVs
if st.sidebar.button("📜 Process CVs"):
    if not directory_path:
        st.error("⚠️ Please select a folder first.")
    elif not os.path.exists(directory_path):
        st.error("⚠️ The folder does not exist. Please select a valid folder.")
    else:
        with st.spinner("⏳ Processing CVs..."):

            # Split text into chunks
            all_chunks, resume_paths, resume_ids = chunking(directory_path, chunk_size, chunk_overlap)

            # Generate embeddings
            embeddings1, _ = create_embedding(model_name, all_chunks, ["placeholder"])
            embeddings1_list = embeddings1.tolist()

            # Store embeddings in Milvus 
            store_data(
                collection_name=collection_name,
                embeddings=embeddings1,
                embeddings_list=embeddings1_list,
                all_chunks=all_chunks,
                resume_paths=resume_paths,
                resume_ids=resume_ids
            )

            st.success("✅ All CVs have been processed")
            





