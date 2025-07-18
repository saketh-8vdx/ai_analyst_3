
# app.py  — PDF Document Analyzer with parallel processing & folder upload
import streamlit as st
import concurrent.futures
import zipfile
import tempfile
import os
import time
from typing import List, Dict, Any, Tuple
import requests
import json

from ai_analyst import read_pdf, create_chunks, store_embeddings_excel, generate_description,extract_tables_from_reports
from processing_files import convert_pdf_to_text_new, get_chunked_documents_pipeline, get_reducto_chunks
from langchain.callbacks.base import BaseCallbackHandler
from crewai import LLM, Crew, Process


st.set_page_config(
    page_title="PDF Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .query-section {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .response-section {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .file-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .streaming-response {
        background-color: #f0f8ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    .message-log {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .message-info { color: #0066cc; font-weight: bold; }
    .message-success { color: #28a745; font-weight: bold; }
    .message-warning { color: #ffc107; font-weight: bold; }
    .message-error { color: #dc3545; font-weight: bold; }
    .message-tool { color: #6f42c1; font-weight: bold; }
    .message-agent { color: #fd7e14; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)



for key, default in {
    "pdf_dicts": [],
    "processed_files": set(),
    "use_fastapi": False,
    "message_log": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to capture and render LangChain / CrewAI events"""

    def __init__(self, message_container):
        self.message_container = message_container
        self.messages = []

    # helper to add a message into both local list + Streamlit UI
    def add_message(self, kind: str, content: str):
        ts = time.strftime("%H:%M:%S")
        msg = {"timestamp": ts, "type": kind, "content": content}
        self.messages.append(msg)
        st.session_state.message_log.append(msg)
        with self.message_container:
            st.markdown(
                f"""
                <div class='message-log'>
                    <div class='message-{kind.lower()}'>[{ts}] {kind.upper()}:</div>
                    <div>{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # LLM callbacks
    def on_llm_start(self, *_):
        self.add_message("info", "🤖 LLM processing started…")

    def on_llm_new_token(self, *_):
        pass  # avoid token spam

    def on_llm_end(self, *_):
        self.add_message("success", "✅ LLM processing completed")

    def on_llm_error(self, error, *_):
        self.add_message("error", f"❌ LLM Error: {error}")

    # Chain callbacks
    def on_chain_start(self, serialized, *_):
        self.add_message("info", f"🔗 Starting chain: {serialized.get('name', 'Unknown')}")

    def on_chain_end(self, *_):
        self.add_message("success", "✅ Chain completed")

    def on_chain_error(self, error, *_):
        self.add_message("error", f"❌ Chain Error: {error}")

    # Tool callbacks
    def on_tool_start(self, serialized, input_str, *_):
        self.add_message("tool", f"🔧 Using tool: {serialized.get('name', 'Unknown')}")
        self.add_message("info", f"Input: {input_str[:100]}…")

    def on_tool_end(self, output, *_):
        self.add_message("success", "✅ Tool completed")
        if output:
            self.add_message("info", f"Output: {str(output)[:200]}…")

    def on_tool_error(self, error, *_):
        self.add_message("error", f"❌ Tool Error: {error}")

    # Agent callbacks
    def on_agent_action(self, action, *_):
        self.add_message("agent", f"👤 Agent action: {action.tool}")

    def on_agent_finish(self, finish, *_):
        self.add_message("success", "🎯 Agent finished processing")
        if hasattr(finish, "return_values"):
            self.add_message("info", f"Final output: {str(finish.return_values)[:200]}…")


def process_pdf_bytes(file_name: str, file_bytes: bytes) -> Dict[str, Any] | None:
    """Heavy‑lifting routine that parses a PDF (bytes) and returns metadata."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Read, chunk, embed
        # page_texts = convert_pdf_to_text_new(tmp_path)
        # chunks = get_chunked_documents_pipeline(page_texts, tmp_path)
        chunks = get_reducto_chunks(tmp_path)
        if not chunks:
            return None

        vec_dir = f"/tmp/{os.path.basename(file_name)}"
        vectorstore = store_embeddings_excel(chunks, vec_dir)
        description = generate_description(vectorstore)

        return {
            "pdf_file": file_name,
            "description": description,
            "vectorstore": vec_dir,
        }
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def gather_uploads(mode: str) -> List[Tuple[str, bytes]]:
    """Return list of (filename, bytes) ready for processing."""
    uploads: List[Tuple[str, bytes]] = []

    if mode == "Files":
        files = st.file_uploader(
            "Choose one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="file_uploader_files",
        )
        if files:
            uploads = [(f.name, f.getvalue()) for f in files]
    else:  # Folder (ZIP)
        zip_file = st.file_uploader(
            "Upload a ZIP folder containing PDFs",
            type=["zip"],
            accept_multiple_files=False,
            key="file_uploader_zip",
        )
        if zip_file:
            with zipfile.ZipFile(zip_file) as zf:
                for member in zf.namelist():
                    if member.lower().endswith(".pdf"):
                        uploads.append((member, zf.read(member)))

    return uploads


def parallel_process(uploads: List[Tuple[str, bytes]], max_workers: int = 8):
    """Run process_pdf_bytes over all uploads in parallel."""
    progress = st.progress(0.0)
    status = st.empty()
    new_docs: List[Dict[str, Any]] = []

    done = 0
    total = len(uploads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_name = {
            ex.submit(process_pdf_bytes, name, data): name for name, data in uploads
        }
        for future in concurrent.futures.as_completed(future_to_name):
            fname = future_to_name[future]
            try:
                pdf_dict = future.result()
                if pdf_dict:
                    new_docs.append(pdf_dict)
                    st.session_state.processed_files.add(fname)
                    st.success(f"✅ {fname} processed")
                else:
                    st.warning(f"⚠️ {fname}: no valid chunks")
            except Exception as e:
                st.error(f"❌ {fname}: {e}")
            done += 1
            progress.progress(done / total)
            status.info(f"Processed {done}/{total} file(s)")

    progress.empty()
    status.empty()
    return new_docs

def query_documents_with_messages(query: str, pdf_dicts: List[Dict[str, Any]], message_container):
    from ai_analyst import build_tools, build_agent, build_task,build_table_task,build_table_agent

    handler = StreamlitCallbackHandler(message_container)
    llm = LLM(model="openai/gpt-4o", temperature=0, callbacks=[handler])

    handler.add_message("info", f"🔍 Starting query: {query}")
    handler.add_message("info", f"📚 Using {len(pdf_dicts)} document(s)")

    tools = build_tools(pdf_dicts)
    handler.add_message("success", f"✅ Built {len(tools)} search tools")

    agent = build_agent(tools, query, llm)
    task = build_task(agent, query)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)

    handler.add_message("info", "⚡ Executing analysis…")
    result = crew.kickoff()
    handler.add_message("success", "🎉 Analysis completed successfully!")
    # data = extract_tables_from_reports(str(result))
    return str(result)


def query_documents_local(query: str, pdf_dicts: List[Dict[str, Any]]):
    from ai_analyst import answer_query
    try:
        res =answer_query(query, pdf_dicts)
        table_data = extract_tables_from_reports(str(res))
        return table_data
    except Exception as e:
        return f"Error querying documents: {e}"


def query_documents_fastapi(query: str, pdf_dicts: List[Dict[str, Any]]):
    try:
        # Placeholder — your FastAPI endpoint would need to accept pdf_dicts
        return query_documents_local(query, pdf_dicts)
    except Exception as e:
        return f"Error querying FastAPI endpoint: {e}"


def query_documents(query: str, pdf_dicts: List[Dict[str, Any]], message_container=None):
    if message_container and not st.session_state.use_fastapi:
        return query_documents_with_messages(query, pdf_dicts, message_container)
    elif st.session_state.use_fastapi:
        return query_documents_fastapi(query, pdf_dicts)
    else:
        return query_documents_local(query, pdf_dicts)


st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Upload Documents", "Query Documents", "Document Overview", "Settings"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
st.session_state.use_fastapi = st.sidebar.checkbox(
    "Use FastAPI Endpoint", value=st.session_state.use_fastapi
)
st.sidebar.info("🔗 FastAPI enabled" if st.session_state.use_fastapi else "💻 Local processing")

if st.session_state.pdf_dicts:
    st.sidebar.success(f"📚 {len(st.session_state.pdf_dicts)} documents loaded")


if page == "Upload Documents":
    st.header("📤 Upload Documents")
    mode = st.radio("Select upload type", ["Files", "Folder (ZIP)"], horizontal=True)
    uploads = gather_uploads(mode)

    if uploads:
        st.write(f"**{len(uploads)} item(s) ready for processing**")
        for fn, _ in uploads:
            label = "processed" if fn in st.session_state.processed_files else "pending"
            st.markdown(f"- {fn}  _({label})_")
    else:
        st.info("Add some PDFs or a ZIP archive to get started.")

    if uploads and st.button("🚀 Process Documents", type="primary"):
        new_docs = parallel_process(
            [(n, b) for n, b in uploads if n not in st.session_state.processed_files]
        )
        st.session_state.pdf_dicts.extend(new_docs)
        st.success(f"🎉 Added {len(new_docs)} new document(s)")

        for d in new_docs:
            with st.expander(d["pdf_file"]):
                st.write(d["description"])
                st.write(f"Vector store: {d['vectorstore']}")

elif page == "Query Documents":
    st.header("🔍 Query Your Documents")

    if not st.session_state.pdf_dicts:
        st.warning("⚠️ No documents have been processed yet. Please upload and process documents first.")
    else:
        st.success(f"✅ {len(st.session_state.pdf_dicts)} document(s) ready for querying")
        for pdf_dict in st.session_state.pdf_dicts:
            st.write(f"• {pdf_dict['pdf_file']}")

        query = st.text_area("Enter your question:", height=100)

        st.subheader("💡 Example Queries")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("What are the main topics?"):
                query = "What are the main topics discussed in these documents?"
        with col2:
            if st.button("Summarize content"):
                query = "Provide a summary of the key information in these documents."
        with col3:
            if st.button("Find specific info"):
                query = "What specific details can you find about dates, locations, or amounts?"

        if st.button("🔍 Search Documents", type="primary", disabled=not query.strip()):
            response_container = st.container()
            message_container = st.container()
            st.session_state.message_log = []

            with st.spinner("Searching documents…"):
                response = query_documents(query, st.session_state.pdf_dicts, message_container)

            with response_container:
                st.subheader("📝 Final Response")
                if response:
                    st.markdown(f"""<div class='streaming-response'>{response}</div>""", unsafe_allow_html=True)
                else:
                    st.warning("No response received")

elif page == "Document Overview":
    st.header("📊 Document Overview")
    if not st.session_state.pdf_dicts:
        st.info("No documents have been processed yet.")
    else:
        st.success(f"📚 Total Documents: {len(st.session_state.pdf_dicts)}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Documents", len(st.session_state.pdf_dicts))
        col2.metric("Processing Method", "FastAPI" if st.session_state.use_fastapi else "Local")
        col3.metric("Status", "Ready")

        for i, d in enumerate(st.session_state.pdf_dicts, 1):
            st.markdown(
                f"""
                <div class='file-info'>
                    <h4>📄 Document {i}: {d['pdf_file']}</h4>
                    <p><strong>Description:</strong> {d['description'][:200]}…</p>
                    <p><strong>Vector Store:</strong> {d['vectorstore']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"View full description for {d['pdf_file']}"):
                st.write(d["description"])

elif page == "Settings":
    st.header("⚙️ Application Settings")
    st.subheader("Processing Configuration")
    method = st.selectbox(
        "Processing Method", ["Local Processing", "FastAPI Endpoint"], index=1 if st.session_state.use_fastapi else 0
    )
    st.session_state.use_fastapi = method == "FastAPI Endpoint"
    st.info("🔗 Using FastAPI endpoint" if st.session_state.use_fastapi else "💻 Using local processing")

    if st.session_state.use_fastapi:
        st.subheader("FastAPI Configuration")
        fastapi_url = st.text_input("FastAPI Endpoint URL", value="http://localhost:8000")
        if st.button("Test Connection"):
            try:
                r = requests.get(f"{fastapi_url}/docs")
                if r.status_code == 200:
                    st.success("✅ FastAPI endpoint is accessible")
                else:
                    st.warning("⚠️ FastAPI endpoint responded but may not be fully functional")
            except Exception as e:
                st.error(f"❌ Cannot connect to FastAPI endpoint: {e}")

    st.subheader("Data Management")
    if st.button("🗑️ Clear All Documents", type="secondary"):
        st.session_state.pdf_dicts = []
        st.session_state.processed_files = set()
        st.success("All documents cleared!")


st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666'>Built with Streamlit • Powered by AI Analysis</div>",
    unsafe_allow_html=True,
)
light_css = """
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .upload-section { background-color: #f0f2f6; padding: 2rem; border-radius: 10px; margin: 1rem 0; }
    .query-section { background-color: #e8f4fd; padding: 2rem; border-radius: 10px; margin: 1rem 0; }
    .response-section { background-color: #f9f9f9; padding: 2rem; border-radius: 10px; margin: 1rem 0; }
    .file-info { background-color: #e8f5e8; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; }
    .stProgress > div > div > div > div { background-color: #1f77b4; }
    .streaming-response { background-color: #f0f8ff; border-left: 4px solid #1f77b4; padding: 1rem; margin: 1rem 0; font-family: 'Courier New', monospace; color: #000; }
    .message-log { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 1rem; margin: 0.5rem 0; font-family: 'Courier New', monospace; font-size: 0.9rem; max-height: 400px; overflow-y: auto; color: #000; }
    .message-info { color: #0066cc; font-weight: bold; }
    .message-success { color: #28a745; font-weight: bold; }
    .message-warning { color: #ffc107; font-weight: bold; }
    .message-error { color: #dc3545; font-weight: bold; }
    .message-tool { color: #6f42c1; font-weight: bold; }
    .message-agent { color: #fd7e14; font-weight: bold; }
</style>
"""

dark_css = """
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #66b2ff; text-align: center; margin-bottom: 2rem; }
    .upload-section { background-color: #1e1e1e; padding: 2rem; border-radius: 10px; margin: 1rem 0; }
    .query-section { background-color: #222; padding: 2rem; border-radius: 10px; margin: 1rem 0; }
    .response-section { background-color: #2a2a2a; padding: 2rem; border-radius: 10px; margin: 1rem 0; }
    .file-info { background-color: #2c3e50; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; color: #fff; }
    .stProgress > div > div > div > div { background-color: #66b2ff; }
    .streaming-response { background-color: #111; border-left: 4px solid #66b2ff; padding: 1rem; margin: 1rem 0; font-family: 'Courier New', monospace; color: #eee; }
    .message-log { background-color: #222; border: 1px solid #444; border-radius: 5px; padding: 1rem; margin: 0.5rem 0; font-family: 'Courier New', monospace; font-size: 0.9rem; max-height: 400px; overflow-y: auto; color: #eee; }
    .message-info { color: #66b2ff; font-weight: bold; }
    .message-success { color: #28a745; font-weight: bold; }
    .message-warning { color: #ffc107; font-weight: bold; }
    .message-error { color: #dc3545; font-weight: bold; }
    .message-tool { color: #a678ff; font-weight: bold; }
    .message-agent { color: #fd7e14; font-weight: bold; }
</style>
"""

st.markdown(light_css, unsafe_allow_html=True)
