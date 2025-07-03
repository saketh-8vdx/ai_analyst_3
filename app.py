# import streamlit as st
# import requests
# import json
# import tempfile
# import os
# from typing import List, Dict, Any
# import time
# from ai_analyst import read_pdf, create_chunks, store_embeddings_excel, generate_description
# from langchain.callbacks.base import BaseCallbackHandler
# from crewai import LLM

# from processing_files import convert_pdf_to_text_new, get_chunked_documents_pipeline


# # Configure the page
# st.set_page_config(
#     page_title="PDF Document Analyzer",
#     page_icon="📄",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .upload-section {
#         background-color: #f0f2f6;
#         padding: 2rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
#     .query-section {
#         background-color: #e8f4fd;
#         padding: 2rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
#     .response-section {
#         background-color: #f9f9f9;
#         padding: 2rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
#     .file-info {
#         background-color: #e8f5e8;
#         padding: 1rem;
#         border-radius: 5px;
#         margin: 0.5rem 0;
#     }
#     .stProgress > div > div > div > div {
#         background-color: #1f77b4;
#     }
#     .streaming-response {
#         background-color: #f0f8ff;
#         border-left: 4px solid #1f77b4;
#         padding: 1rem;
#         margin: 1rem 0;
#         font-family: 'Courier New', monospace;
#     }
#     .message-log {
#         background-color: #f8f9fa;
#         border: 1px solid #dee2e6;
#         border-radius: 5px;
#         padding: 1rem;
#         margin: 0.5rem 0;
#         font-family: 'Courier New', monospace;
#         font-size: 0.9rem;
#         max-height: 400px;
#         overflow-y: auto;
#     }
#     .message-info {
#         color: #0066cc;
#         font-weight: bold;
#     }
#     .message-success {
#         color: #28a745;
#         font-weight: bold;
#     }
#     .message-warning {
#         color: #ffc107;
#         font-weight: bold;
#     }
#     .message-error {
#         color: #dc3545;
#         font-weight: bold;
#     }
#     .message-tool {
#         color: #6f42c1;
#         font-weight: bold;
#     }
#     .message-agent {
#         color: #fd7e14;
#         font-weight: bold;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'pdf_dicts' not in st.session_state:
#     st.session_state.pdf_dicts = []
# if 'processed_files' not in st.session_state:
#     st.session_state.processed_files = set()
# if 'use_fastapi' not in st.session_state:
#     st.session_state.use_fastapi = False
# if 'message_log' not in st.session_state:
#     st.session_state.message_log = []

# class StreamlitCallbackHandler(BaseCallbackHandler):
#     """Custom callback handler to capture all messages for Streamlit display"""
    
#     def __init__(self, message_container):
#         self.message_container = message_container
#         self.messages = []
        
#     def add_message(self, message_type: str, content: str):
#         """Add a message to the log"""
#         timestamp = time.strftime("%H:%M:%S")
#         message = {
#             "timestamp": timestamp,
#             "type": message_type,
#             "content": content
#         }
#         self.messages.append(message)
        
#         # Store in session state for later display
#         st.session_state.message_log.append(message)
        
#         # Update the Streamlit container
#         with self.message_container:
#             st.markdown(f"""
#             <div class="message-log">
#                 <div class="message-{message_type.lower()}">[{timestamp}] {message_type.upper()}:</div>
#                 <div>{content}</div>
#             </div>
#             """, unsafe_allow_html=True)
    
#     def on_llm_start(self, serialized, prompts, **kwargs):
#         """Called when LLM starts"""
#         self.add_message("info", "🤖 LLM processing started...")
    
#     def on_llm_new_token(self, token, **kwargs):
#         """Called when LLM generates a new token"""
#         # We'll handle this differently to avoid spam
#         pass
    
#     def on_llm_end(self, response, **kwargs):
#         """Called when LLM finishes"""
#         self.add_message("success", "✅ LLM processing completed")
    
#     def on_llm_error(self, error, **kwargs):
#         """Called when LLM encounters an error"""
#         self.add_message("error", f"❌ LLM Error: {str(error)}")
    
#     def on_chain_start(self, serialized, inputs, **kwargs):
#         """Called when a chain starts"""
#         chain_name = serialized.get("name", "Unknown Chain")
#         self.add_message("info", f"🔗 Starting chain: {chain_name}")
    
#     def on_chain_end(self, outputs, **kwargs):
#         """Called when a chain ends"""
#         self.add_message("success", "✅ Chain completed")
    
#     def on_chain_error(self, error, **kwargs):
#         """Called when a chain encounters an error"""
#         self.add_message("error", f"❌ Chain Error: {str(error)}")
    
#     def on_tool_start(self, serialized, input_str, **kwargs):
#         """Called when a tool starts"""
#         tool_name = serialized.get("name", "Unknown Tool")
#         self.add_message("tool", f"🔧 Using tool: {tool_name}")
#         self.add_message("info", f"Input: {input_str[:100]}{'...' if len(input_str) > 100 else ''}")
    
#     def on_tool_end(self, output, **kwargs):
#         """Called when a tool ends"""
#         self.add_message("success", f"✅ Tool completed")
#         if output:
#             self.add_message("info", f"Output: {str(output)[:200]}{'...' if len(str(output)) > 200 else ''}")
    
#     def on_tool_error(self, error, **kwargs):
#         """Called when a tool encounters an error"""
#         self.add_message("error", f"❌ Tool Error: {str(error)}")
    
#     def on_agent_action(self, action, **kwargs):
#         """Called when an agent takes an action"""
#         self.add_message("agent", f"👤 Agent action: {action.tool}")
#         if hasattr(action, 'tool_input'):
#             self.add_message("info", f"Action input: {str(action.tool_input)[:100]}{'...' if len(str(action.tool_input)) > 100 else ''}")
    
#     def on_agent_finish(self, finish, **kwargs):
#         """Called when an agent finishes"""
#         self.add_message("success", "🎯 Agent finished processing")
#         if hasattr(finish, 'return_values'):
#             self.add_message("info", f"Final output: {str(finish.return_values)[:200]}{'...' if len(str(finish.return_values)) > 200 else ''}")

# def process_pdf_file(uploaded_file) -> Dict[str, Any]:
#     """Process a single PDF file and return pdf_dict"""
#     try:
#         # Create temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             tmp_path = tmp_file.name
        
#         # Read PDF text
#         with st.spinner(f"Reading PDF: {uploaded_file.name}"):
#             page_texts = convert_pdf_to_text_new(tmp_path)
#             print(page_texts)
#             # text = read_pdf(tmp_path)
        
#         # if not text.strip():
#         #     st.error(f"No text content found in {uploaded_file.name}")
#         #     return None
        
#         # Create chunks
#         with st.spinner(f"Creating chunks for: {uploaded_file.name}"):
#             # chunks = create_chunks(text, uploaded_file.name)
#             chunks = get_chunked_documents_pipeline(page_texts,tmp_path)
#             print(chunks)
#             print(len(chunks))
        
#         if not chunks:
#             st.error(f"No valid chunks created for {uploaded_file.name}")
#             return None
        
#         # Store embeddings
#         with st.spinner(f"Creating embeddings for: {uploaded_file.name}"):
#             temp_dir = f'/tmp/{uploaded_file.name}'
#             vectorstore = store_embeddings_excel(chunks, temp_dir)
        
#         # Generate description
#         with st.spinner(f"Generating description for: {uploaded_file.name}"):
#             description = generate_description(vectorstore)
#             print(description)
        
#         # Clean up temporary file
#         os.unlink(tmp_path)
        
#         return {
#             "pdf_file": uploaded_file.name,
#             "description": description,
#             "vectorstore": temp_dir
#         }
        
#     except Exception as e:
#         st.error(f"Error processing {uploaded_file.name}: {str(e)}")
#         return None

# def query_documents_with_messages(query: str, pdf_dicts: List[Dict[str, Any]], message_container) -> str:
#     """Query documents and capture all messages"""
#     try:
#         from ai_analyst import build_tools, build_agent, build_task
#         from crewai import Crew, Process
        
#         # Create callback handler
#         callback_handler = StreamlitCallbackHandler(message_container)
        
#         # Create LLM with callbacks
#         llm = LLM(
#             model="openai/gpt-4o",
#             temperature=0,
#             callbacks=[callback_handler]
#         )
        
#         # Add initial message
#         callback_handler.add_message("info", f"🔍 Starting query: {query}")
#         callback_handler.add_message("info", f"📚 Using {len(pdf_dicts)} document(s)")
        
#         # Build tools
#         callback_handler.add_message("info", "🔧 Building search tools...")
#         tools = build_tools(pdf_dicts)
#         callback_handler.add_message("success", f"✅ Built {len(tools)} search tools")
        
#         # Build agent
#         callback_handler.add_message("info", "👤 Creating AI agent...")
#         qa_agent = build_agent(tools, query, llm)
#         callback_handler.add_message("success", "✅ Agent created successfully")
        
#         # Build task
#         callback_handler.add_message("info", "📋 Creating analysis task...")
#         task = build_task(qa_agent, query)
#         callback_handler.add_message("success", "✅ Task created successfully")
        
#         # Create crew
#         callback_handler.add_message("info", "🚀 Starting CrewAI analysis...")
#         crew = Crew(
#             agents=[qa_agent],
#             tasks=[task],
#             process=Process.sequential,
#             verbose=True
#         )
        
#         # Execute
#         callback_handler.add_message("info", "⚡ Executing analysis...")
#         result = crew.kickoff()
        
#         callback_handler.add_message("success", "🎉 Analysis completed successfully!")
        
#         return str(result)
        
#     except Exception as e:
#         error_msg = f"Error querying documents: {str(e)}"
#         message_container.error(error_msg)
#         return error_msg

# def query_documents_local(query: str, pdf_dicts: List[Dict[str, Any]]) -> str:
#     """Query documents using local ai_analyst functions"""
#     try:
#         from ai_analyst import answer_query
#         result = answer_query(query, pdf_dicts)
#         return result
#     except Exception as e:
#         return f"Error querying documents: {str(e)}"

# def query_documents_fastapi(query: str, pdf_dicts: List[Dict[str, Any]]) -> str:
#     """Query documents using FastAPI endpoint"""
#     try:
#         # For now, we'll use the local function since the FastAPI endpoint 
#         # expects pdf_dicts to be already available in the server
#         # In a production setup, you'd modify the FastAPI endpoint to accept pdf_dicts
        
#         # This is a placeholder for the actual FastAPI call
#         # response = requests.get(f"http://localhost:8000/ask?query={query}")
#         # return response.text
        
#         # For now, use local function
#         return query_documents_local(query, pdf_dicts)
        
#     except Exception as e:
#         return f"Error querying FastAPI endpoint: {str(e)}"

# def query_documents(query: str, pdf_dicts: List[Dict[str, Any]], message_container=None) -> str:
#     """Query documents using selected method"""
#     if message_container and not st.session_state.use_fastapi:
#         return query_documents_with_messages(query, pdf_dicts, message_container)
#     elif st.session_state.use_fastapi:
#         return query_documents_fastapi(query, pdf_dicts)
#     else:
#         return query_documents_local(query, pdf_dicts)

# # Main application
# st.markdown('<h1 class="main-header">📄 PDF Document Analyzer</h1>', unsafe_allow_html=True)

# # Sidebar for navigation and settings
# st.sidebar.title("Navigation")
# page = st.sidebar.selectbox(
#     "Choose a page",
#     ["Upload Documents", "Query Documents", "Document Overview", "Settings"]
# )

# # Settings in sidebar
# st.sidebar.markdown("---")
# st.sidebar.subheader("⚙️ Settings")
# st.session_state.use_fastapi = st.sidebar.checkbox(
#     "Use FastAPI Endpoint", 
#     value=st.session_state.use_fastapi,
#     help="Toggle between local processing and FastAPI endpoint"
# )

# if st.session_state.use_fastapi:
#     st.sidebar.info("🔗 Using FastAPI endpoint")
# else:
#     st.sidebar.info("💻 Using local processing")

# # Show current status
# if st.session_state.pdf_dicts:
#     st.sidebar.success(f"📚 {len(st.session_state.pdf_dicts)} documents loaded")

# if page == "Upload Documents":
#     st.markdown('<div class="upload-section">', unsafe_allow_html=True)
#     st.header("📤 Upload PDF Documents")
    
#     uploaded_files = st.file_uploader(
#         "Choose PDF files",
#         type=['pdf'],
#         accept_multiple_files=True,
#         help="Upload one or more PDF files to analyze"
#     )
    
#     if uploaded_files:
#         st.write(f"**Uploaded {len(uploaded_files)} file(s):**")
        
#         # Show file list
#         for file in uploaded_files:
#             if file.name in st.session_state.processed_files:
#                 st.success(f"✅ {file.name} (already processed)")
#             else:
#                 st.info(f"📄 {file.name}")
        
#         # Process files button
#         if st.button("🚀 Process Documents", type="primary"):
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             new_pdf_dicts = []
            
#             for i, uploaded_file in enumerate(uploaded_files):
#                 if uploaded_file.name not in st.session_state.processed_files:
#                     status_text.text(f"Processing {uploaded_file.name}...")
                    
#                     pdf_dict = process_pdf_file(uploaded_file)
#                     if pdf_dict:
#                         new_pdf_dicts.append(pdf_dict)
#                         st.session_state.processed_files.add(uploaded_file.name)
                    
#                     progress_bar.progress((i + 1) / len(uploaded_files))
            
#             # Add new pdf_dicts to session state
#             st.session_state.pdf_dicts.extend(new_pdf_dicts)
            
#             status_text.text("✅ Processing complete!")
#             st.success(f"Successfully processed {len(new_pdf_dicts)} new document(s)")
            
#             # Show processed documents
#             if st.session_state.pdf_dicts:
#                 st.subheader("📋 Processed Documents")
#                 for pdf_dict in st.session_state.pdf_dicts:
#                     with st.expander(f"📄 {pdf_dict['pdf_file']}"):
#                         st.write("**Description:**")
#                         st.write(pdf_dict['description'])
#                         st.write(f"**Vector Store:** {pdf_dict['vectorstore']}")
    
#     st.markdown('</div>', unsafe_allow_html=True)

# elif page == "Query Documents":
#     st.markdown('<div class="query-section">', unsafe_allow_html=True)
#     st.header("🔍 Query Your Documents")
    
#     if not st.session_state.pdf_dicts:
#         st.warning("⚠️ No documents have been processed yet. Please upload and process documents first.")
#         st.info("Go to the 'Upload Documents' page to get started.")
#     else:
#         st.success(f"✅ {len(st.session_state.pdf_dicts)} document(s) ready for querying")
        
#         # Show available documents
#         st.subheader("📚 Available Documents")
#         for pdf_dict in st.session_state.pdf_dicts:
#             st.write(f"• {pdf_dict['pdf_file']}")
        
#         # Query input
#         query = st.text_area(
#             "Enter your question:",
#             placeholder="Ask anything about your uploaded documents...",
#             height=100
#         )
        
#         st.subheader("💡 Example Queries")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             if st.button("What are the main topics?"):
#                 st.session_state.example_query = "What are the main topics discussed in these documents?"
#         with col2:
#             if st.button("Summarize content"):
#                 st.session_state.example_query = "Provide a summary of the key information in these documents."
#         with col3:
#             if st.button("Find specific info"):
#                 st.session_state.example_query = "What specific details can you find about dates, locations, or amounts?"
        
#         # Use example query if selected
#         if 'example_query' in st.session_state:
#             query = st.session_state.example_query
#             del st.session_state.example_query
        
#         if st.button("🔍 Search Documents", type="primary", disabled=not query.strip()):
#             if query.strip():
#                 # Create containers for response and messages
#                 response_container = st.container()
#                 message_container = st.container()
                
#                 # Clear previous messages
#                 st.session_state.message_log = []
                
#                 with st.spinner("Searching documents..."):
#                     response = query_documents(query, st.session_state.pdf_dicts, message_container)
                
                
#                 with message_container:
#                     st.subheader("🔍 Processing Steps")
#                     if st.session_state.message_log:
#                         for message in st.session_state.message_log:
#                             st.markdown(f"""
#                             <div class="message-log">
#                                 <div class="message-{message['type'].lower()}">[{message['timestamp']}] {message['type'].upper()}:</div>
#                                 <div>{message['content']}</div>
#                             </div>
#                             """, unsafe_allow_html=True)
#                     else:
#                         st.info("No processing messages captured")
                
#                 with response_container:
#                     st.markdown('<div class="response-section">', unsafe_allow_html=True)
#                     st.subheader("📝 Final Response")
                    
#                     # Display response with formatting
#                     if response:
#                         st.markdown('<div class="streaming-response">', unsafe_allow_html=True)
#                         st.write(response)
#                         st.markdown('</div>', unsafe_allow_html=True)
#                     else:
#                         st.warning("No response received")
                    
#                     st.markdown('</div>', unsafe_allow_html=True)
    
#     st.markdown('</div>', unsafe_allow_html=True)

# elif page == "Document Overview":
#     st.markdown('<div class="response-section">', unsafe_allow_html=True)
#     st.header("📊 Document Overview")
    
#     if not st.session_state.pdf_dicts:
#         st.info("No documents have been processed yet.")
#     else:
#         st.success(f"📚 Total Documents: {len(st.session_state.pdf_dicts)}")
        
#         # Summary statistics
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Total Documents", len(st.session_state.pdf_dicts))
#         with col2:
#             st.metric("Processing Method", "FastAPI" if st.session_state.use_fastapi else "Local")
#         with col3:
#             st.metric("Status", "Ready" if st.session_state.pdf_dicts else "No Documents")
        
#         # Document details
#         for i, pdf_dict in enumerate(st.session_state.pdf_dicts, 1):
#             st.markdown(f"""
#             <div class="file-info">
#                 <h4>📄 Document {i}: {pdf_dict['pdf_file']}</h4>
#                 <p><strong>Description:</strong> {pdf_dict['description'][:200]}...</p>
#                 <p><strong>Vector Store:</strong> {pdf_dict['vectorstore']}</p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             with st.expander(f"View full description for {pdf_dict['pdf_file']}"):
#                 st.write(pdf_dict['description'])
    
#     st.markdown('</div>', unsafe_allow_html=True)

# elif page == "Settings":
#     st.markdown('<div class="response-section">', unsafe_allow_html=True)
#     st.header("⚙️ Application Settings")
    
#     st.subheader("Processing Configuration")
    
#     # Processing method
#     processing_method = st.selectbox(
#         "Processing Method",
#         ["Local Processing", "FastAPI Endpoint"],
#         index=1 if st.session_state.use_fastapi else 0
#     )
    
#     if processing_method == "FastAPI Endpoint":
#         st.session_state.use_fastapi = True
#         st.info("🔗 Using FastAPI endpoint for document processing and querying")
#     else:
#         st.session_state.use_fastapi = False
#         st.info("💻 Using local processing for document processing and querying")
    
#     # FastAPI endpoint configuration
#     if st.session_state.use_fastapi:
#         st.subheader("FastAPI Configuration")
#         fastapi_url = st.text_input(
#             "FastAPI Endpoint URL",
#             value="http://localhost:8000",
#             help="URL of the FastAPI server"
#         )
        
#         # Test connection
#         if st.button("Test Connection"):
#             try:
#                 response = requests.get(f"{fastapi_url}/docs")
#                 if response.status_code == 200:
#                     st.success("✅ FastAPI endpoint is accessible")
#                 else:
#                     st.warning("⚠️ FastAPI endpoint responded but may not be fully functional")
#             except Exception as e:
#                 st.error(f"❌ Cannot connect to FastAPI endpoint: {str(e)}")
    
#     # Clear data
#     st.subheader("Data Management")
#     if st.button("🗑️ Clear All Documents", type="secondary"):
#         st.session_state.pdf_dicts = []
#         st.session_state.processed_files = set()
#         st.success("All documents cleared!")
    
#     st.markdown('</div>', unsafe_allow_html=True)

# # Footer
# st.markdown("---")
# st.markdown(
#     """
#     <div style='text-align: center; color: #666;'>
#         <p>Built with Streamlit • Powered by AI Analysis</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# ) 





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

from ai_analyst import read_pdf, create_chunks, store_embeddings_excel, generate_description
from processing_files import convert_pdf_to_text_new, get_chunked_documents_pipeline
from langchain.callbacks.base import BaseCallbackHandler
from crewai import LLM, Crew, Process

# ──────────────────────────────────────────────────────────────────────────
# ░▓ PAGE / THEME CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# ░▓ CUSTOM CSS (unchanged styling from v1)
# ──────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────
# ░▓ SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────────
for key, default in {
    "pdf_dicts": [],
    "processed_files": set(),
    "use_fastapi": False,
    "message_log": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────────────────────────────────
# ░▓ STREAMLIT CALLBACK HANDLER (unchanged)
# ──────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────
# ░▓ PARALLEL PDF PROCESSING HELPERS
# ──────────────────────────────────────────────────────────────────────────

def process_pdf_bytes(file_name: str, file_bytes: bytes) -> Dict[str, Any] | None:
    """Heavy‑lifting routine that parses a PDF (bytes) and returns metadata."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Read, chunk, embed
        page_texts = convert_pdf_to_text_new(tmp_path)
        chunks = get_chunked_documents_pipeline(page_texts, tmp_path)
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


def parallel_process(uploads: List[Tuple[str, bytes]], max_workers: int = 4):
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

# ──────────────────────────────────────────────────────────────────────────
# ░▓ QUERY HELPERS (unchanged from v1)
# ──────────────────────────────────────────────────────────────────────────

def query_documents_with_messages(query: str, pdf_dicts: List[Dict[str, Any]], message_container):
    from ai_analyst import build_tools, build_agent, build_task

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
    return str(result)


def query_documents_local(query: str, pdf_dicts: List[Dict[str, Any]]):
    from ai_analyst import answer_query
    try:
        return answer_query(query, pdf_dicts)
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

# ──────────────────────────────────────────────────────────────────────────
# ░▓ SIDEBAR NAVIGATION & GLOBAL SETTINGS
# ──────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────
# ░▓ PAGE LOGIC
# ──────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────
# ░▓ FOOTER
# ──────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666'>Built with Streamlit • Powered by AI Analysis</div>",
    unsafe_allow_html=True,
)
