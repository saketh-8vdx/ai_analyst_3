__import__("pysqlite3")          # registers the modern _sqlite3 C-extension
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import os, faiss
from typing import List, Dict, Any, Type
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process          # core framework
from crewai.tools import BaseTool                      # custom tools
from langchain.vectorstores.faiss import FAISS         # your indices
from langchain.embeddings import HuggingFaceEmbeddings # or OpenAIEmbeddings
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from pydantic import Field
from langchain.callbacks.base import AsyncCallbackHandler
from crewai import Agent, Crew, Process
import streamlit as st
import openai
from crewai import LLM
import json
# stream_llm = LLM(
#     model="openai/o3"
# )

stream_llm = LLM(
    model="openai/gpt-4o",   
    stream=True,             
    temperature=0
)


openai.api_key = st.secrets["OPENAI_API_KEY"]

from typing import Any
import re

class VectorStoreSearchInput(BaseModel):
    """Input schema for VectorStoreSearchTool"""
    query: str = Field(description="Search query to find relevant information")

class VectorStoreSearchTool(BaseTool):
    """Custom tool for searching vector stores"""
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of what this tool does")
    vectorstore: str = Field(description="FAISS vectorstore instance")
    args_schema: type[BaseModel] = VectorStoreSearchInput
    
    def _run(self, query: str) -> str:
        """Execute the search"""
        try:
            # Search for relevant documents

            from langchain.embeddings.openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

            vectorstore_new = FAISS.load_local(self.vectorstore, embeddings, allow_dangerous_deserialization=True)
            docs = vectorstore_new.similarity_search(query, k=3)
            # docs = self.vectorstore.similarity_search(query, k=3)
            
            if not docs:
                return "No relevant information found."
            
            # Combine results
            results = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content  # Limit content length
                results.append(f"Result {i}: {content}")
            
            return "\n\n".join(results)
        except Exception as e:
            return f"Error searching vectorstore: {str(e)}"

def build_tools(pdf_dicts: List[Dict[str, Any]]) -> List[BaseTool]:
    """Build tools from PDF dictionaries"""
    print(f"Building tools from {len(pdf_dicts)} PDF dictionaries")
    tools = []
    
    for d in pdf_dicts:
        if d is None:
            continue
            
        tool = VectorStoreSearchTool(
            name=f"search_{d['pdf_file']}",
            description=(
                f"Use when the query relates to: {d['description']}. "
                f"PDF source: {d['pdf_file']}"
            ),
            vectorstore=d["vectorstore"]
        )
        tools.append(tool)
    
    return tools

# def build_agent(all_tools: List[BaseTool],query: str,llm) -> Agent:
#     return Agent(
#         role="Document QA Expert",
#         goal=(
#             "For every file you have a tool with descriptions of the PDFs.  \n"
#             f"by clearly understanding the users query which is {query}, you need to decide which tools to use \n" 
#             "in order to fetch the context that will help you answer the query.  \n"
#             "Give the final answer as the response to the user query.  \n"
#         ),
#         backstory=(
#             "You have one search tool per PDF.  \n"
#             "For every query: decide which PDFs match, run those tools "
#             "one-by-one, read the returned text, and craft a concise answer."
#             "You can choose any number of tools based on the input query and the descriptions of the PDFs.  \n"
#         ),
#         allow_code_execution=False,  # set True if you’d like code-scratch-pad
#         tools=all_tools,
#         llm = stream_llm,
#         verbose=True

   
#     )

# def build_task(agent: Agent, query: str) -> Task:
#     return Task(
#         description=(
#             "Think step-by-step.\n"
#             f"**Question to answer:** {query}\n\n"
#             "Decide which PDFs are relevant based on their descriptions.  \n"
#             "Call the corresponding search_* tools (one or more) in order to answer the users query \n"
#             "Synthesise a ≤250-word answer, citing snippets verbatim "
#             "when helpful."
#         ),
#         expected_output="Final answer to be shown to the user",
#         agent=agent,
#         async_execution=False,
#         input_variables={"question": query},  
#     )


def build_agent(all_tools: List[BaseTool], query: str,llm) -> Agent:
    return Agent(
        role="Comprehensive Document Analysis Expert",
        goal=(
            "You are an expert at analyzing user queries and providing comprehensive answers by using "
            "ALL relevant document search tools. You must first understand the query thoroughly, "
            "then use MULTIPLE document search tools to gather complete information from all available sources."
        ),
        backstory=(
            "You have extensive experience in document analysis and information retrieval. "
            "You excel at understanding the nuances of user questions and using ALL available sources. "
            "You always think step-by-step: first analyze the query, then evaluate which documents could contain "
            "relevant information, and finally use MULTIPLE search tools to gather comprehensive context. "
            "You believe in being thorough rather than selective, using more tools rather than fewer to ensure "
            "complete coverage of the available information."
        ),
        allow_code_execution=False,
        tools=all_tools,
        llm=stream_llm,
        verbose=True
    )

def build_task(agent: Agent, query: str) -> Task:
    return Task(
        description=(
            "**Step-by-Step Analysis Process:**\n\n"
            "**Step 1: Query Understanding**\n"
            f"- Carefully analyze the user's question: '{query}'\n"
            "- Identify the key topics, entities, and information needs\n"
            "- Determine what type of information would best answer this query\n"
            "- Consider if the query is about financial data, company overview, legal matters, etc.\n\n"
            
            "**Step 2: Document Selection Strategy**\n"
            "- Review ALL available document descriptions for each search tool\n"
            "- Identify which documents could potentially contain relevant information\n"
            "- Consider document types (financial reports, presentations, agreements, etc.)\n"
            "- For comprehensive answers, use MULTIPLE tools when documents might contain relevant info\n"
            "- For financial queries, check both financial reports AND presentations\n"
            "- For company overview queries, check investor presentations AND financial reports\n"
            "- For legal queries, check agreements AND any related documents\n"
            "- When in doubt, use MORE tools rather than fewer to ensure comprehensive coverage\n\n"
            
            "**Step 3: Context Gathering**\n"
            "- Use ALL potentially relevant search tools to gather comprehensive context\n"
            "- For each tool you use, explain why you chose it based on the query and document description\n"
            "- Search with specific, targeted queries that match the user's information needs\n"
            "- Use the document descriptions to formulate better search queries\n"
            "- IMPORTANT: Use multiple tools if the query could be answered from different document types\n\n"
            
            "**Step 4: Answer Synthesis**\n"
            "- Combine information from ALL relevant sources you searched\n"
            "- Provide a comprehensive, well-structured answer (≤250 words)\n"
            "- Include specific details and cite relevant information from the documents\n"
            "- Ensure the answer directly addresses the user's original question\n"
            "- If information is found in multiple documents, synthesize it coherently\n\n"
            
            "**Available Documents:**\n"
            "- You have access to multiple PDF documents with different purposes and content\n"
            "- Each document has a specific description explaining its content and purpose\n"
            "- Use the search tools comprehensively based on these descriptions\n\n"
            
            "**CRITICAL GUIDELINES:**\n"
            "- ALWAYS use MULTIPLE tools when the query could be answered from different document types\n"
            "- Explain your reasoning for selecting each document\n"
            "- If you're unsure about relevance, USE THE TOOL ANYWAY - better to have more information\n"
            "- For general queries like 'summary' or 'overview', use ALL available tools\n"
            "- Be comprehensive rather than selective in your tool usage\n"
            "- Use the document descriptions to guide your search strategy\n"
            "- For the query 'give me the summary', you MUST use ALL available tools to provide a comprehensive overview\n"
            "- When the query is broad or general, default to using ALL tools for maximum coverage\n"
        ),
        expected_output=(
            "A comprehensive answer that includes:\n"
            "1. Brief explanation of document selection reasoning\n"
            "2. Detailed answer to the user's query\n"
            "3. Specific information and citations from relevant documents\n"
            "4. Clear, well-structured response that directly addresses the question"
        ),
        agent=agent,
        async_execution=False,
        input_variables={"question": query},
    )

TABLE_FUNCTION_SCHEMA = {
    "name": "generate_table_response",
    "description": "Generate a table response in JSON format based on the user query and extracted content",
    "parameters": {
        "type": "object",
        "properties": {
            "Table_Name": {
                "type": "string",
                "description": "Name of the table present in the document."
            },
            "column_headers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of column headers for the table"
            },
            "rows": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Each row is an array of cell values"
                },
                "description": "List of rows, where each row is a list of cell values"
            },
            "Description": {
                "type": "string",
                "description": "Description of the table, what table about, what is the data in the table in one sentence only."
            }
        },
        "required": ["Table_Name","column_headers", "rows", "Description"]
    }
}

# def build_table_agent(all_tools: List[BaseTool], query: str, llm) -> Agent:
#     return Agent(
#         role="JSON Table Analysis and Generation Specialist",
#         goal=(
#             "Analyze user queries to understand what tabular data is requested and generate perfect JSON tables. "
#             f"Current query: {query}\n\n"
#             "Your process:\n"
#             "1. Analyze the query to identify what type of table/data is needed\n"
#             "2. Use relevant PDF tools to extract content based on the query\n"
#             "3. Check if the extracted content contains the requested table\n"
#             "4. If table exists: convert it to JSON format\n"
#             "5. If table doesn't exist: generate a table from available data\n"
#             "6. Always return response using the generate_table_response function with the exact structure: {\"Table_Name\": \"\", \"column_headers\": [], \"rows\": [[], [], []], \"Description\": \"\"}\n"
#         ),
#         backstory=(
#             "You are an expert table analyst and generator with deep understanding of data structures and JSON formatting. "
#             "You have access to multiple PDF search tools, each containing different document content. "
#             "Your expertise includes:\n"
#             "- Identifying table requirements from user queries\n"
#             "- Recognizing existing tables in extracted content\n"
#             "- Generating new tables from unstructured data\n"
#             "- Creating perfectly formatted JSON tables\n"
#             "- Understanding data relationships and hierarchies\n"
#             "- Converting complex data into clear JSON tabular formats\n\n"
#             "You must ALWAYS use the generate_table_response function to return your response. Never return plain text or explanations outside the JSON structure."
#         ),
#         allow_code_execution=False,
#         tools=all_tools,
#         llm=llm,
#         verbose=True,
#         function_calling=True,
#         functions=[TABLE_FUNCTION_SCHEMA]
#     )

# def build_table_task(agent: Agent, query: str) -> Task:
#     return Task(
#         description=(
#             "**JSON Table Analysis and Generation Task**\n\n"
#             f"**User Query:** {query}\n\n"
#             "**Step-by-step process:**\n"
#             "1. **Query Analysis**: Determine what type of table or tabular data the user is requesting\n"
#             "2. **Tool Selection**: Identify which PDF search tools are relevant based on the query\n"
#             "3. **Content Extraction**: Use the selected tools to extract relevant content\n"
#             "4. **Table Detection**: Check if the extracted content already contains the requested table\n"
#             "5. **Table Generation**: \n"
#             "   - If table exists: Extract and convert it to JSON format\n"
#             "   - If table doesn't exist: Generate a new table from the available data\n"
#             "6. **Function Call**: Use the generate_table_response function to return the result\n\n"
#             "**CRITICAL OUTPUT REQUIREMENTS:**\n"
#             "- You MUST use the generate_table_response function to return your response\n"
#             "- The function will automatically format the data as: {\"column_headers\": [], \"rows\": [[], [], []]}\n"
#             "- No additional text, explanations, or markdown formatting\n"
#             "- Include all relevant data from the extracted content\n"
#             "- Ensure the table directly answers the user's query\n"
#             "- The response will be automatically validated as proper JSON\n"
#         ),
#         expected_output="Function call to generate_table_response with properly structured table data that answers the user query",
#         agent=agent,
#         async_execution=False,
#         input_variables={"question": query},
#     )



# def build_table_agent(all_tools: List[BaseTool], query: str, llm) -> Agent:
#     return Agent(
#         role="JSON Table Analysis and Generation Specialist",
#         goal=(
#             "Analyze user queries to understand what tabular data is requested and generate perfect JSON tables. "
#             f"Current query: {query}\n\n"
#             "Your process:\n"
#             "1. Analyze the query to identify what type of table/data is needed\n"
#             "2. Use relevant PDF tools to extract content based on the query\n"
#             "3. Check if the extracted content contains the requested table\n"
#             "4. If table exists: convert it to JSON format\n"
#             "5. If table doesn't exist: generate a table from available data\n"
#             "6. Always return response using the generate_table_response function with the exact structure: {\"Table_Name\": \"\", \"column_headers\": [], \"rows\": [[], [], []], \"Description\": \"\"}\n"
#         ),
#         backstory=(
#             "You are an expert table analyst and generator with deep understanding of data structures and JSON formatting. "
#             "You have access to multiple PDF search tools, each containing different document content. "
#             "Your expertise includes:\n"
#             "- Identifying table requirements from user queries\n"
#             "- Recognizing existing tables in extracted content\n"
#             "- Generating new tables from unstructured data\n"
#             "- Creating perfectly formatted JSON tables\n"
#             "- Understanding data relationships and hierarchies\n"
#             "- Converting complex data into clear JSON tabular formats\n\n"
#             "You must ALWAYS use the generate_table_response function to return your response. Never return plain text or explanations outside the JSON structure."
#         ),
#         allow_code_execution=False,
#         tools=all_tools,
#         llm=llm,
#         verbose=True,
#         function_calling=True,
#         functions=[TABLE_FUNCTION_SCHEMA]
#     )


# def build_table_task(agent: Agent, query: str) -> Task:
#     return Task(
#         description=(
#             "**JSON Table Analysis and Generation Task**\n\n"
#             f"**User Query:** {query}\n\n"
#             "**Step-by-step process:**\n"
#             "1. **Query Analysis**: Determine what type of table or tabular data the user is requesting\n"
#             "2. **Tool Selection**: Identify which PDF search tools are relevant based on the query\n"
#             "3. **Content Extraction**: Use the selected tools to extract relevant content\n"
#             "4. **Table Detection**: Check if the extracted content already contains the requested table\n"
#             "5. **Table Generation**: \n"
#             "   - If table exists: Extract and convert it to JSON format\n"
#             "   - If table doesn't exist: Generate a new table from the available data\n"
#             "6. **Function Call**: Use the generate_table_response function to return the result\n\n"
#             "**CRITICAL OUTPUT REQUIREMENTS:**\n"
#             "- You MUST use the generate_table_response function to return your response\n"
#             "- The function requires EXACT structure: {\"Table_Name\": \"\", \"column_headers\": [], \"rows\": [[], [], []], \"Description\": \"\"}\n"
#             "- NO additional text, explanations, or markdown formatting\n"
#             "- Include all relevant data from the extracted content\n"
#             "- Ensure the table directly answers the user's query\n"
#             "- The response will be automatically validated as proper JSON\n"
#             "- Table_Name: Provide a descriptive name for the table\n"
#             "- column_headers: List of column names as strings\n"
#             "- rows: List of lists, where each inner list represents a row of data\n"
#             "- Description: One sentence describing what the table contains\n"
#         ),
#         expected_output="Function call to generate_table_response with properly structured table data that answers the user query, A valid JSON object containing Table Name, Columns, Rows, Description",
#         agent=agent,
#         async_execution=False,
#         input_variables={"question": query},
#     )



def build_table_agent(all_tools: List[BaseTool], query: str, llm) -> Agent:
    return Agent(
        role="JSON Table Analysis and Generation Specialist",
        goal=(
            "Analyze user queries to understand what tabular data is requested and generate perfect JSON tables. "
            f"Current query: {query}\n\n"
            "Your process:\n"
            "1. Analyze the query to identify what type of table/data is needed\n"
            "2. Use relevant PDF tools to extract content based on the query\n"
            "3. Check if the extracted content contains the requested table\n"
            "4. If table exists: convert it to JSON format\n"
            "5. If table doesn't exist: generate a table from available data\n"
            "6. Always return response using the generate_table_response function with the exact structure: {\"Table_Name\": \"\", \"column_headers\": [], \"rows\": [[], [], []]}\n"
        ),
        backstory=(
            "You are an expert table analyst and generator with deep understanding of data structures and JSON formatting. "
            "You have access to multiple PDF search tools, each containing different document content. "
            "Your expertise includes:\n"
            "- Identifying table requirements from user queries\n"
            "- Recognizing existing tables in extracted content\n"
            "- Generating new tables from unstructured data\n"
            "- Creating perfectly formatted JSON tables\n"
            "- Understanding data relationships and hierarchies\n"
            "- Converting complex data into clear JSON tabular formats\n\n"
            "You must ALWAYS use the generate_table_response function to return your response. Never return plain text or explanations outside the JSON structure."
        ),
        allow_code_execution=False,
        tools=all_tools,
        llm=llm,
        verbose=True
    )


def build_table_task(agent: Agent, query: str) -> Task:
    return Task(
        description=(
            "**JSON Table Analysis and Generation Task**\n\n"
            f"**User Query:** {query}\n\n"
            "**Step-by-step process:**\n"
            "1. **Query Analysis**: Determine what type of table or tabular data the user is requesting\n"
            "2. **Tool Selection**: Identify which PDF search tools are relevant based on the query\n"
            "3. **Content Extraction**: Use the selected tools to extract relevant content\n"
            "4. **Table Detection**: Check if the extracted content already contains the requested table\n"
            "5. **Table Generation**: \n"
            "   - If table exists: Extract and convert it to JSON format\n"
            "   - If table doesn't exist: Generate a new table from the available data\n"
            "6. **Function Call**: Use the generate_table_response function to return the result\n\n"
            "**CRITICAL OUTPUT REQUIREMENTS:**\n"
            "- You MUST use the generate_table_response function to return your response\n"
            "- The function requires EXACT structure: {\"Table_Name\": \"\", \"column_headers\": [], \"rows\": [[], [], []]}\n"
            "- NO additional text, explanations, or markdown formatting\n"
            "- Include all relevant data from the extracted content\n"
            "- Ensure the table directly answers the user's query\n"
            "- The response will be automatically validated as proper JSON\n"
            "- Table_Name: Provide a descriptive name for the table\n"
            "- column_headers: List of column names as strings\n"
            "- rows: List of lists, where each inner list represents a row of data\n"
            "- Description: One sentence describing what the table contains\n"
            "- If no data available return empty table with all the keys\n"
            "- Properly handle the case where no data is available, do not return any description only valid json with all the keys retur in final result.\n"
        ),
        expected_output="Function call to generate_table_response with properly structured table data that answers the user query, A valid JSON object containing Table Name, Columns, Rows",
        agent=agent,
        async_execution=False,
        input_variables={"question": query},
    )





def answer_query(query: str, pdf_dicts: List[Dict[str, Any]], verbose: bool = True):
    tools = build_tools(pdf_dicts)
    qa_agent = build_table_agent(tools,query)
    task = build_table_task(qa_agent, query)

    crew = Crew(
        agents=[qa_agent],
        tasks=[task],
        process=Process.hierarchical,
        verbose=verbose, # NEW
    planning=True,                  # let CrewAI auto-chunk the plan
    manager_llm="gpt-4o"
    )
    return crew.kickoff()


def extract_tables_from_reports(final_result=None):
    """
    
    Parameters:
      vectorstore: vectorstore instance.
      query: Instructions used to generate the tables IN proper json format from given unstructured json table data.
      system_prompt: The system instruction to guide the Table generation.
    
    Returns:
      A JSON object containing generated table with with keys "columns" (a list of column headers, hierarchies flattened) and "rows" (a list of row entries).
    """

    query = "Understand the unstructured json table data and generate the tables in proper json format. \n You will get the Table name, Columns and Rows in the unstructured json data. \n understand it promperly and generate the final result as strict json object of table with proper Table name , Columns and Rows. \n Properly handle the case where no data is available, do not return any description only valid json with all the keys retur in final result."
    system_prompt = "You are a helpful assistant that generates tables in proper json format from unstructured json table data."

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": (
                f"Query :\n {query} \n\n Unstructured JSON table data :\n {final_result}"
            )
        }
    ]
    
    functions = [
        {
            "name": "extract_tables",
            "description": (
                "Generate the tables in proper json format from an Unstructured JSON table data."
                "Return the result as a valid JSON object with keys: "
                "'headers' (a list of flat column headers, with hierarchical names combined), "
                "and 'rows' (a list of rows, where each row is a list of cell values including row headers)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "object",
                        "properties": {
                            "headers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Flat list of column headers after combining hierarchical names."
                            },
                            "rows": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "A single row of cell values, matching the headers."
                                },
                                "description": "All rows of the table."
                            }
                        },
                        "required": ["headers", "rows"]
                    }
                },
                "required": ["table"]
            }
        }
    ]

    # Call the GPT‑4 model (using a model variant that supports function calling, e.g. gpt-4-0314)
    response = openai.chat.completions.create(
        model="gpt-4o",  # or your designated GPT-4o model identifier
        messages=messages,
        functions=functions,
        function_call={"name": "extract_tables"},
    )
    
    if response.choices[0].message.function_call:
        function_arguments = response.choices[0].message.function_call.arguments
        print("function_arguments: ",function_arguments)
        try:
            function_arguments = json.loads(function_arguments)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Problematic text: {function_arguments}..")
        
        print(function_arguments)
        return function_arguments
    else:
        # In case no function call is returned, return the plain text
        return {"error": "No function call returned."}





#second version



# import asyncio
# from typing import AsyncGenerator, List, Dict, Any
# from langchain.callbacks.base import AsyncCallbackHandler
# from crewai import LLM, Crew, Process

# # Global LLM configuration
# _MODEL = "gpt-4o"
# _TEMPERATURE = 0.1
# _STREAM = True

# class QueueCallback(AsyncCallbackHandler):
#     """Forward every new LLM token into an asyncio.Queue."""
#     def __init__(self, queue: asyncio.Queue):
#         self.queue = queue
#         self._finished = False

#     async def on_llm_new_token(self, token: str, **kwargs):
#         if not self._finished:
#             await self.queue.put(token)

#     async def on_chain_end(self, outputs, **kwargs):
#         if not self._finished:
#             self._finished = True
#             await self.queue.put("[DONE]")

#     async def on_chain_error(self, error, **kwargs):
#         if not self._finished:
#             self._finished = True
#             await self.queue.put(f"[ERROR]: {str(error)}")
#             await self.queue.put("[DONE]")

# def new_streaming_llm(callback: QueueCallback) -> LLM:
#     return LLM(
#         model=_MODEL,
#         temperature=_TEMPERATURE,
#         stream=_STREAM,
#         callbacks=[callback],
#         verbose=False
#     )

# async def run_query_stream(question: str, pdf_dicts: list[dict]) -> AsyncGenerator[str, None]:
#     queue: asyncio.Queue[str] = asyncio.Queue()
    
#     try:
#         # 1) Build tools, agent, crew
#         tools = build_tools(pdf_dicts)
#         qa_agent = build_agent(tools, question)
#         qa_agent.llm = new_streaming_llm(QueueCallback(queue))
#         task = build_task(qa_agent, question)
#         crew = Crew(agents=[qa_agent], tasks=[task], process=Process.sequential)

#         # 2) Start crew execution in background
#         loop = asyncio.get_running_loop()
#         executor_task = loop.run_in_executor(None, crew.kickoff)

#         # 3) Stream tokens as they arrive
#         while True:
#             try:
#                 # Wait for tokens with timeout
#                 token = await asyncio.wait_for(queue.get(), timeout=2.0)
                
#                 if token == "[DONE]":
#                     break
#                 elif token.startswith("[ERROR]"):
#                     yield token
#                     break
#                 else:
#                     yield token
                    
#             except asyncio.TimeoutError:
#                 # Check if execution is complete
#                 if executor_task.done():
#                     # Process any remaining tokens
#                     while not queue.empty():
#                         try:
#                             token = queue.get_nowait()
#                             if token == "[DONE]":
#                                 return
#                             yield token
#                         except:
#                             break
#                     break
#                 continue
                
#     except Exception as e:
#         yield f"[ERROR]: {str(e)}"
#     finally:
#         # Cleanup
#         if 'executor_task' in locals() and not executor_task.done():
#             executor_task.cancel()






#version - 3

# import asyncio
# from typing import AsyncGenerator, List, Dict, Any
# from langchain.callbacks.base import AsyncCallbackHandler
# from crewai import LLM, Crew, Process
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Global LLM configuration
# _MODEL = "gpt-4o"
# _TEMPERATURE = 0.1
# _STREAM = True

# class QueueCallback(AsyncCallbackHandler):
#     """Forward every new LLM token into an asyncio.Queue with proper error handling."""
#     def __init__(self, queue: asyncio.Queue):
#         self.queue = queue
#         self._finished = False
#         self._error_occurred = False

#     async def on_llm_new_token(self, token: str, **kwargs):
#         """Handle new tokens from LLM"""
#         if not self._finished and not self._error_occurred:
#             try:
#                 await self.queue.put(token)
#                 logger.debug(f"Token queued: {token[:50]}...")
#             except Exception as e:
#                 logger.error(f"Error queuing token: {e}")
#                 self._error_occurred = True

#     async def on_chain_end(self, outputs, **kwargs):
#         """Handle chain completion"""
#         if not self._finished:
#             self._finished = True
#             await self.queue.put("[DONE]")
#             logger.info("Chain completed successfully")

#     async def on_chain_error(self, error, **kwargs):
#         """Handle chain errors"""
#         if not self._finished:
#             self._finished = True
#             self._error_occurred = True
#             error_msg = f"Chain error: {str(error)}"
#             logger.error(error_msg)
#             await self.queue.put(f"[ERROR]: {error_msg}")
#             await self.queue.put("[DONE]")

#     async def on_llm_error(self, error, **kwargs):
#         """Handle LLM errors"""
#         if not self._finished:
#             self._finished = True
#             self._error_occurred = True
#             error_msg = f"LLM error: {str(error)}"
#             logger.error(error_msg)
#             await self.queue.put(f"[ERROR]: {error_msg}")
#             await self.queue.put("[DONE]")

# def new_streaming_llm(callback: QueueCallback) -> LLM:
#     """Create a new streaming LLM with the callback attached"""
#     return LLM(
#         model=_MODEL,
#         temperature=_TEMPERATURE,
#         stream=_STREAM,
#         callbacks=[callback],
#         verbose=False
#     )

# async def run_query_stream(question: str, pdf_dicts: list[dict]) -> AsyncGenerator[str, None]:
#     """
#     Stream query results with proper async coordination and error handling
#     """
#     queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)  # Prevent memory issues
#     executor_task = None
    
#     try:
#         logger.info(f"Starting query stream for: {question[:100]}...")
        
#         # 1) Build tools, agent, crew
#         tools = build_tools(pdf_dicts)
#         qa_agent = build_agent(tools, question)
#         callback = QueueCallback(queue)
#         qa_agent.llm = new_streaming_llm(callback)
#         task = build_task(qa_agent, question)
#         crew = Crew(agents=[qa_agent], tasks=[task], process=Process.sequential)

#         # 2) Start crew execution in background thread
#         loop = asyncio.get_running_loop()
        
#         def run_crew():
#             """Wrapper function to run crew and handle exceptions"""
#             try:
#                 result = crew.kickoff()
#                 logger.info("Crew execution completed")
#                 return result
#             except Exception as e:
#                 logger.error(f"Crew execution failed: {e}")
#                 # Put error in queue synchronously since we're in a thread
#                 asyncio.run_coroutine_threadsafe(
#                     queue.put(f"[ERROR]: Crew execution failed: {str(e)}"), 
#                     loop
#                 )
#                 asyncio.run_coroutine_threadsafe(queue.put("[DONE]"), loop)
#                 raise

#         executor_task = loop.run_in_executor(None, run_crew)
        
#         # 3) Stream tokens as they arrive with timeout handling
#         consecutive_timeouts = 0
#         max_consecutive_timeouts = 10
        
#         while True:
#             try:
#                 # Wait for tokens with reasonable timeout
#                 token = await asyncio.wait_for(queue.get(), timeout=3.0)
#                 consecutive_timeouts = 0  # Reset timeout counter
                
#                 if token == "[DONE]":
#                     logger.info("Stream completed")
#                     break
#                 elif token.startswith("[ERROR]"):
#                     logger.error(f"Error token received: {token}")
#                     yield token[8:]  # Remove [ERROR]: prefix
#                     break
#                 else:
#                     yield token
                    
#             except asyncio.TimeoutError:
#                 consecutive_timeouts += 1
#                 logger.debug(f"Timeout {consecutive_timeouts}/{max_consecutive_timeouts}")
                
#                 # Check if executor task is done
#                 if executor_task.done():
#                     logger.info("Executor task completed, processing remaining tokens")
#                     # Process any remaining tokens
#                     remaining_tokens = 0
#                     while not queue.empty() and remaining_tokens < 50:  # Prevent infinite loop
#                         try:
#                             token = queue.get_nowait()
#                             remaining_tokens += 1
#                             if token == "[DONE]":
#                                 return
#                             elif not token.startswith("[ERROR]"):
#                                 yield token
#                         except asyncio.QueueEmpty:
#                             break
#                     break
                
#                 # If too many consecutive timeouts, assume something is wrong
#                 if consecutive_timeouts >= max_consecutive_timeouts:
#                     logger.warning("Too many consecutive timeouts, ending stream")
#                     yield "[ERROR]: Stream timeout - no response from AI"
#                     break
                    
#                 continue
                
#     except Exception as e:
#         error_msg = f"Stream error: {str(e)}"
#         logger.error(error_msg)
#         yield f"[ERROR]: {error_msg}"
#     finally:
#         # Cleanup
#         if executor_task and not executor_task.done():
#             logger.info("Cancelling executor task")
#             executor_task.cancel()
#             try:
#                 await executor_task
#             except asyncio.CancelledError:
#                 pass
#         logger.info("Stream cleanup completed")






# pdf_dicts= [
#     {
#       "pdf_file": "Your trip overview - Airbnb.pdf",
#       "description": "The document provides an overview of an Airbnb reservation for a stay at Roshni apartment, located at Final Plot 211, Dixit Road, Mumbai, Maharashtra, India. The reservation details include a check-in date of June 8, 2025, and a check-out date of June 15, 2025. The host, Jyothi, is described as a welcoming individual with a passion for traveling and hosting guests, ensuring that the accommodations are comfortable, clean, and well-equipped. Key logistical information for the stay includes a check-in method involving building staff and a maximum guest limit of two. Pets are not allowed in the apartment. Guests are encouraged to contact the host for specific check-in instructions. The document also provides a comprehensive list of support and resources available through Airbnb, such as help with safety issues, anti-discrimination measures, disability support, and options for cancellation. Financial details are outlined with a total payment of ₹24,801.05, and guests are reminded of the ability to obtain a receipt and a PDF for visa purposes. The document emphasizes the availability of 24/7 support globally for any assistance guests might need during their stay. The cancellation policy allows for a partial refund if canceled before the check-in time, with varying refund conditions thereafter. Additionally, the document mentions resources and support for Airbnb hosts, including community forums, hosting classes, and tools for hosting responsibly. It also highlights Airbnb's commitment to safety and discrimination-free environments through initiatives like AirCover and emergency stays provided by Airbnb.org.",
#       "vectorstore": "/tmp/faiss_indexYour trip overview - Airbnb.pdf"
#     },
#     {
#       "pdf_file": "Reservation Details - HM4C93MBSF.pdf",
#       "description": "The document is a booking confirmation for a 2BHK apartment located 10 minutes from the Mumbai (BOM) airport. The reservation details specify a check-in time of 1:00 PM on Saturday, June 7, and a checkout time of 11:00 AM on Saturday, June 14. The booking is for one guest, and the host of the apartment is Işıl İrem. The total amount paid for the stay is ₹18,743.83, and the confirmation code for this booking is HM4C93MBSF.",
#       "vectorstore": "/tmp/faiss_indexReservation Details - HM4C93MBSF.pdf"
#     },
#     {
#       "pdf_file": "invoice (1).pdf",
#       "description": "The document is a tax invoice, bill of supply, or cash memo issued by Amazon Seller Services Pvt. Ltd. for a purchase made on Amazon.in. It is specified as the original copy for the recipient, emphasizing that it is not a demand for payment. The invoice pertains to an order placed by Saketh on May 24, 2025, with the invoice generated on May 25, 2025. The order number is 403-4408380-0235562, and the invoice number is HYD8-979696. The purchased item is an Apple 2025 MacBook Air with specific features, including a 13-inch display, Apple M4 chip with 10-core CPU and GPU, 24GB unified memory, and 512GB storage, in a Sky Blue color. The unit price of the MacBook is ₹113,974.58, with a quantity of one unit ordered. The total net amount is ₹113,974.58, with a GST rate of 18% (divided into 9% CGST and 9% SGST) applied, amounting to ₹20,515.42. The total invoice value is ₹134,490.00, which is also expressed in words as 'One Hundred Thirty-four Thousand Four Hundred Ninety only.' The document specifies that the tax is not payable under reverse charge, and the payment was made via card, with a transaction ID provided. The seller is CLICKTECH RETAIL PRIVATE LIMITED, located in Hyderabad, Telangana, with their PAN and GST registration numbers included. The billing and shipping address for the buyer, Saketh, is in Potkapalli village, Peddapalli district, Karimnagar, Telangana. The place of supply and delivery is also noted as Telangana.",
#       "vectorstore": "/tmp/faiss_indexinvoice (1).pdf"
#     }
#   ]


pdf_files = [
        
  "Your trip overview - Airbnb.pdf",
  "Reservation Details - HM4C93MBSF.pdf",
  "invoice (1).pdf"



    ]
    
def read_pdf(pdf_path: str) -> str:
    """Read PDF file and extract text content"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
            
    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")

def create_chunks(text: str, pdf_file: str) -> List[Dict[str, Any]]:

    if not text:
        return []
    metadata = {"source": pdf_file}

# Create a Document object from the text
    doc = Document(page_content=text, metadata=metadata)

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      
        chunk_overlap=200,    
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  
    )


    chunks = text_splitter.split_documents([doc])
    print(chunks[0])


    return chunks



        
  
    
    return chunks

def store_embeddings_excel(chunked_documents,temp_dir='/tmp/doc1.pdf'):

    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])


    vectorstore = FAISS.from_documents(chunked_documents, embeddings)
    os.makedirs(temp_dir, exist_ok=True)
    vectorstore.save_local(temp_dir)
    return vectorstore

def generate_description(vectorstore):

    
    dummy_query = (
    "Give a high-level overview of the document including main topics, key arguments, and overall purpose."
)
    docs = vectorstore.similarity_search(dummy_query, k=25)

    
    summarization_prompt = PromptTemplate.from_template(
        """
        You are a helpful assistant that summarizes a document.
        Based on the following excerpts from a document, write a detailed summary that captures the main ideas, topics, and structure.

        Document Excerpts:
        {text}

        Summary:
        """
    )


    input_text = "\n\n".join([doc.page_content for doc in docs])



    llm = ChatOpenAI(model="gpt-4o", openai_api_key=st.secrets["OPENAI_API_KEY"])
    

    summarizer = LLMChain(llm=llm, prompt=summarization_prompt)

    response = summarizer.run({"text": input_text})
    
    return response



# pdf_dicts = []

# for pdf in pdf_files:
#     text = read_pdf(pdf)
#     chunks = create_chunks(text, pdf)
    
#     if not chunks:
#         print(f"No valid chunks created for {pdf}. Skipping.")
#         continue
    
#     vectorstore = store_embeddings_excel(chunks, temp_dir=f'/tmp/{pdf}')
#     summary = generate_description(vectorstore)
#     print(summary)

    
#     pdf_dicts.append({
#         "pdf_file": str(pdf),
#         "description":summary,
#         "vectorstore": f'/tmp/{pdf}'
#     })



# print(pdf_dicts)



pdf_dicts = [
    {
        
    'pdf_file': 'Your trip overview - Airbnb.pdf', 
    'description': 'The document is an overview of an Airbnb reservation for a stay in Mumbai, India, scheduled from June 8 to June 15, 2025. The accommodation is located at Roshni Apartment on Dixit Road and is hosted by Jyothi, who enjoys traveling and making guests feel welcome. The apartment offers a comfortable and fully equipped environment for travelers.\n\nGuests are advised to contact the host, Jyothi, for specific check-in details, as the apartment utilizes self check-in facilitated by building staff. House rules specify a maximum of two guests and prohibit pets. The reservation, which costs ₹24,801.05, includes a confirmation code (HMEENPP5EB) and offers options for cancellation, including a partial refund if canceled before check-in on June 8.\n\nAirbnb provides 24/7 global support, including resources on safety issues, anti-discrimination, and disability support. There are also facilities for hosts, such as AirCover, hosting resources, and community forums. Additional support includes options for reporting neighborhood concerns and hosting responsibly. The document also mentions a section for getting a PDF for visa purposes and printing reservation details.',
    'vectorstore': '/tmp/Your trip overview - Airbnb.pdf'}, 
    
    
    
    {'pdf_file': 'Reservation Details - HM4C93MBSF.pdf', 'description': 'The document provides details of a booking confirmation for a 2BHK apartment in Mumbai. The apartment is conveniently located just 10 minutes from the BOM (Chhatrapati Shivaji Maharaj International) airport. The check-in is scheduled for 1:00 PM on Saturday, June 7, and the check-out is set for 11:00 AM on Saturday, June 14. The reservation is for one guest, and it has been confirmed with the code HM4C93MBSF. The host of the apartment is Işıl İrem. The total amount paid for the stay amounts to ₹18,743.83. The document serves as an official confirmation of the booking, including essential details about the stay, location, and financial transaction.', 'vectorstore': '/tmp/Reservation Details - HM4C93MBSF.pdf'}, {'pdf_file': 'invoice (1).pdf', 'description': "The document is a tax invoice issued by Amazon Seller Services Pvt. Ltd. for a purchase made by Saketh through CLICKTECH RETAIL PRIVATE LIMITED. The invoice is associated with an order placed on May 24, 2025, and invoiced the following day. The purchase includes a 2025 Apple MacBook Air featuring a 13-inch screen, Apple M4 chip, 10-core CPU, 10-core GPU, 24GB Unified Memory, and 512GB storage, priced at ₹113,974.58. The total amount, including 9% Central Goods and Services Tax (CGST) and 9% State Goods and Services Tax (SGST), is ₹134,490.00. The invoice specifies that the tax is not payable under reverse charge and that payment was made via card. The seller's and buyer's details, including PAN and GST registration numbers, billing and shipping addresses, are provided. The transaction occurs within Telangana, as indicated by the place of supply and delivery. The invoice serves as a record of the transaction for input GST credit purposes, and it clarifies that it is not a demand for payment.", 'vectorstore': '/tmp/invoice (1).pdf'}]



# question = "where is my airbnb stay and  which company laptop i booked?"
# result = answer_query(question, pdf_dicts)
# print(result)


