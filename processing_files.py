import pandas as pd
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import tiktoken
from io import BytesIO
import fitz  # PyMuPDF
import concurrent.futures
import logging
from llama_parse import LlamaParse
import unicodedata
import uuid
import faiss
encoder = tiktoken.get_encoding("cl100k_base")
import os
import openai
import streamlit as st


openai.api_key = st.secrets["OPENAI_API_KEY"]

def convert_pdf_to_text_new(pdf_path):
    """
    Converts PDF to text using PyMuPDF (fitz) with parallel processing.
    Returns list of tuples containing (text, page_number) if successful, or None if conversion fails.
    """
    try:
        print(pdf_path)
        doc = fitz.open(pdf_path)  
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda p: process_page_ocr_2(p, pdf_path), doc))
        return results
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")
        return None

def pdf_split_text_with_token_limit_new(text, max_tokens=8000, max_chars=49000):
    """
    Splits text into chunks that respect both token and character limits.
    Args:
        text: Input text to split
        max_tokens: Maximum number of tokens per chunk
        max_chars: Maximum characters per chunk (AWS Bedrock limit)
    """
    # First check character length
    chunks = []
    current_chunk = ""
    
    # Split by paragraphs first
    paragraphs = text.split('\n')
    print("done !!")
    
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += '\n' + para if current_chunk else para
            
    if current_chunk:
        chunks.append(current_chunk)
        
    # Further split by tokens if needed
    token_chunks = []
    for chunk in chunks:
        tokens = encoder.encode(chunk)
        if len(tokens) > max_tokens:
            # Split into smaller chunks based on tokens
            sub_chunks = []
            current_tokens = []
            
            for token in tokens:
                if len(current_tokens) + 1 > max_tokens:
                    sub_chunks.append(encoder.decode(current_tokens))
                    current_tokens = [token]
                else:
                    current_tokens.append(token)
                    
            if current_tokens:
                sub_chunks.append(encoder.decode(current_tokens))
            token_chunks.extend(sub_chunks)
        else:
            token_chunks.append(chunk)
            
    return token_chunks

def get_chunked_documents_pipeline(page_texts,temp_file_path,  max_tokens=8000):
    chunked_documents = []
    if page_texts:
        def process_page(page_data):
            text,page_number = page_data
            chunks = pdf_split_text_with_token_limit_new(text, max_tokens=max_tokens)
            metadata={
                "source": temp_file_path,
                "page_number": page_number
            }
            return [
                Document(
                    page_content=chunk,
                    metadata = metadata
                    
                    
                )
                for chunk in chunks
            ]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_page, page_texts))
            for result in results:
                chunked_documents.extend(result)
    
    else:
        logging.error(f"Skipping unreadable PDF: {temp_file_path}")
    return chunked_documents
  
def process_pdf_new(document_content, s3_key, max_tokens):
    """
    Processes the PDF content by splitting it into chunks and adding metadata with parallel processing.
    Args:
        document_content: PDF binary content
        s3_key: S3 key for storage reference
        max_tokens: Maximum tokens per chunk
    """
    temp_file_path = f'/tmp/{time.time()}_{uuid.uuid4()}_temp_document.pdf'
    
    try:
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(document_content)
        
        page_texts = convert_pdf_to_text_new(temp_file_path)
       
        # page_texts = process_pdf_ocr(temp_file_path)
        print(f"length of page_texts = {len(page_texts)}")
        print("Conversion Done!!")
        
        chunked_documents = get_chunked_documents_pipeline(page_texts, temp_file_path, s3_key, max_tokens)
        print(chunked_documents)
        print(f"length of the chunked = {len(chunked_documents)}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        chunked_documents = []
    
    finally:
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Removed temporary file: {temp_file_path}")
        except Exception as cleanup_error:
             print(f"Warning: Could not remove temporary file {temp_file_path}: {cleanup_error}")
    

        # if os.path.exists(temp_file_path):
        #     os.remove(temp_file_path)
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Removed temporary file: {temp_file_path}")
        except Exception as cleanup_error:
             print(f"Warning: Could not remove temporary file {temp_file_path}: {cleanup_error}")
    

        # if os.path.exists(temp_file_path):
        #     os.remove(temp_file_path)
    
    return chunked_documents

def decode_text_gpt(encoded_text):
    
    print(f"Decoding text: {encoded_text}")
    
    print("inside the function call")

  
    messages = [
        {
            "role": "system",
            "content": """You are a text encoding specialist. Your task is to:
                1. Identify encoded characters in the text
                2. Convert them to standard readable characters
                3. Preserve any valid formatting or structure
                4. convert the given text into the standard format
          """
        },
        {
            "role": "user",
            "content": f"convert this text to standard format: {encoded_text}.  ONLY return the converted text without any explanations.preserve the structure accurately and return the response in markdown format"
        }
    ]
    
    try:
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2
        )
        
        print(response.choices[0].message.content.strip())
        
        
        return response.choices[0].message.content.strip()
        
        
    except Exception as e:
        print(f"Error decoding text: {e}")
        return encoded_text  # Return original text on error

def process_page_ocr_2(page, pdf_file):
    try:
        text = page.get_text()
        print("step-1 done")
        # print(text)
        
        
        print("done")
        
        
        
        should_use_llama = False
        
        
        if not text.strip():
            should_use_llama = True
            print(f"Empty text detected on page {page.number} in {pdf_file}, using LlamaParse")
        
       
        elif any(ord(c) > 127 for c in text) or '\ufffd' in text or '�' in text:
            decoded_text = decode_text_gpt(text)
            print(decoded_text)
            
            print(f"Encoded characters detected on page {page.number} in {pdf_file}, using LlamaParse")
            return (decoded_text, page.number + 1)
        
       
        elif len(text.strip()) > 0 and len([c for c in text if c.isalnum()]) / len(text) < 0.5:
            decoded_text = decode_text_gpt(text)
            print(decoded_text)
           
            print(f"Low readability detected on page {page.number} in {pdf_file}, using LlamaParse")
            return (decoded_text, page.number + 1)
        
        if should_use_llama:
            print(f"Parsing page {page.number} in {pdf_file} with LlamaParse")
            api_key = 'llx-el6V5Ubnd6YAx1BLGo1ak4vG03wDNAHmQyjoB1l8X6lS3dNp'
            
            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown",  
                analyze_layout=True,
                verbose=True,
                language="en",
                target_pages=f"{page.number}",
               
                ocr_mode="advanced",  
                encoding_cleanup=True,  
                normalize_text=True, 
                image_processing="high_resolution",
                extra_languages=["el", "ru", "ar", "zh", "ja"],
                user_prompt="Convert the given text into the standard readable format"
                
            )
            
            documents = parser.load_data(pdf_file)
            if documents and len(documents) > 0:
                raw_text = documents[0].get_content()
               
                normalized_text = unicodedata.normalize('NFKD', raw_text)
               
                cleaned_text = ''.join(c for c in normalized_text if unicodedata.category(c)[0] != 'C')
                print(f"Successfully parsed page {page.number} with LlamaParse")
                return (cleaned_text, page.number + 1)
            else:
                print(f"LlamaParse returned no content for page {page.number}")
                return ("", page.number + 1)
        
      
        normalized_text = unicodedata.normalize('NFKD', text)
        cleaned_text = ''.join(c for c in normalized_text if unicodedata.category(c)[0] != 'C')
        print(cleaned_text)
        return (cleaned_text, page.number + 1)
    
    except Exception as e:
        print(f"Error processing page {page.number} in {pdf_file}: {e}")
       
        try:
            print(f"Attempting fallback to LlamaParse for page {page.number}")
            api_key = 'llx-el6V5Ubnd6YAx1BLGo1ak4vG03wDNAHmQyjoB1l8X6lS3dNp'
            
            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown",
                analyze_layout=True,
                verbose=True,
                language="en",
                target_pages=f"{page.number}",
                ocr_mode="advanced",
                encoding_cleanup=True,
                normalize_text=True,
                image_processing="high_resolution",
                extra_languages=["el", "ru", "ar", "zh", "ja"],
                user_prompt="Convert the given text into the standard readable format"
               
            )
            
            documents = parser.load_data(pdf_file)
            if documents and len(documents) > 0:
                raw_text = documents[0].get_content()
                
                normalized_text = unicodedata.normalize('NFKD', raw_text)
                cleaned_text = ''.join(c for c in normalized_text if unicodedata.category(c)[0] != 'C')
                print(cleaned_text)
                print(f"Successfully parsed page {page.number} with LlamaParse fallback")
                return (cleaned_text, page.number + 1)
            else:
                print(f"LlamaParse fallback returned no content for page {page.number}")
        except Exception as fallback_e:
            print(f"Fallback to LlamaParse also failed for page {page.number}: {fallback_e}")
        
        return ("", page.number + 1)
