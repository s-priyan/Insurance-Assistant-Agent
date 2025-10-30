from typing import Any
import os

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv(override=True)


embedding_model = OpenAIEmbeddings(
    model="text-embedding",  # Your Azure deployment name
    base_url="https://{your-resource-name}.openai.azure.com/openai/v1/",
    api_key="your-azure-api-key"
)


def recursive_chunker(content: str):

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=50,  
        separators=["\n\n", "\n", " ", ""], 
        keep_separator=False, 
        length_function = len,
        is_separator_regex = False,
    )

    chunks = text_spliter.create_documents([content])
    
    return chunks



def knowledge_base(
    file_path : str,
    persist_dir : str = None,
    embedding_model : Any = None,
    vectorstore : str = None,
    **vectorstore_kwargs,
) -> None | str:
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

    except FileNotFoundError as e:
        print(f"Error finding file {file_path}: {e}")

    chunks = recursive_chunker(content)

    if vectorstore == "FAISS":

        vector_db = FAISS.from_documents(
            documents= chunks,
            embedding = embedding_model,
            **vectorstore_kwargs,
        )
        
        vector_db.save_local(persist_dir)

    else:
        raise NotImplementedError(f"vector store type : {vectorstore} is not implemeted.")
    

knowledge_base(file_path="../data/content.md", persist_dir="../vectore/insurance", embedding_model=embedding_model, vectorstore="FAISS")