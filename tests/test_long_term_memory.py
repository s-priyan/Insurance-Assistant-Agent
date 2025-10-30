import unittest
from unittest.mock import patch, mock_open, MagicMock
import sys

# Mock external dependencies
sys.modules['langchain_community'] = MagicMock()
sys.modules['langchain_community.vectorstores'] = MagicMock()
sys.modules['langchain_text_splitters'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def recursive_chunker(content: str):
    """Function under test"""
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=50,  
        separators=["\n\n", "\n", " ", ""], 
        keep_separator=False, 
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_spliter.create_documents([content])
    return chunks


def knowledge_base(
    file_path: str,
    persist_dir: str = None,
    embedding_model = None,
    vectorstore: str = None,
    **vectorstore_kwargs,
):
    """Function under test"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError as e:
        print(f"Error finding file {file_path}: {e}")
        raise

    chunks = recursive_chunker(content)

    if vectorstore == "FAISS":
        vector_db = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model,
            **vectorstore_kwargs,
        )
        vector_db.save_local(persist_dir)
    else:
        raise NotImplementedError(f"vector store type : {vectorstore} is not implemeted.")


class TestRecursiveChunker(unittest.TestCase):
    
    @patch('langchain_text_splitters.RecursiveCharacterTextSplitter')
    def test_chunker_creates_documents(self, mock_splitter_class):
        """Test that chunker creates documents from content"""
        # Setup mock
        mock_splitter = MagicMock()
        mock_chunks = [MagicMock(), MagicMock()]
        mock_splitter.create_documents.return_value = mock_chunks
        mock_splitter_class.return_value = mock_splitter
        
        # Test
        content = "This is a long text that needs to be chunked into smaller pieces."
        result = recursive_chunker(content)
        
        # Assertions
        mock_splitter_class.assert_called_once_with(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=False,
            length_function=len,
            is_separator_regex=False,
        )
        mock_splitter.create_documents.assert_called_once_with([content])
        self.assertEqual(result, mock_chunks)
    
    @patch('langchain_text_splitters.RecursiveCharacterTextSplitter')
    def test_chunker_with_empty_content(self, mock_splitter_class):
        """Test chunker with empty content"""
        mock_splitter = MagicMock()
        mock_splitter.create_documents.return_value = []
        mock_splitter_class.return_value = mock_splitter
        
        result = recursive_chunker("")
        
        mock_splitter.create_documents.assert_called_once_with([""])
        self.assertEqual(result, [])


class TestKnowledgeBase(unittest.TestCase):
    
    @patch('langchain_community.vectorstores.FAISS.from_documents')
    @patch('builtins.open', new_callable=mock_open, read_data="Test content for knowledge base")
    def test_knowledge_base_creation_success(self, mock_file, mock_faiss_from_docs):
        """Test successful knowledge base creation"""
        # Setup mocks
        mock_vector_db = MagicMock()
        mock_faiss_from_docs.return_value = mock_vector_db
        mock_embedding_model = MagicMock()
        
        with patch('test_long_term_memory.recursive_chunker') as mock_chunker:
            mock_chunks = [MagicMock(), MagicMock()]
            mock_chunker.return_value = mock_chunks
            
            # Test
            knowledge_base(
                file_path="test.md",
                persist_dir="./vector_store",
                embedding_model=mock_embedding_model,
                vectorstore="FAISS"
            )
            
            # Assertions
            mock_file.assert_called_once_with("test.md", 'r', encoding='utf-8')
            mock_chunker.assert_called_once_with("Test content for knowledge base")
            mock_faiss_from_docs.assert_called_once_with(
                documents=mock_chunks,
                embedding=mock_embedding_model
            )
            mock_vector_db.save_local.assert_called_once_with("./vector_store")
    
    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_knowledge_base_file_not_found(self, mock_file):
        """Test handling of missing file"""
        with self.assertRaises(FileNotFoundError):
            knowledge_base(
                file_path="nonexistent.md",
                persist_dir="./vector_store",
                embedding_model=MagicMock(),
                vectorstore="FAISS"
            )
    
    @patch('builtins.open', new_callable=mock_open, read_data="Test content")
    def test_knowledge_base_unsupported_vectorstore(self, mock_file):
        """Test handling of unsupported vector store type"""
        with patch('test_long_term_memory.recursive_chunker') as mock_chunker:
            mock_chunker.return_value = [MagicMock()]
            
            with self.assertRaises(NotImplementedError) as context:
                knowledge_base(
                    file_path="test.md",
                    persist_dir="./vector_store",
                    embedding_model=MagicMock(),
                    vectorstore="UNKNOWN"
                )
            
            self.assertIn("UNKNOWN", str(context.exception))
    
    @patch('langchain_community.vectorstores.FAISS.from_documents')
    @patch('builtins.open', new_callable=mock_open, read_data="Content")
    def test_knowledge_base_with_kwargs(self, mock_file, mock_faiss_from_docs):
        """Test knowledge base creation with additional kwargs"""
        mock_vector_db = MagicMock()
        mock_faiss_from_docs.return_value = mock_vector_db
        
        with patch('test_long_term_memory.recursive_chunker') as mock_chunker:
            mock_chunker.return_value = [MagicMock()]
            
            knowledge_base(
                file_path="test.md",
                persist_dir="./store",
                embedding_model=MagicMock(),
                vectorstore="FAISS",
                distance_strategy="cosine"
            )
            
            # Check that kwargs were passed
            call_kwargs = mock_faiss_from_docs.call_args[1]
            self.assertIn("distance_strategy", call_kwargs)


if __name__ == '__main__':
    unittest.main()