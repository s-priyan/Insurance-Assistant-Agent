import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock external dependencies
sys.modules['gradio'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain_community'] = MagicMock()
sys.modules['langchain_community.vectorstores'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

from langchain_community.vectorstores import FAISS
from typing import Any


def content_reteieval(
    persist_dir: str,
    embedding_model: Any,
    query: str,
    top_k: int,
) -> str:
    """Function under test"""
    try:
        vector_db = FAISS.load_local(
            folder_path=persist_dir,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        
        results = vector_db.similarity_search(query=query, k=top_k)
        result_content = "".join([doc.page_content for doc in results])
        
        return result_content
    
    except Exception as e:
        raise RuntimeError(
            f"an error occured while retrieving document: {str(e)}"
        )


def chat(message, history, openai_client, deployment, system_prompt_base):
    """Function under test - modified to accept dependencies"""
    retrieve_contents = content_reteieval(
        persist_dir="../vectore/insurance",
        embedding_model=MagicMock(),
        query=message,
        top_k=3
    )
    
    system_prompt = system_prompt_base + f"\n\n## Given Context:\n{retrieve_contents}\n\n"
    system_prompt += f"With this context, please chat with the user"
    
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    response = openai_client.chat.completions.create(model=deployment, messages=messages)
    
    return response.choices[0].message.content


class TestContentRetrieval(unittest.TestCase):
    
    @patch('langchain_community.vectorstores.FAISS.load_local')
    def test_content_retrieval_success(self, mock_load_local):
        """Test successful content retrieval from vector database"""
        # Setup mock documents
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Insurance claim process: "
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Contact us at claims@allianz.com"
        
        # Setup mock vector DB
        mock_vector_db = MagicMock()
        mock_vector_db.similarity_search.return_value = [mock_doc1, mock_doc2]
        mock_load_local.return_value = mock_vector_db
        
        mock_embedding_model = MagicMock()
        
        # Test
        result = content_reteieval(
            persist_dir="./vector_store",
            embedding_model=mock_embedding_model,
            query="How to file a claim?",
            top_k=2
        )
        
        # Assertions
        mock_load_local.assert_called_once_with(
            folder_path="./vector_store",
            embeddings=mock_embedding_model,
            allow_dangerous_deserialization=True
        )
        mock_vector_db.similarity_search.assert_called_once_with(
            query="How to file a claim?",
            k=2
        )
        self.assertEqual(result, "Insurance claim process: Contact us at claims@allianz.com")
    
    @patch('langchain_community.vectorstores.FAISS.load_local')
    def test_content_retrieval_empty_results(self, mock_load_local):
        """Test content retrieval with no matching documents"""
        mock_vector_db = MagicMock()
        mock_vector_db.similarity_search.return_value = []
        mock_load_local.return_value = mock_vector_db
        
        result = content_reteieval(
            persist_dir="./vector_store",
            embedding_model=MagicMock(),
            query="Unknown query",
            top_k=3
        )
        
        self.assertEqual(result, "")
    
    @patch('langchain_community.vectorstores.FAISS.load_local')
    def test_content_retrieval_with_exception(self, mock_load_local):
        """Test handling of exceptions during retrieval"""
        mock_load_local.side_effect = Exception("Database connection failed")
        
        with self.assertRaises(RuntimeError) as context:
            content_reteieval(
                persist_dir="./vector_store",
                embedding_model=MagicMock(),
                query="test query",
                top_k=3
            )
        
        self.assertIn("an error occured while retrieving document", str(context.exception))
        self.assertIn("Database connection failed", str(context.exception))
    
    @patch('langchain_community.vectorstores.FAISS.load_local')
    def test_content_retrieval_multiple_docs(self, mock_load_local):
        """Test retrieval with multiple documents"""
        # Create multiple mock documents
        mock_docs = []
        for i in range(5):
            doc = MagicMock()
            doc.page_content = f"Content {i}. "
            mock_docs.append(doc)
        
        mock_vector_db = MagicMock()
        mock_vector_db.similarity_search.return_value = mock_docs
        mock_load_local.return_value = mock_vector_db
        
        result = content_reteieval(
            persist_dir="./vector_store",
            embedding_model=MagicMock(),
            query="test",
            top_k=5
        )
        
        expected = "Content 0. Content 1. Content 2. Content 3. Content 4. "
        self.assertEqual(result, expected)


class TestChat(unittest.TestCase):
    
    @patch('test_workflow_ui.content_reteieval')
    def test_chat_basic_response(self, mock_retrieval):
        """Test basic chat response generation"""
        # Setup mocks
        mock_retrieval.return_value = "Insurance claims are processed within 48 hours."
        
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "We process claims within 48 hours."
        mock_openai.chat.completions.create.return_value = mock_response
        
        system_prompt_base = "You are an insurance assistant."
        
        # Test
        result = chat(
            message="How long does claim processing take?",
            history=[],
            openai_client=mock_openai,
            deployment="gpt-4",
            system_prompt_base=system_prompt_base
        )
        
        # Assertions
        mock_retrieval.assert_called_once()
        self.assertEqual(result, "We process claims within 48 hours.")
        
        # Check that OpenAI was called with correct structure
        call_args = mock_openai.chat.completions.create.call_args
        self.assertEqual(call_args[1]['model'], "gpt-4")
        messages = call_args[1]['messages']
        self.assertEqual(messages[0]['role'], "system")
        self.assertEqual(messages[-1]['role'], "user")
        self.assertEqual(messages[-1]['content'], "How long does claim processing take?")
    
    @patch('test_workflow_ui.content_reteieval')
    def test_chat_with_history(self, mock_retrieval):
        """Test chat with conversation history"""
        mock_retrieval.return_value = "Context information"
        
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Follow-up answer"
        mock_openai.chat.completions.create.return_value = mock_response
        
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"}
        ]
        
        result = chat(
            message="Follow-up question",
            history=history,
            openai_client=mock_openai,
            deployment="gpt-4",
            system_prompt_base="You are an assistant."
        )
        
        # Check that history was included
        call_args = mock_openai.chat.completions.create.call_args
        messages = call_args[1]['messages']
        self.assertGreater(len(messages), 2)  # System + history + current message
    
    @patch('test_workflow_ui.content_reteieval')
    def test_chat_includes_context_in_prompt(self, mock_retrieval):
        """Test that retrieved context is included in system prompt"""
        context_content = "Important insurance policy information"
        mock_retrieval.return_value = context_content
        
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai.chat.completions.create.return_value = mock_response
        
        chat(
            message="Test question",
            history=[],
            openai_client=mock_openai,
            deployment="gpt-4",
            system_prompt_base="Base prompt"
        )
        
        # Check that context was added to system prompt
        call_args = mock_openai.chat.completions.create.call_args
        system_message = call_args[1]['messages'][0]['content']
        self.assertIn("Given Context:", system_message)
        self.assertIn(context_content, system_message)


if __name__ == '__main__':
    unittest.main()