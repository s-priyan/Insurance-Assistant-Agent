from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
import gradio as gr
import os
from typing import Any

from langchain_community.vectorstores import FAISS


load_dotenv(override=True)


openai = init_chat_model(
    "gpt-4.1",
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding",  # Your Azure deployment name
    base_url="https://{your-resource-name}.openai.azure.com/openai/v1/",
    api_key="your-azure-api-key"
)

system_prompt = f"You are acting as intelligent chat bot reperesenting allianz car insurance company. You are answering questions about car insurance claims\
particularly questions related to insurance claims instruction, recommending different insurance claim according to the customer scenarios. \
Your responsibility is to represent allianze car insurance company and answer question only with given context as faithfully as possible. \
Be professional and engaging, as if talking to a potential customers \
If you don't know the answer, say so."

def content_reteieval(
        persist_dir: str,
        embedding_model: Any,
        query: str,
        top_k: int,
) -> list[str] | str:
    
    try:
        vector_db = FAISS.load_local(folder_path=persist_dir, 
                                     embeddings=embedding_model, 
                                     allow_dangerous_deserialization=True)

        results = vector_db.similarity_search(
            query= query, k = top_k
        )

        result_content = "".join([doc.page_content for doc in results])

        return result_content
    
    except Exception as e:
        raise RuntimeError(
            f"an error occured while retrieving document: {str(e)}"
        )



def chat(message, history):
    global system_prompt
    retrieve_contents = content_reteieval(persist_dir="../vectore/insurance", embedding_model=embedding_model, query=message, top_k=3)
    system_prompt += f"\n\n## Given Context:\n{retrieve_contents}\n\n"
    system_prompt += f"With this context, please chat with the user"
    # print(system_prompt)
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=deployment, messages=messages)
    return response.choices[0].message.content

# answer = chat(message="what types of car insurance claims are you providing?", history=[])
# print(answer)

gr.ChatInterface(chat, type="messages").launch()