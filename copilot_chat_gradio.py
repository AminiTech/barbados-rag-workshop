import os
import json
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import gradio as gr

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# Initialize global variables
qdrant_url = os.getenv("QDRANT_URL")
qdrant = QdrantClient(qdrant_url)

# initialize the encoder and llm
encoder = SentenceTransformer("all-MiniLM-L6-v2")
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")

class KnowledgeGraphChat:
    def __init__(self):
        self.answer_template = """
        Query: {query}
        Relevant data: {relevant_data}
        
        Based on the above data, provide a detailed answer to the query.
        """
    
    def search_knowledge_graph(self, query):
        # Perform a search in the knowledge graph
        hits = qdrant.search(
            collection_name="recomendations_knowledge_graph",
            query_vector=encoder.encode(query).tolist(),
            limit=1,
        )
        relevant_data = "No relevant data found."
        for hit in hits:
            print(f"Found relevant data: {hit.payload}, score: {hit.score}")
            relevant_data = hit.payload
            break  # Assuming only one hit is processed for simplicity

        return {'relevant_data': relevant_data}

    def generate_answer(self, query):
        relevant_data = self.search_knowledge_graph(query)
        prompt = PromptTemplate(template=self.answer_template, input_variables=["query", "relevant_data"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        answer = llm_chain.run({'query': query, **relevant_data})
        return answer

# Initialize the chat system
kg_chat = KnowledgeGraphChat()

def chat_interface(query):
    """
    Interface function for the Gradio app.

    Args:
        query (str): The user query.

    Returns:
        str: The generated answer.
    """
    response = kg_chat.generate_answer(query)

    return response

# Gradio interface
iface = gr.Interface(fn=chat_interface,
                     inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
                     outputs="text",
                     title="Knowledge Graph Chat Interface",
                     description="Ask any question and get answers based on the knowledge graph.")

if __name__ == "__main__":
    iface.launch()
