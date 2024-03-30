import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# Initialize global variables
qdrant_url = os.getenv("QDRANT_URL")
qdrant = QdrantClient(qdrant_url)
encoder = SentenceTransformer("all-MiniLM-L6-v2")
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")

app = FastAPI()

class InputData(BaseModel):
    query: str
    raw_data: dict

class GenerateRecommendation:
    def __init__(self):
        self.recommendation_template = """
        query: {query}
        farm data: {raw_data}
        recommendation context: {recommendation_context}
        
        From the above data and context, generate some farm recommendations based on the query provided. Be precise and accurate.
        """

    def similarity_search(self, query):
        hits = qdrant.search(
            collection_name="recomendations_knowledge_graph",
            query_vector=encoder.encode(query).tolist(),
            limit=1,
        )
        for hit in hits:
            print(hit.payload, "score:", hit.score)
        return {'recommendation_context': hit.payload}

    def get_recommendations(self, input_data):
        input_data['recommendation_context'] = self.similarity_search(input_data['query'])
        prompt = PromptTemplate(template=self.recommendation_template, input_variables=["query", "raw_data", "recommendation_context"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        recommendation_result = llm_chain.run(input_data)
        return {'recommendation': recommendation_result}

@app.post("/recommendations/")
async def generate_recommendation(input_data: InputData):
    recommendation_generator = GenerateRecommendation()
    try:
        recommendation_dict = recommendation_generator.get_recommendations(input_data.dict())
        return recommendation_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))