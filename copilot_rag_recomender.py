import os
import json
import pandas as pd
from dotenv import load_dotenv
from ydata_profiling import ProfileReport
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
#qdrant = QdrantClient(":memory:")

# initialize the encoder and llm
encoder = SentenceTransformer("all-MiniLM-L6-v2")
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")

class GenerateRecommendation:
    def __init__(self):
        self.recommendation_template = """
        query: {query}
        farm data: {raw_data}
        recommendation context: {recommendation_context}
        
        From the above data and context, generate some farm recommendations based on the query provided. Be precise and accurate.
        """
    def similarity_search(self, query):
        # Search for similar documents
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

if __name__ == "__main__":
    # Load the input data 
    input_data = {
        'query': "What is the recomended depth for planting maize?",
        'raw_data': {
        "farm_id": "005edc63-fb63-4b0c-b7ec-e785143fc869",
        "farm_centroid_point": "POINT (27.688995344653094 -12.529628915828152)",
        "farm_centroid_lat": "27.688995344653094",
        "farm_centroid_lon": "-12.529628915828152",
        "country_name": "Zambia",
        "admin_one_name": "Chingola",
        "admin_two_name": "Ipafu",
        "farm_location": "{\"type\": \"MultiPolygon\", \"coordinates\": [[[[27.6890303, -12.5294266], [27.6892607, -12.5297085], [27.6890175, -12.5298295], [27.6887007, -12.5295733], [27.6890303, -12.5294266]]]]}",
        "farm_stats": [
            {
                "CCCI": 0.012982628962892928,
                "CI": 0.18606249253621868,
                "NDMI": -0.020972227614392238,
                "NDVI": 0.4630636348056877,
                "area_geometry": 0.1372,
                "area_under_production": 0.13,
                "avg_humidity": 95.13,
                "avg_rain_accumulation": 0.04,
                "avg_temperature": 25.95,
                "crop": "Maize",
                "crop_health": "Good",
                "season": "previous_2017-2018",
                "stage": "late_season_stage",
                "suspected_nitrogen_content_deficiency": "Near normal Nitrogen Content",
                "timestamp": 1527064040060,
                "water_stress": "High Water Stress"
            },
            {
                "CCCI": 0.010484740593810346,
                "CI": 0.049090591608647066,
                "NDMI": 0.11178111145982256,
                "NDVI": 0.25472997696055666,
                "area_geometry": 0.1372,
                "area_under_production": 0.13,
                "avg_humidity": 99.79,
                "avg_rain_accumulation": 2.57,
                "avg_temperature": 25.91,
                "crop": "Maize",
                "crop_health": "Average",
                "season": "previous_2017-2018",
                "stage": "mid_season_stage",
                "suspected_nitrogen_content_deficiency": "Near normal Nitrogen Content",
                "timestamp": 1520583839170,
                "water_stress": "Normal"
            }],
        "soil_properties": [
            {
                "Property": "aluminium_extractable",
                "Value": 120.5,
                "Unit": "ppm"
            },
            {
                "Property": "bulk_density",
                "Value": 1.32,
                "Unit": "g/cc"
            },
            {
                "Property": "calcium_extractable",
                "Value": 402.4,
                "Unit": "ppm"
            },
            {
                "Property": "carbon_organic",
                "Value": 6.4,
                "Unit": "g/kg"
            },
            {
                "Property": "carbon_total",
                "Value": 17.2,
                "Unit": "g/kg"
            },
            {
                "Property": "clay_content",
                "Value": 19.0,
                "Unit": "%"
            },
            {
                "Property": "silt_content",
                "Value": 21.0,
                "Unit": "%"
            },
            {
                "Property": "sand_content",
                "Value": 61.0,
                "Unit": "%"
            },
            {
                "Property": "stone_content",
                "Value": 3.5,
                "Unit": "%"
            },
            {
                "Property": "iron_extractable",
                "Value": 80.5,
                "Unit": "ppm"
            },
            {
                "Property": "nitrogen_total",
                "Value": 1.2,
                "Unit": "g/kg"
            },
            {
                "Property": "phosphorous_extractable",
                "Value": 7.2,
                "Unit": "ppm"
            },
            {
                "Property": "potassium_extractable",
                "Value": 65.7,
                "Unit": "ppm"
            },
            {
                "Property": "ph",
                "Value": 5.9,
                "Unit": ""
            }]
    }
    }
    # Initialize the recommendation generator
    recommendation_generator = GenerateRecommendation()

    # Get Recommendations
    recommendation_dict = recommendation_generator.get_recommendations(input_data)
    print(recommendation_dict)
    print("Done!")