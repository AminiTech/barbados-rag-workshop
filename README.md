# How to run the RAG Powered Copilot demo

1. Install the required python packages

``` 
pip install -r requirements.txt
```

2. Run Qdrant vector database using docker

```
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```


3. Encode your data and store them in the Qdrant vector database as embeddings

- Put all your documents/text files in the copilot_knowledge_graph folder
- Run the copilot_vector_db.py script to encode the documents and store them in the Qdrant vector database

```
python copilot_vector_db.py
```
4. Update the .env file with the Qdrant vector database url exposed by the docker container running locally on you machine, huggingface API token, and your OpenAI key

```
OPENAI_API_KEY= ADD_API_KEY_HERE
QDRANT_URL=http://localhost:6333
HUGGINGFACEHUB_API_TOKEN= ADD_TOKEN_HERE
```
5. Run the copilot_chat_gradio.py script to generate the response to the user query on a gradio interface

``` 
python copilot_chat_gradio.py
``` 

## References
1. https://python.langchain.com/docs/get_started/introduction
2. https://qdrant.tech/documentation/
3. https://huggingface.co/spaces/mteb/leaderboard




