import os
import argparse
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

def load_documents_from_folder(folder_path):
    """
    Load documents from a folder.

    Args:
        folder_path (str): Path to the folder containing .txt files.

    Returns:
        list: List of dictionaries representing the documents.
    """
    documents = []
    for idx, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                description = file.read()
                document = {
                    "id": idx,
                    "description": description
                }
                documents.append(document)
    return documents

def upload_to_qdrant(folder_path, collection_name, host, model):
    """
    Upload documents to a Qdrant collection.

    Args:
        folder_path (str): Path to the folder containing .txt files.
        collection_name (str): Name of the Qdrant collection.
        host (str): Qdrant host in the format "host:port".
        model (str): Name of the model to use for encoding.
    """
    encoder = SentenceTransformer(model)

    host, port = host.split(':')
    qdrant = QdrantClient(host, port=int(port))

    documents = load_documents_from_folder(folder_path)

    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )

    qdrant.upload_records(
        collection_name=collection_name,
        records=[
            models.Record(
                id=doc["id"], vector=encoder.encode(doc["description"]).tolist(), payload=doc
            )
            for doc in documents
        ],
    )

def main():
    """
    Main function to load documents and upload to Qdrant collection.
    """
    parser = argparse.ArgumentParser(description="Load documents and upload to Qdrant collection")
    parser.add_argument("--folder_path", default='copilot_knowledge_graph', help="Path to the folder containing .txt files")
    parser.add_argument("--collection_name", default='recomendations_knowledge_graph', help="Name of the Qdrant collection")
    parser.add_argument("--host", default="localhost:6333", help="Qdrant host")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Name of the model to use for encoding")

    args = parser.parse_args()

    upload_to_qdrant(args.folder_path, args.collection_name, args.host, args.model)

if __name__ == "__main__":
    main()