from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

def create_embedding(model_name, all_chunks, query):
    model = SentenceTransformer(model_name)
    embeddings1 = model.encode(all_chunks)
    embeddings2 = model.encode(query)
    return embeddings1, embeddings2
    
def store_data(collection_name, embeddings, embeddings_list, all_chunks, resume_paths, resume_ids):

    client = MilvusClient(
        uri="http://192.168.80.25:19530",
        token="root:Milvus"
    )

    # Prepare data for insertion
    data = [
        {
            "vector": embeddings_list[i],
            "text": all_chunks[i],
            "resume_paths": resume_paths[i],
            "resume_ids": resume_ids[i]
        }
        for i in range(len(embeddings))
    ]

    # Insert data into the collection
    client.insert(collection_name=collection_name, data=data)

def search_resumes(query_embeddings, collection_name, k):

    client = MilvusClient(
        uri="http://192.168.80.25:19530",
        token="root:Milvus"
    )

    # Perform the search
    res = client.search(
        collection_name=collection_name,
        anns_field="vector",
        data=query_embeddings,
        limit=k,
        search_params={"metric_type": "COSINE"},
        output_fields=["text", "resume_paths", "resume_ids"]
    )

    # Process the results
    top_k_distance = []
    top_k_chunks = []
    resume_ids = []
    resume_paths = []

    for hits in res: 
        for hit in hits:  
            top_k_distance.append(hit["distance"]) 
            top_k_chunks.append(hit["entity"]["text"])
            resume_ids.append(hit["entity"]["resume_ids"])
            resume_paths.append(hit["entity"]["resume_paths"])

    return top_k_distance, top_k_chunks, resume_ids, resume_paths

