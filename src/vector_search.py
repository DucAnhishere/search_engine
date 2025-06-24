from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType

def create_embedding(model_name, all_chunks, query):
    model = SentenceTransformer(model_name)
    embeddings1 = model.encode(all_chunks)
    embeddings2 = model.encode(query)
    return embeddings1, embeddings2
    
def store_data(collection_name, embeddings, embeddings_list, all_chunks, resume_paths, resume_ids):
    client = MilvusClient(
        uri="http://localhost:19530",
        token="root:Milvus"
    )

    # Kiểm tra và tạo collection nếu chưa có
    if not client.has_collection(collection_name):
        print(f"📁 Collection '{collection_name}' not found. Creating...")

        # Tạo schema theo yêu cầu
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=5000)
        schema.add_field(field_name="resume_paths", datatype=DataType.VARCHAR, max_length=2000)
        schema.add_field(field_name="resume_ids", datatype=DataType.VARCHAR, max_length=500)

        # Tạo collection
        client.create_collection(
            collection_name=collection_name,
            schema=schema
        )

        # Tạo index cho trường vector
        client.create_index(
            collection_name=collection_name,
            field_name="vector",
            index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"}
        )

    # Chuẩn bị dữ liệu
    data = [
        {
            "vector": embeddings_list[i],
            "text": all_chunks[i],
            "resume_paths": resume_paths[i],
            "resume_ids": resume_ids[i]
        }
        for i in range(len(embeddings))
    ]

    # Insert dữ liệu
    client.insert(collection_name=collection_name, data=data)
    print(f"✅ Inserted {len(data)} entries into '{collection_name}'")


def search_resumes(embeddings, collection_name, k):
    """Search for similar resumes"""
    from pymilvus import MilvusClient
    
    client = MilvusClient(host="localhost", port="19530")
    
    # Check if collection exists
    if not client.has_collection(collection_name):
        raise Exception(f"Collection '{collection_name}' does not exist. Please create it first by indexing some documents.")
    
    # Load collection if not already loaded
    try:
        client.load_collection(collection_name)
    except Exception as e:
        print(f"Collection might already be loaded: {e}")

    # Perform the search
    res = client.search(
        collection_name=collection_name,
        anns_field="vector",
        data=embeddings,
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

def create_collection_if_not_exists(collection_name, dimension):
    """Create Milvus collection if it doesn't exist"""
    from pymilvus import MilvusClient, DataType
    
    client = MilvusClient(host="localhost", port="19530")
    
    # Check if collection already exists
    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        # Load it if it exists but not loaded
        try:
            client.load_collection(collection_name)
            print(f"Collection '{collection_name}' loaded.")
        except:
            pass
        return
    
    # Define collection schema
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    
    # Add fields
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="resume_paths", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="resume_ids", datatype=DataType.VARCHAR, max_length=100)
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )
    
    # Create index for vector field
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 1024}
    )
    
    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
    
    print(f"Collection '{collection_name}' created successfully!")
    
    # Auto-load the collection after creation
    client.load_collection(collection_name)
    print(f"Collection '{collection_name}' loaded into memory!")

def load_collection(collection_name):
    """Load collection into memory for searching"""
    from pymilvus import MilvusClient
    
    client = MilvusClient(host="localhost", port="19530")
    
    # Check if collection exists
    if not client.has_collection(collection_name):
        raise Exception(f"Collection '{collection_name}' does not exist.")
    
    # Load collection
    client.load_collection(collection_name)
    print(f"Collection '{collection_name}' loaded successfully!")

