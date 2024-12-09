from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from common import comlogger

logger = comlogger.get_shared_logger()

class MilvusClient:
    def __init__(self, host, port):
        self.collection_name = "face_embeddings"
        connections.connect("default", host=host, port=port)
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        if not utility.has_collection(self.collection_name):
            logger.info(f"Creating collection: {self.collection_name}")

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
                FieldSchema(name="face_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=255)
            ]
            
            schema = CollectionSchema(fields=fields, description="Face embeddings collection")
            collection = Collection(name=self.collection_name, schema=schema)
            logger.info(f"Collection created: {collection}")
            
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()
            logger.info(f"Collection loaded: {collection}")