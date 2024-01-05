from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

host = "localhost"
port = "19530"

connections.connect("default", host=host, port=port)
Collection