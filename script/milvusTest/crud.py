import jieba
import numpy as np
import os
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
from utils import *
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

if model_type in ['WoBERT', 'RoFormer']:
    tokenizer = get_tokenizer(
        dict_path, pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
    )
else:
    tokenizer = get_tokenizer(dict_path)

# 建立模型
if model_type == 'RoFormer':
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model='roformer',
        pooling=pooling,
        dropout_rate=dropout_rate
    )
elif 'NEZHA' in model_type:
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model='nezha',
        pooling=pooling,
        dropout_rate=dropout_rate
    )
else:
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        pooling=pooling,
        dropout_rate=dropout_rate
    )


if __name__ == '__main__':
    encoder.load_weights('best_model.weights')
    # 可以使用
    # encoder.load_weights('abc.weights')
    a, b, c = '四级螺纹钢筋', '三级螺纹钢', '三级螺纹钢筋'
    tmp = '四级螺纹钢'
    d_token_ids = [tokenizer.encode(tmp, maxlen=128)[0]]
    vector = encoder.predict([d_token_ids,
                            np.zeros_like(d_token_ids)],
                           verbose=True)

    a_token_ids, b_token_ids = [tokenizer.encode(a, maxlen=128)[0]], [tokenizer.encode(b, maxlen=128)[0]]
    c_token_ids = [tokenizer.encode(c, maxlen=128)[0]]

    vec1 = encoder.predict([a_token_ids,
                            np.zeros_like(a_token_ids)],
                           verbose=True)
    vec2 = encoder.predict([b_token_ids,
                            np.zeros_like(b_token_ids)],
                           verbose=True)
    vec3 = encoder.predict([c_token_ids,
                            np.zeros_like(c_token_ids)],
                           verbose=True)

    db = "default"
    host = "localhost"
    port = "19530"
    connections.connect(db, host=host, port=port)
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]

    # 新建数据库表
    # schema = CollectionSchema(fields, "wm_test")
    # conn = Collection("wm_test", schema)

    # 已有数据库表
    conn = Collection("material")
    # 插入
    vec = []
    vec.append(vec1.tolist()[0])
    vec.append(vec2.tolist()[0])
    vec.append(vec3.tolist()[0])
    data = [[vec], [a, b, c]]
    conn.insert(data)
    conn.flush()
    # 查询
    conn = Collection("material")
    conn.load()
    # result = conn.query(expr="id==446785399793955564", output_fields=["id", "embedding"])
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 128},
    }
    result = conn.search(vec1, "embedding", search_params, limit=1, expr="id > 0", output_fields=["id", "name"])

    # 查询
    conn = Collection("test1")
    conn.load()
    result = conn.query(expr="id==446785399793955564", output_fields=["id", "embedding"])
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 128},
    }
    result = conn.search(vec1, "embedding", search_params, limit=1, expr="id > 0", output_fields=["id", "name"])