import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', required=True, type=str)
parser.add_argument('--embedding_path', required=True, type=str)
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--gpu_id', default="0", type=str)
parser.add_argument('--chain_type', default="refine", type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
file_path = args.file_path
embedding_path = args.embedding_path
model_path = args.model_path

import torch
from langchain import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

prompt_template = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. 你是一个乐于助人的助手。\n"
    "<</SYS>>\n\n"
    "{context}\n{question} [/INST]"
)

refine_prompt_template = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. 你是一个乐于助人的助手。\n"
    "<</SYS>>\n\n"
    "这是原始问题: {question}\n"
    "已有的回答: {existing_answer}\n"
    "现在还有一些文字，（如果有需要）你可以根据它们完善现有的回答。"
    "\n\n"
    "{context_str}\n"
    "\n\n"
    "请根据新的文段，进一步完善你的回答。"
    " [/INST]"
)

initial_qa_template = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. 你是一个乐于助人的助手。\n"
    "<</SYS>>\n\n"
    "以下为背景知识：\n"
    "{context_str}"
    "\n"
    "请根据以上背景知识, 回答这个问题：{question}。"
    " [/INST]"
)


def getDocSearch():
    textList = []
    for root, dir, files in os.walk(file_path):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                with open(path) as f:
                    text = f.read()
                    textList.append(text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100)
    documents = text_splitter.create_documents(textList)
    texts = text_splitter.split_documents(documents)

    print("Loading the embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_path)
    docsearch = FAISS.from_documents(texts, embeddings)
    return docsearch


if __name__ == '__main__':
    load_type = torch.float16
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs are available.")
    # 单文档
    # loader = TextLoader(file_path)
    # 多文档
    loader = DirectoryLoader(file_path, glob='*.txt', show_progress=True, loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print("Loading the embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_path)
    docsearch = FAISS.from_documents(texts, embeddings)
    # 多文档
    # docsearch = getDocSearch()

    print("loading LLM...")
    model = HuggingFacePipeline.from_model_id(model_id=model_path,
                                              task="text-generation",
                                              device=0,
                                              model_kwargs={
                                                  "torch_dtype": load_type,
                                                  "low_cpu_mem_usage": True,
                                                  "temperature": 0.2,
                                                  "repetition_penalty": 1.1,
                                                  "max_length": 1000}
                                              )

    if args.chain_type == "stuff":
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True)

    elif args.chain_type == "refine":
        refine_prompt = PromptTemplate(
            input_variables=["question", "existing_answer", "context_str"],
            template=refine_prompt_template,
        )
        initial_qa_prompt = PromptTemplate(
            input_variables=["context_str", "question"],
            template=initial_qa_template,
        )
        chain_type_kwargs = {"question_prompt": initial_qa_prompt, "refine_prompt": refine_prompt}
        qa = RetrievalQA.from_chain_type(
            llm=model, chain_type="refine",
            retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

    while True:
        query = input("请输入问题：")
        if len(query.strip()) == 0:
            break
        result = qa({"query": query})
        # print(qa.run(query))
        print(result)
