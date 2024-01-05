import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default="", required=True, type=str)
parser.add_argument('--embedding_path', default="", required=True, type=str)
parser.add_argument('--model_path', default="", required=True, type=str)
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
from langchain.llms import VLLM
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os, pdb

prompt_template = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. 你是一个乐于助人的助手。请基于你的知识以及聊天历史记录用中文回答当前问题\n"
    "<</SYS>>\n\n"
    "这是聊天历史记录：{chat_history}\n"
    "这是本次问题：{question}\n [/INST]"
)

# prompt_template = (
#     "[INST] <<SYS>>\n"
#     "You are a helpful assistant. 你是一个乐于助人的助手。请基于你的知识以及聊天历史记录用中文回答当前问题\n"
#     "<</SYS>>\n\n"
#     "这是聊天历史记录：{chat_history}\n"
#     "{context}\n{question} [/INST]"
# )


# prompt_template = (
#     "[INST] <<SYS>>\n"
#     "You are a helpful assistant. 你是一个乐于助人的助手。请基于你的知识以及聊天历史记录回答当前问题\n"
#     "<</SYS>>\n\n"
#     "这是当前问题：{question}\n[/INST]"
# )

refine_prompt_template = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. 你是一个乐于助人的助手。\n"
    "<</SYS>>\n\n"
    "这是原始问题: {question}\n"
    "这是聊天历史记录：{chat_history}\n"
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
    "You are a helpful assistant. 你是一个乐于助人的助手。请基于你的知识以及聊天历史记录用中文回答当前问题\n"
    "<</SYS>>\n\n"
    "这是聊天历史记录：{chat_history}\n"
    "以下为背景知识：\n"
    "{context_str}"
    "\n"
    "请根据以上背景知识, 回答这个问题：{question}。"
    " [/INST]"
)


def getDocSearch():
    # 单文件
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        print("Loading the embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_path)
        docsearch = FAISS.from_documents(texts, embeddings)
    # 多文件
    else:
        loader = DirectoryLoader(file_path, glob="*.txt", loader_cls=TextLoader, show_progress=True)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name=embedding_path)
        docsearch = FAISS.from_documents(texts, embeddings)
    return docsearch


if __name__ == '__main__':
    load_type = torch.float16
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs are available.")

    print("loading doc...")
    docSearch = getDocSearch()
    print("loading LLM...")
    model = VLLM(model="mosaicml/mpt-7b",
                 trust_remote_code=True,  # mandatory for hf models
                 max_new_tokens=128,
                 top_k=40,
                 top_p=0.9,
                 temperature=0.8,
    )
    model = HuggingFacePipeline.from_model_id(model_id=model_path,
                                              task="text-generation",
                                              device=0,
                                              # model_kwargs={
                                              #     "torch_dtype": load_type,
                                              #     "low_cpu_mem_usage": True,
                                              #     "temperature": 1,
                                              #     "max_length": 4096,
                                              #     "repetition_penalty": 1.1}
                                              # )
                                              pipeline_kwargs={
                                                "max_new_tokens": 2048,
                                                "do_sample": True,
                                                "temperature": 0.2,
                                                "top_k": 40,
                                                "top_p": 0.9,
                                                "repetition_penalty": 1.1},
                                              model_kwargs={
                                                    "torch_dtype": load_type,
                                                    "low_cpu_mem_usage": True}
                                             )

    if args.chain_type == "stuff":
        PROMPT = PromptTemplate.from_template(prompt_template)
        memory = ConversationBufferMemory(memory_key="chat_history", input_key='question', output_key='answer', return_messages=True, k=3)
        combine_docs_chain_kwargs = {"prompt": PROMPT}
        qa = ConversationalRetrievalChain.from_llm(
            llm=model,
            chain_type='stuff',
            memory=memory,
            retriever=docSearch.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True,
            condense_question_prompt=PROMPT
            # combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        )
    elif args.chain_type == "refine":
        refine_prompt = PromptTemplate.from_template(refine_prompt_template)
        initial_qa_prompt = PromptTemplate.from_template(initial_qa_template)
        combine_docs_chain_kwargs = {"question_prompt": initial_qa_prompt, "refine_prompt": refine_prompt}
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer', k=3)
        qa = ConversationalRetrievalChain.from_llm(
            llm=model,
            chain_type='refine',
            memory=memory,
            retriever=docSearch.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True,
            combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        )
    while True:
        query = input("请输入问题：")
        if len(query.strip()) == 0:
            break
        # print(qa.predict(input=query))
        #
        res = qa({'question':query})
        print(res)
        print(res['answer'])
VLLM()