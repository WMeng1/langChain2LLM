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
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory.buffer import ConversationBufferMemory
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os

prompt_template = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. 你是一个乐于助人的助手。请基于你的知识以及聊天历史记录用中文回答当前问题\n"
    "<</SYS>>\n\n"
    "这是聊天历史记录：{chat_history}\n"
    "这是本次问题：{question}\n [/INST]"
)

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
    # 若传入文件是txt文件，则读取该文件
    if file_path.endswith('.txt'):
        with open(file_path) as f:
            text = f.read()
            textList.append(text)
    # 否则，读取文件夹下所有txt
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, length_function=len)
    for root, dir, files in os.walk(file_path):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                with open(file_path) as f:
                    text = f.read()
                    textList.append(text)
    documents = text_splitter.create_documents(textList)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print("Loading the embedding model...")
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
    model = HuggingFacePipeline.from_model_id(model_id=model_path,
                                              task="text-generation",
                                              device=0,
                                              model_kwargs={
                                                  "torch_dtype": load_type,
                                                  "low_cpu_mem_usage": True,
                                                  "temperature": 0.2,
                                                  "max_length": 2048,
                                                  "repetition_penalty": 1.1}
                                              )

    if args.chain_type == "stuff":
        PROMPT = PromptTemplate.from_template(prompt_template)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)
        combine_docs_chain_kwargs = {"prompt": PROMPT}
        qa = ConversationalRetrievalChain.from_llm(
            llm=model,
            chain_type='stuff',
            # retriever=docsearch.as_retriever(
            #     search_type="similarity_score_threshold",
            #     search_kwargs={'score_threshold': 0.95}
            # ),
            memory=memory,
            retriever=docSearch.as_retriever(search_kwargs={"k": 1}),
            # combine_docs_chain_kwargs=combine_docs_chain_kwargs,
            condense_question_prompt=PROMPT
        )
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
        # 返回文档信息
        memory = ConversationBufferMemory(memory_key="chat_history", input_key='question', output_key='answer',
                                          return_messages=True, k=3)
        # 不返回
        # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)
        qa = ConversationalRetrievalChain.from_llm(
            llm=model,
            chain_type='refine',
            memory=memory,
            retriever=docSearch.as_retriever(search_kwargs={"k": 1}),
            # condense_question_prompt=refine_prompt
        )
    while True:
        query = input("请输入问题：")
        if len(query.strip()) == 0:
            break
        # print(qa.predict(input=query))
        #
        res = qa({'question':query})
        print(res)
