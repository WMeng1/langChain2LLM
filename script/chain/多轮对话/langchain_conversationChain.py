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
from langchain.chains import ConversationChain
from langchain.prompts.chat import PromptTemplate
from langchain.memory import ConversationBufferMemory


prompt_template = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. 你是一个乐于助人的助手。请基于你的知识以及聊天历史记录回答当前问题\n"
    "<</SYS>>\n\n"
    "这是聊天历史记录：{chat_history}\n"
    "这是当前问题：{input}\n[/INST]"
)

if __name__ == '__main__':
    load_type = torch.float16
    print("loading LLM...")
    model = HuggingFacePipeline.from_model_id(model_id=model_path,
                                              task="text-generation",
                                              device=0,
                                              model_kwargs={
                                                  "torch_dtype": load_type,
                                                  "low_cpu_mem_usage": True,
                                                  "temperature": 1,
                                                  "max_length": 4096,
                                                  "repetition_penalty": 1.1}
                                              )
    prompt = PromptTemplate.from_template(prompt_template)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)
    qa = ConversationChain(llm=model, prompt=prompt, memory=memory, verbose=True)
    while True:
        a = input("请输入问题：")
        print(qa.predict(input=a))
