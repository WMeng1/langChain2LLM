'''
langchain自带的vllm测试
    alpaca2可以推理
    百川输出乱码，github上有同样的issue，说baichuan2-13B-chat没问题，7B-chat有问题
    有人说是因为两个模型训练所使用的类有差异导致的
'''

import argparse
import torch
from langchain.chains import ConversationChain
from langchain.llms import VLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import PromptTemplate

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="", required=True, type=str)
args = parser.parse_args()
model_path = args.model_path

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
    model = VLLM(model=model_path,
                 trust_remote_code=True,  # mandatory for hf models
                 max_new_tokens=128,
                 top_k=40,
                 top_p=0.9,
                 temperature=0.2,
                 frequency_penalty=1.2,
                 stop=[".</s>"]
    )
    # model = HuggingFacePipeline.from_model_id(model_id=model_path,
    #                                           task="text-generation",
    #                                           device=0,
    #                                           model_kwargs={
    #                                               "torch_dtype": load_type,
    #                                               "low_cpu_mem_usage": True,
    #                                               "temperature": 1,
    #                                               "max_length": 4096,
    #                                               "repetition_penalty": 1.1}
    #                                           )
    prompt = PromptTemplate.from_template(prompt_template)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)
    qa = ConversationChain(llm=model, prompt=prompt, memory=memory, verbose=True)
    while True:
        a = input("请输入问题：")
        print(qa.predict(input=a))
