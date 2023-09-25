from langchain.llms import VLLMOpenAI, OpenAIChat
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--port', default="10874", required=False, type=str)
parser.add_argument('--host', default="localhost", required=False, type=str)
# 模型部署别名
parser.add_argument('--model_name', default="alpaca2", required=False, type=str)
args = parser.parse_args()
port = args.port
host = args.host
model_name = args.model_name

prompt_template = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. 你是一个乐于助人的助手。请基于你的知识以及聊天历史记录回答当前问题\n"
    "<</SYS>>\n\n"
    "这是聊天历史记录：{chat_history}\n"
    "这是当前问题：{input}\n[/INST]"
)

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://{}:{}/v1".format(host, port),
    model_name=model_name,
    max_tokens=512,
    top_p=0.9,
    temperature=0.2,
    frequency_penalty=1.2,
    model_kwargs={
        "top_k": 40,
        "stop": [".</s>"]
    }
)

prompt = PromptTemplate.from_template(prompt_template)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)

qa = ConversationChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

while True:
    a = input("请输入问题：")
    print(qa.predict(input=a))