from vllm import LLM, SamplingParams
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="", required=True, type=str)
args = parser.parse_args()
model_path = args.model_path

prompts = [
    "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手，请基于你的知识用中文回答当前问题。\n<</SYS>>\n\n这是本次的问题：李白是谁\n [/INST]",

    "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手，请基于你的知识用中文回答当前问题。\n<</SYS>>\n\n这是本次的问题：请简述人工智能的未来\n [/INST]",

    "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手，请基于你的知识用中文回答当前问题。\n<</SYS>>\n\n这是本次的问题：《雪中悍刀行的作者是谁》\n [/INST]",

    "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手，请基于你的知识用中文回答当前问题。\n<</SYS>>\n\n这是本次的问题：天空为什么是蓝色的\n [/INST]",
]

# sampling_params = SamplingParams(temperature=0.8, top_k=40, top_p=0.95)
sampling_params = SamplingParams(
            frequency_penalty=1.2,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            max_tokens=1024
    )
llm = LLM(model=model_path, trust_remote_code=True)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")