'''
    vllm原生离线推理脚本
'''

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List
import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default="", required=True, type=str)
parser.add_argument('--embedding_path', default="", required=True, type=str)
parser.add_argument('--model_path', default="", required=True, type=str)
parser.add_argument('--gpu_id', default="0", type=str)
parser.add_argument('--chain_type', default="refine", type=str)
args = parser.parse_args()
model_path = args.model_path


def build_chat_input(tokenizer,
                     messages: List[dict],
                     max_new_tokens: int = 2048,
                     model_max_length: int = 4096,
                     user_token_id: int = 195,
                     assistant_token_id: int = 196):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens
    max_input_tokens = model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(user_token_id)
            else:
                round_tokens.append(assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(
                round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return input_tokens


# Sample prompts.
prompts = [
    [{
        "role": "user",
        "content": "李白是谁？"
    }],
    [{
        "role": "user",
        "content": "请简述助学贷款是什么"
    }],
    [{
        "role": "user",
        "content": "雪中悍刀行的作者是谁？"
    }],
    [{
        "role": "user",
        "content": "AI的未来是什么，请在100字内简述"
    }],
]

# Taken from generation_config.
sampling_params = SamplingParams(temperature=0.2,
                                 top_k=5,
                                 top_p=0.9,
                                 max_tokens=1024,
                                 stop=[".</s>"],
                                 frequency_penalty=1.2
                                 )

# Create an LLM.
llm = LLM(model=model_path,
          trust_remote_code=True,
          tensor_parallel_size=1,
          tokenizer_mode='slow')
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          use_fast=False,
                                          trust_remote_code=True)
# bypass chat format issue with explicit tokenization
prompt_token_ids = [
    build_chat_input(tokenizer, prompt, max_new_tokens=1024)
    for prompt in prompts
]
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompt_token_ids=prompt_token_ids,
                       sampling_params=sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r},\n Generated text: {generated_text!r}")