# 在线部署，通用api
python -m vllm.entrypoints.api_server --model=/root/autodl-tmp/workspace/models/lora-merged-model/ --trust-remote-code --host=127.0.0.1 --port=10874

# 模型调用
curl http://localhost:10874/generate -d \
'{
  "prompt": "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手，请基于你的知识用中文回答当前问题。\n<</SYS>>\n\n这是本次的问题：请简述人工智能的未来\n [/INST]",
  "n": 1,
  "top_p": 0.9,
  "top_k": 40,
  "temperature": 0.2,
  "frequency_penalty": 1.2,
  "max_tokens": 1024
}'

# 在线部署，openAI格式的api
python -m vllm.entrypoints.openai.api_server --model=/root/autodl-tmp/workspace/models/lora-merged-model/ --served-model-name=alpaca2 --trust-remote-code --host=127.0.0.1 --port=10874

curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d \
'{
  "prompt": "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手，请基于你的知识用中文回答当前问题。\n<</SYS>>\n\n这是本次的问题：请简述人工智能的未来\n [/INST]",
  "n": 1,
  "top_p": 0.9,
  "top_k": 40,
  "temperature": 0.2,
  "frequency_penalty": 1.2,
  "max_tokens": 1024
}'

