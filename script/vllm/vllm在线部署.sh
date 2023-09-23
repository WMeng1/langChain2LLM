# 在线部署
python -m vllm.entrypoints.api_server --model=/root/autodl-tmp/workspace/models/lora-merged-model/ --trust-remote-code --host=127.0.0.1 --port=10874

# 模型调用
curl http://localhost:10874/generate \
> -d '{
> "prompt": "李白是谁",
> "use_beam_search": true,
> "n": 4,
> "temperature": 0
> }'