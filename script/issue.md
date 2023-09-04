运行 bash run_sft.sh
[INFO|trainer.py:1686] 2023-09-03 15:59:38,794 >> ***** Running training *****
[INFO|trainer.py:1687] 2023-09-03 15:59:38,794 >>   Num examples = 5,619
[INFO|trainer.py:1688] 2023-09-03 15:59:38,794 >>   Num Epochs = 1
[INFO|trainer.py:1689] 2023-09-03 15:59:38,794 >>   Instantaneous batch size per device = 1
[INFO|trainer.py:1692] 2023-09-03 15:59:38,794 >>   Total train batch size (w. parallel, distributed & accumulation) = 8
[INFO|trainer.py:1693] 2023-09-03 15:59:38,794 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:1694] 2023-09-03 15:59:38,794 >>   Total optimization steps = 702
[INFO|trainer.py:1695] 2023-09-03 15:59:38,798 >>   Number of trainable parameters = 472,973,312
  0%|                                                                          | 0/702 [00:00<?, ?it/s]Traceback (most recent call last):
  File "run_clm_sft_with_peft.py", line 445, in <module>
    main()
  File "run_clm_sft_with_peft.py", line 418, in main
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/transformers/trainer.py", line 1539, in train
    return inner_training_loop(
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/transformers/trainer.py", line 1809, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/transformers/trainer.py", line 2654, in training_step
    loss = self.compute_loss(model, inputs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/transformers/trainer.py", line 2679, in compute_loss
    outputs = model(**inputs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/deepspeed/utils/nvtx.py", line 15, in wrapped_fn
    ret_val = func(*args, **kwargs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/deepspeed/runtime/engine.py", line 1801, in forward
    loss = self.module(*inputs, **kwargs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/peft/peft_model.py", line 529, in forward
    return self.base_model(
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py", line 806, in forward
    outputs = self.model(
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py", line 693, in forward
    layer_outputs = decoder_layer(
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py", line 408, in forward
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py", line 305, in forward
    query_states = self.q_proj(hidden_states)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/peft/tuners/lora.py", line 356, in forward
    result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`
  0%|                                                                          | 0/702 [00:00<?, ?it/s]
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 2421) of binary: /root/Chinese-LLaMA-Alpaca-2/llama/bin/python
Traceback (most recent call last):
  File "/root/Chinese-LLaMA-Alpaca-2/llama/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 346, in wrapper
    return f(*args, **kwargs)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/torch/distributed/run.py", line 794, in main
    run(args)
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/root/Chinese-LLaMA-Alpaca-2/llama/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 250, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
run_clm_sft_with_peft.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2023-09-03_15:59:44
  host      : autodl-container-58c811aa3c-319653a4
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 2421)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================