# vLLM

vLLM is a high-performance, memory-efficient inference engine for open-source LLMs. It delivers efficient scheduling, KV-cache handling,
batching, and decoding—all wrapped in a production-ready server. For most use cases, TGI, vLLM, and SGLang will be equivalently good options.

**Core features**:
- **PagedAttention for memory efficiency**
- **Continuous batching**
- **Optimized CUDA/HIP execution**
- **Speculative decoding & chunked prefill**
- **Multi-backend and hardware support**: Runs across NVIDIA, AMD, and AWS Neuron to name a few

## Configuration

![config](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/vllm/vllm_config.png)

- **Max Number of Sequences**: The maximum number of sequences (requests) that can be processed together in a single batch. Controls
the batch size by sequence count, affecting throughput and memory usage. For example, if max_num_seqs=8, up to 8 different prompts can
be handled at once, regardless of their individual lengths, as long as the total token count also fits within the Max Number of Batched Tokens.
- **Max Number of Batched Tokens**: The maximum total number of tokens (summed across all sequences) that can be processed in a single
batch. Limits batch size by token count, balancing throughput and GPU memory allocation.
- **Tensor Parallel Size**: The number of GPUs across which model weights are split within each layer. Increasing this allows larger
models to run and frees up GPU memory for KV cache, but may introduce synchronization overhead.
- **KV Cache DType**: the data type used for storing the key-value cache during generation. Options include "auto", "fp8", "fp8_e5m2",
and "fp8_e4m3". Using lower precision types can reduce memory usage but may slightly impact generation quality.

For more advanced configuration you can pass any of the [Engine Arguments that vLLM supports](https://docs.vllm.ai/en/stable/api/vllm/engine/arg_utils.html#vllm.engine.arg_utils.EngineArgs)
as container arguments. For example changing the `enable_lora` to `true` would look like this:

![vllm-advanced](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/vllm/vllm-advanced.png)

## Supported models

vLLM has wide support for large language models and embedding models. We recommend reading the
[supported models](https://docs.vllm.ai/en/stable/models/supported_models.html?h=supported+models) section in the vLLM documentation for a full list.

vLLM also supports model implementations that are available in Transformers. Currently not all models work but support is planned for most
decoder language models are supported, and vision language models.

## Parallelism and Scaling

vLLM supports several parallelism strategies for distributed inference. The two most common ones are **Tensor Parallelism (TP)** and **Data Parallelism (DP)**. Understanding when and how to use each is essential for optimal performance.

### Default Behavior on Inference Endpoints

When you create an endpoint, after you've selected an instance type (e.g., 4× A10G, 8× H100). The defaults are:

- **`tensor_parallel_size`** = number of GPUs on the instance (shards the model across all GPUs)
- **`data_parallel_size`** = 1 (single copy of the model)

This default configuration prioritizes fitting larger models by using all available GPU memory. However, you might want to tweak these settings if:
- Your model fits on fewer GPUs than your instance has, and you want **higher throughput** by running multiple copies of the model
- You're running a large model that barely fits and want to **reduce TP** to leave more GPU memory for KV cache

### Tensor Parallelism (TP)

Tensor parallelism splits the model's weights across multiple GPUs within each layer. Each GPU holds a slice of the model and computes its portion of the output, then synchronizes with other GPUs.

**When to use:** Your model is too large to fit on a single GPU.

| Model | GPU Memory Required | Single GPU (80GB) | Minimum TP |
|-------|---------------------|-------------------|------------|
| Llama 3 8B (FP16) | ~16GB | ✅ Fits | 1 |
| Llama 3 70B (FP16) | ~140GB | ❌ Too large | 2 |
| Llama 3.1 405B (FP16) | ~810GB | ❌ Too large | 8 |

### Data Parallelism (DP)

Data parallelism runs multiple independent copies of the model on different GPUs. Each copy handles different requests independently, increasing throughput.

**When to use:** You want higher throughput and your model fits on fewer GPUs than your instance provides.

**Configuration:** Set `data_parallel_size` to the number of copies you want.

### Combining TP and DP

On multi-GPU instances, you can combine both strategies. The key formula:

```
tensor_parallel_size × data_parallel_size = total GPUs on instance
```

#### Optimizing for Throughput

If your model fits on a single GPU but you need high throughput, lower TP and increase DP to run multiple copies of the model.

**Example:** Serving Llama 3 8B (~16GB) on a 4× A100 80GB instance:

| Configuration | TP | DP | Copies | Behavior |
|--------------|----|----|--------|----------|
| Default | 4 | 1 | 1 | Model sharded across all 4 GPUs |
| Optimized | 1 | 4 | 4 | 4 independent copies, 4× throughput potential |

With `tensor_parallel_size=1` and `data_parallel_size=4`, each GPU runs its own full copy of the model, letting you handle 4× more concurrent requests.

#### More Examples

**8× H100 80GB:**

| Model | Memory | TP | DP | Result |
|-------|--------|----|----|--------|
| Mistral 7B (~14GB) | Fits on 1 GPU | 1 | 8 | 8 independent copies, maximum throughput |
| Llama 3 70B (~140GB) | Needs 2 GPUs | 2 | 4 | 4 copies, each sharded across 2 GPUs |
| Llama 3.1 405B (~810GB) | Needs 8 GPUs | 8 | 1 | 1 copy using all GPUs |

**4× A10G (24GB each):**

| Model | Memory | TP | DP | Result |
|-------|--------|----|----|--------|
| Mistral 7B (~14GB) | Fits on 1 GPU | 1 | 4 | 4 copies |
| Llama 3 8B (~16GB) | Fits on 1 GPU | 1 | 4 | 4 copies |
| Qwen 2.5 32B (~64GB) | Needs 4 GPUs | 4 | 1 | 1 copy sharded across all |

### Choosing the Right Configuration

1. **Calculate minimum TP:** How many GPUs are needed to fit your model in memory?
2. **Set TP to that minimum**
3. **Set DP** = (total instance GPUs) ÷ TP

**Example:** You want to deploy Llama 3 70B on 8× H100 80GB.
- Model needs ~140GB → minimum 2× 80GB GPUs → `tensor_parallel_size=2`
- Instance has 8 GPUs → `data_parallel_size=8÷2=4`
- Result: 4 copies, each on 2 GPUs

### Common Mistakes

| Configuration | Problem | Solution |
|---------------|---------|----------|
| Default TP=4, DP=1 for 7B model on 4× A10G | Only 1 copy, missing out on throughput | Set `tensor_parallel_size=1`, `data_parallel_size=4` for 4 copies |
| TP=1, DP=1 for 7B on 4× A10G | 3 GPUs sitting idle | Increase `data_parallel_size=4` |
| TP=1 for 70B on single 80GB GPU | Out of memory | Use a multi-GPU instance and increase `tensor_parallel_size=2` |

## References

We also recommend reading the [vLLM documentation](https://docs.vllm.ai/en/stable/) for more in-depth information.