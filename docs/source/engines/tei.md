# Text Embeddings Inference (TEI)

Text Embeddings Inference (TEI) is a robust, production-ready engine designed for fast and efficient generation of text
embeddings from a wide range of models. Built for scalability and reliability, TEI streamlines the deployment
of embedding models for search, retrieval, clustering, and semantic understanding tasks.

Key Features:
- **Efficient Resource Utilization**: Benefit from small Docker images and rapid boot times.
- **Dynamic Batching**: TEI incorporates token-based dynamic batching thus optimizing resource utilization during inference.
- **Optimized Inference**: TEI leverages Flash Attention, Candle, and cuBLASLt by using optimized transformers code for inference.
- **Support for models** in both the Safetensors and ONNX format
- **Production-Ready**: TEI supports distributed tracing through Open Telemetry and exports Prometheus metrics.

## Configuration

![config](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/tei/tei.png)

- **Max Tokens (per batch)**: Number of tokens that can be added to a batch before forcing queries to wait in the internal queue. 
- **Max Concurrent Requests**: The maximum number of requests that the server can handle at once.
- **Pooling**: Setting to override the model pooling configuration. Default is not to override the model configuration.

## Supported models

You can find the models that are supported by TGI by either:
- Browse supported models on the [Hugging Face Hub](https://huggingface.co/models?other=text-embeddings-inference&sort=trending)
- In the TEI documentation under the [supported models](https://huggingface.co/docs/text-embeddings-inference/supported_models) section

## References

We also recommend reading the [TEI documentation](https://huggingface.co/docs/text-embeddings-inference/index) for more in-depth information.
