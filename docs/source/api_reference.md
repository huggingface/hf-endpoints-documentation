# API Reference (Swagger)

Inference Endpoints can be used through the [UI](https://endpoints.huggingface.co/endpoints) and programmatically through an API.
Here you'll find the [open-API specification](https://api.endpoints.huggingface.cloud/) for each available route, which you can call directly,
or through the [Hugging Face Hub python client](https://huggingface.co/docs/huggingface_hub/guides/inference_endpoints).

<iframe src="https://api.endpoints.huggingface.cloud/"  style='height: 60vh; width: 100%;' frameborder="0" id="iframe">Browser not compatible.</iframe>

## Catalog API

Separately from the general API above, a public **Catalog API** is available under `/api/v1` on
`https://endpoints.huggingface.co`, scoped to the [Model Catalog](./quick_start): listing catalog
items and deploying a catalog model or a specific recipe. Listing is unauthenticated; the deploy
routes require an `Authorization: Bearer <token>` header with your Hugging Face access token.

You can browse and try every route interactively at
[`/api/docs`](https://endpoints.huggingface.co/api/docs), or fetch the raw specification from
[`/api/openapi.json`](https://endpoints.huggingface.co/api/openapi.json).

