# Hugging Face Inference Endpoints documentation

## Setup

```bash
pip install hf-doc-builder==0.4.0 watchdog --upgrade
```

## Local Development

```bash
doc-builder preview endpoints docs/source/ --not_python_module
```

## Build Docs

```bash
doc-builder build endpoints docs/source/ --build_dir build/ --not_python_module
```

## Add assets/Images

Adding images/assets is only possible through `https://` links meaning you need to use `https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/` prefix.

example

```bash
<img src="https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/0_login.png" alt="Login" />
```

## Generate API Reference

1. Copy openapi spec from `https://api.endpoints.huggingface.cloud/api-doc/openapi.json`
2. create markdown `widdershins --environment env.json openapi.json -o myOutput.md`
3. copy into `api_reference.mdx`

