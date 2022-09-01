# Hugging Face Inference Endpoints documentation

## Setup

```bash
pip install hf-doc-builder==0.4.0 watchdog --upgrade
```

## Local Development

```bash
doc-builder preview endpoints source/ --not_python_module
```

## Build Docs

```bash
doc-builder build endpoints source/ --build_dir build/ --not_python_module
```

## Add assets/Images

Adding images/assets is only possible through `https://` links meaning you need to use `https://github.com/huggingface/hf-endpoints-documentation/blob/main/assets/` prefix.

example

```bash
<img src="https://github.com/huggingface/hf-endpoints-documentation/blob/main/assets/0_login.png" alt="Login" />
```