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

The homepage hero illustration is generated from `illustration.svelte` (source of truth for geometry/animation). Engine logos are vendored under `assets/logos/` (copied from the Inference Endpoints site) so the illustration does not depend on external URLs.

After editing geometry or animation, regenerate committed SVGs:

```bash
node scripts/generate-illustration.mjs
```

To refresh logos from production (if icons change upstream):

```bash
curl -sL -o assets/logos/FILE.svg https://endpoints.huggingface.co/logos/FILE.svg
```

This writes animated SVGs plus `-static` variants (frozen route segments, no SMIL). The docs page layers both and hides the animated `<object>` when SMIL is unavailable (e.g. Safari), falling back to the static `<img>`. Logo `<image>` tags use paths relative to the illustration SVG (`logos/*.svg`).

example

```bash
<img src="https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/0_login.png" alt="Login" />
```

## Generate API Reference

1. Copy openapi spec from `https://api.endpoints.huggingface.cloud/api-doc/openapi.json`
2. create markdown `widdershins --environment env.json openapi.json -o myOutput.md`
3. copy into `api_reference.mdx`


