# MCP Server

The Inference Endpoints MCP (Model Context Protocol) Server connects your MCP-compatible AI assistant (for example Claude Code, Codex, Cursor, Gemini CLI, or Open Code) directly to your Inference Endpoints. Once connected, your assistant can inspect, create, and manage your endpoints on your behalf, all from within your editor, chat, or CLI.

## What you can do

- List and inspect your endpoints, including their status, compute, and model configuration.
- Get a recommended configuration for any Hub model, then create an endpoint from it.
- Update, pause, resume, scale to zero, or delete an existing endpoint.
- Fetch logs, replicas, and metrics for troubleshooting or monitoring.
- Test your endpoints by making inference calls.
- Check your quotas and available cloud providers before deploying.
- Browse the model catalog and review the audit log of actions taken on your endpoints.

## Get started

1. Open the **Connect to our MCP Server** modal from your endpoints dashboard.

2. Pick your client: select your MCP-compatible client (Claude Code, Codex, Cursor, Gemini CLI, or Open Code). The modal shows a ready-to-copy configuration snippet for that client.

3. Paste and restart: copy the snippet into your client's MCP configuration, save, and restart/reload the client. You should see `hf-endpoints` listed as a connected MCP server.

4. The first time it connects, your client opens a browser tab for you to log in to Hugging Face and approve access.

<Tip>
    For full integration with the Hugging Face Hub, we recommend also giving your agent access to the [Hugging Face MCP server](https://huggingface.co/docs/hub/agents-mcp) and the [Hugging Face CLI](https://huggingface.co/docs/hub/agents-cli).
</Tip>

## Using the server

After connecting, ask your assistant to use the Inference Endpoints tools. Example prompts:

- "Deploy `meta-llama/Llama-3.2-1B-Instruct` on GPU under my `research` org, with autoscaling between 0 and 3 replicas, and let me know once it's running."
- "My `chatbot-prod` endpoint feels slow &mdash; check its metrics and recent logs over the last 30 minutes and tell me whether it's a scaling issue or an error spike."
- "Compare GPU quotas across my orgs and suggest the cheapest provider and region I still have quota for."
- "Check if `Qwen/Qwen2.5-7B-Instruct` is in the catalog; if not, get a recommended config for it and show me the diff against my existing `qwen-endpoint`."
- "Why did my `my-endpoint` endpoint fail last night? Check its status, replicas, and logs around the time it went down."
- "Who paused the `summarizer-prod` endpoint, and when? Resume it if it's currently paused."
- "List every endpoint across my namespaces that's been scaled to zero for more than a day, and delete the ones I no longer need &mdash; ask me to confirm each one first."

Your assistant will use the tools exposed by the MCP server to look up your endpoints, configurations, and logs or metrics, then return the results (status, compute, links, and so on) directly in the conversation. You can keep iterating from there &mdash; for example asking it to update the configuration or pause the endpoint next.

## Available tools

Most tools take a `namespace` (your username or an org you belong to, and only those) and, where relevant, an `endpoint_name`.

| Tool | Description |
|------|-------------|
| `list_endpoints` | List your endpoints in a namespace, with filtering and pagination. |
| `get_endpoint` | Fetch the full details of a specific endpoint. |
| `create_endpoint` | Create a new endpoint from a full configuration. |
| `update_endpoint` | Update an existing endpoint's configuration (merged into the current one). |
| `pause_endpoint` | Pause a running endpoint to stop billing; won't scale back up automatically. |
| `resume_endpoint` | Resume a previously paused endpoint. |
| `scale_endpoint_to_zero` | Force an endpoint to zero replicas immediately; can scale back up on the next request. |
| `delete_endpoint` | Permanently delete an endpoint. See [Deleting endpoints](#deleting-endpoints). |
| `call_endpoint` | Make an inference call to an endpoint. |
| `get_recommended_config` | Derive a suggested configuration (compute, engine, image, task) for any Hub model repo. |
| `has_catalog_item` | Cheaply check whether a Hub model repo is in the curated catalog. |
| `list_catalog_items` | List all publicly available catalog items. |
| `get_vendors` | List available cloud providers, regions, and compute configurations. |
| `get_quotas` | Fetch GPU/CPU quota limits and usage for a namespace. |
| `get_endpoint_logs` | Fetch recent log lines for an endpoint, with filtering and pagination. |
| `get_endpoint_replicas` | List an endpoint's current replicas and their status. |
| `get_endpoint_metric` | Fetch a metrics time series (CPU/GPU usage, latency, request counts, etc.) for an endpoint. |
| `get_audit_logs` | Fetch the audit log of actions taken on endpoints in a namespace. Requires a Pro or Enterprise plan. |

<Tip>
Creating and updating endpoints requires a full or partial endpoint configuration object, which can be deeply nested depending on the engine and provider. Rather than writing one by hand, call <code>get_recommended_config</code> first for the model you want to deploy, then pass its output straight into <code>create_endpoint</code>.
</Tip>

## Deleting endpoints

`delete_endpoint` is destructive and permanent, so it follows a preview-then-confirm pattern. Calling it without `confirm` returns a preview instead of deleting anything:

```json
{ "namespace": "my-org", "endpoint_name": "my-endpoint" }
```

```json
{
  "preview": true,
  "message": "This will permanently delete \"my-org/my-endpoint\" and cannot be undone. Call delete_endpoint again with confirm: true to proceed.",
  "endpoint": { "...": "current endpoint details" }
}
```

Only a second call with `confirm: true` actually deletes the endpoint:

```json
{ "namespace": "my-org", "endpoint_name": "my-endpoint", "confirm": true }
```

## Learn more

- [API Reference](../api_reference) &mdash; the full REST API these tools wrap.
- [Hugging Face CLI](https://huggingface.co/docs/hub/en/agents-cli) &mdash; for full integration with the Hugging Face Hub.
