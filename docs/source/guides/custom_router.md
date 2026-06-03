# Custom Router

The custom router feature lets you fully customize your load balancing strategy. It's an advanced feature, but gives you very precise control over each routing decision.

This is useful for example when you need:

- **Queue-based routing** to avoid wasting newly scaled-up replicas on burst traffic.
- **Latency-aware routing** which avoids sending requests to overloaded replicas.
- **Other custom strategies** such as sticky sessions or weighted routing.

## How it works

The custom router feature lets you deploy your own **router** that runs next to each replica. When a request comes in, it always goes to the router on the oldest replica, the **leader**. That router decides which replica should handle the request: it can forward it to another replica over the internal network, or hand it to the replica running right next to it.

```
                         (oldest replica is the leader)
                         ┌────────────────────────────────────────┐ 
                         │ ┌──────────┐    either  ┌────────────┐ │
                     ┌───│►│  Router  │──────┬────►│ Replica 1  │ │
    ┌─────────────┐  │   │ └──────────┘      │     └────────────┘ │
    │   Request   │──┘   └───────────────────│────────────────────┘
    └─────────────┘                          or
                         ┌───────────────────│────────────────────┐ 
                         │ ┌──────────┐      │     ┌────────────┐ │
                         │ │  Router  │      └────►│ Replica 2  │ │
                         │ └──────────┘            └────────────┘ │
                         └────────────────────────────────────────┘
```

Whenever the set of replicas changes, for example on scale-up, scale-down, or rolling update, the platform sends the router an updated list of replica addresses by calling `POST /_custom_router/set-backends`. The router uses this to keep its view of the available replicas up to date and route accordingly.

> When `customRouter` is enabled, the replicas of your endpoint are automatically allowed to reach each other over a private internal network, so the router can forward a request to any other replica. This network is scoped to your endpoint's replicas only; it **does not** expose them to anything else.

## Router contract

Any HTTP server that implements the following two endpoints can be used as a custom router image:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/_custom_router/set-backends` | The current list of replica addresses the router may forward to: `{"backends": ["scheme://host:port", ...]}` |
| `GET`  | `/_custom_router/health` | Health check: return 200 when the router itself is ready to serve traffic |

The router listens on the port set by the `customRouter.port` field (default `3000`). Every other request path is treated as a user request to be proxied.

## Enabling the custom router

Currently, you can only enable the custom router through the API. On endpoint creation or update, add a top-level `customRouter` object with these fields:

- `tag` (required): the Docker image reference for your router, e.g. `your-org/your-router-image:1.0.0`.
- `env` (optional): a map of environment variables passed to the router.
- `port` (optional): the port your router listens on. Defaults to `3000`.

For example:
```bash
curl -X POST "https://api.endpoints.huggingface.cloud/v2/endpoint/$NAMESPACE" \
    -H "Authorization: Bearer $HF_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "name": "my-endpoint",
      "type": "private",
      "provider": { "vendor": "aws", "region": "us-east-1" },
      "compute": {
        "accelerator": "gpu",
        "instanceType": "nvidia-l40s",
        "instanceSize": "x1",
        "scaling": {
          "minReplica": 0,
          "maxReplica": 4,
          "scaleToZeroTimeout": 15,
          "measure": {"pendingRequests": 2}
        }
      },
      "model": {
        "repository": "black-forest-labs/FLUX.1-schnell",
        "task": "text-to-image",
        "framework": "pytorch",
        "image": { "huggingface": {} }
      },
      "customRouter": {
        "tag": "your-org/your-router-image:1.0.0",
        "env": {
          "MY_VAR": "value"
        }
      }
    }'
```

To remove the custom router on update, send `customRouter` with a null or absent `tag`:

```json
{ "customRouter": {} }
```

<Tip>
When `customRouter` is set, the `loadBalancer` field in `experimentalFeatures` is ignored.
</Tip>

## Reference implementation: `queued-least-latency`

When an endpoint scales up under load, new replicas take time to start and load the model. Meanwhile requests pile up, and with naive routing they keep queuing on the busy replicas instead of moving to the new ones once they're ready. This hurts most for slow, non-batching workloads like image generation, where each replica handles one request at a time.

The [endpoints-custom-routers repository](https://github.com/huggingface/endpoints-custom-routers) provides a ready-to-use router, `queued-least-latency`, that fixes this: it queues incoming requests and sends each to the least-loaded replica, so new capacity is used as soon as it's ready.

### Routing strategy

- Incoming requests are pushed onto an in-memory FIFO queue.
- A dispatcher picks the replica with the lowest [EWMA](https://en.wikipedia.org/wiki/Exponential_smoothing) latency that is still under a configurable threshold.
- Replicas that have never been tried are treated as latency 0 and picked first, so new capacity is used immediately on scale-up.
- If all replicas are above the threshold, the dispatcher holds the request in the queue until a replica becomes available or the per-request timeout elapses.
- After each response, the measured end-to-end latency (time to last byte) is fed back into the EWMA for that replica.

### Configuration

All settings are environment variables, passed via the `env` field in the `customRouter` config.

#### Routing

Routing decides which replica each request goes to. The knob you'll usually tune is `CUSTOM_ROUTER_LATENCY_THRESHOLD`.

| Variable | Default | What it does |
|---|---|---|
| `CUSTOM_ROUTER_LATENCY_THRESHOLD` | `3.0` | A replica is treated as "loaded" once its average latency exceeds this many seconds; new requests stop going to it until it recovers. |
| `CUSTOM_ROUTER_EWMA_ALPHA` | `0.3` | How quickly that latency average reacts to recent requests (0 to 1). Higher is more reactive, lower is smoother. Rarely needs changing. |

The latency threshold determines when a replica is considered loaded and new requests should queue at the router instead. The right value depends on whether your model benefits from batching:

- **If batching does not increase your throughput** (or if you prioritize latency over throughput), set a low threshold. Once a replica's EWMA exceeds it, the router stops sending more requests there and queues them instead. Each replica handles one request at a time, which is exactly what you want when batching brings no benefit.
- **If batching does increase your throughput** (e.g. LLMs), you may want a higher threshold so that multiple requests can accumulate on a replica and be processed together. This trades some per-request latency for better overall throughput.

For a **diffusion / image generation** model (e.g. ~40 s per request, no batching benefit), the default threshold (3.0 s) works well. After the first completed request the EWMA is ~40 s, which immediately exceeds 3 s and causes new arrivals to queue at the router. Each replica handles one request at a time, which is fine since batching would not help throughput. Only tune queue depth and timeout:

```json
"customRouter": {
  "tag": "huggingface/queued-least-latency:latest",
  "env": {
    "CUSTOM_ROUTER_QUEUE_MAX_SIZE": "200",
    "CUSTOM_ROUTER_QUEUE_TIMEOUT": "300"
  }
}
```

For an **LLM** (batching increases throughput), replicas can process several concurrent requests, but at a latency cost. With the default 3.0 s threshold, a replica would be flagged as loaded after its very first request and forced to handle requests one at a time. Raise the threshold to let replicas accumulate concurrent requests; how high depends on how much latency you are willing to trade for throughput gains:

```json
"customRouter": {
  "tag": "huggingface/queued-least-latency:latest",
  "env": {
    "CUSTOM_ROUTER_LATENCY_THRESHOLD": "65.0",
    "CUSTOM_ROUTER_QUEUE_MAX_SIZE": "2000"
  }
}
```

#### Queue

The queue configuration decides what happens when every replica is loaded. These are the router's backpressure controls: when the queue is full, the oldest waiting request is dropped with `503 Service Unavailable`, and any request that waits longer than the timeout is also dropped with `503`.

| Variable | Default | What it does |
|---|---|---|
| `CUSTOM_ROUTER_QUEUE_MAX_SIZE` | `1000` | Maximum requests held in the queue; beyond this the oldest waiting request is dropped with `503`. |
| `CUSTOM_ROUTER_QUEUE_TIMEOUT` | `1200` | How long (seconds) a request may wait in the queue before being dropped with `503`. |

#### Operational

Settings that don't affect routing.

| Variable | Default | What it does |
|---|---|---|
| `CUSTOM_ROUTER_PORT` | `3000` | Port the router listens on. Must match the `customRouter.port` you set in the API. |
| `CUSTOM_ROUTER_STATE_LOG_INTERVAL` | `30` | Seconds between log lines reporting per-replica state. |

### Recommended values

Use this table as a quick reference for tuning `queued-least-latency` to your needs:

| Scenario | Recommended approach |
|----------|----------------------|
| Burst traffic + autoscaling | Use `queued-least-latency`: queued requests drain to new replicas on scale-up |
| Heterogeneous replica performance | Use `queued-least-latency`: EWMA routing avoids slow replicas |
| Diffusion / no batching benefit | Keep `CUSTOM_ROUTER_LATENCY_THRESHOLD` at default (3.0 s): one request at a time per replica |
| LLM / batching increases throughput | Raise `CUSTOM_ROUTER_LATENCY_THRESHOLD` to allow concurrent requests per replica |

### Metrics

The reference implementation exposes Prometheus metrics at `GET /_custom_router/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `custom_router_queue_depth` | Gauge | Requests currently waiting in the queue |
| `custom_router_backend_ewma_latency_seconds` | Gauge | EWMA latency per replica (`addr` label) |
| `custom_router_backend_inflight_requests` | Gauge | In-flight requests per replica |
| `custom_router_requests_dispatched_total` | Counter | Requests successfully forwarded |
| `custom_router_requests_evicted_total` | Counter | Requests dropped due to full queue |
| `custom_router_requests_timeout_total` | Counter | Requests dropped due to queue timeout |

`GET /_custom_router/health` also returns a JSON snapshot of current queue depth and per-replica EWMA stats, useful for debugging.

## Building your own router

`queued-least-latency` is a starting point, not a requirement. If its routing strategy doesn't fit your use case (say you need KV-cache-aware routing, proactive scale-up signals, or sticky sessions), you can build your own. Any image that satisfies the [Router contract](#router-contract) can be dropped in as the `customRouter` image. A minimal skeleton in Go:

```go
// POST /_custom_router/set-backends
// Called by the platform every time the set of replicas changes.
func handleSetBackends(w http.ResponseWriter, r *http.Request) {
    var payload struct {
        Backends []string `json:"backends"` // e.g. ["http://10.0.0.1:8080", ...]
    }
    json.NewDecoder(r.Body).Decode(&payload)
    updateMyBackendList(payload.Backends)
    json.NewEncoder(w).Encode(map[string]bool{"ok": true})
}

// GET /_custom_router/health
// Return 200 when ready to serve traffic.
func handleHealth(w http.ResponseWriter, r *http.Request) {
    json.NewEncoder(w).Encode(map[string]bool{"ok": true})
}

// Everything else is a user request to forward.
func handleProxy(w http.ResponseWriter, r *http.Request) {
    backend := pickBackend() // your routing logic here
    forwardTo(w, r, backend)
}
```

The full `queued-least-latency` source in the [endpoints-custom-routers repository](https://github.com/huggingface/endpoints-custom-routers) is a good template to fork.
