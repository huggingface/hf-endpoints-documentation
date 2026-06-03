# Custom Router

The custom router feature lets you inject a sidecar container into every replica pod of your endpoint to take full control of routing decisions. Instead of relying on the default proxy behavior, your sidecar receives every request and decides which backend replica to forward it to.

This is useful when you need:

- **Queue-based routing** to avoid wasting newly scaled-up replicas on burst traffic.
- **Latency-aware routing** that avoids sending requests to slow or overloaded replicas.
- **Custom strategies** such as sticky sessions, weighted routing, or federation across providers.

## How it works

When enabled, a sidecar container is injected into every replica pod. The Endpoints proxy always forwards to the leader pod (the oldest ready replica), and the sidecar running there decides which backend handles the request — potentially forwarding it to another pod over the inter-pod network:

```
External request
      ↓
  Endpoints proxy  (always forwards to the leader pod)
      ↓
  Custom router sidecar  (port 3000, makes the routing decision)
      ↓
  Target replica  (same pod or a peer, via inter-pod network)
```

Whenever the replica set changes — on scale-up, scale-down, or rolling update — the platform calls `POST /_custom_router/set-backends` on the sidecar with the updated list of backend addresses. The sidecar uses this to maintain its internal backend view and route accordingly.

When `customRouter` is enabled, inter-pod networking is automatically opened between the replica pods of your endpoint, allowing the sidecar to forward requests to any peer replica. This network is scoped to your endpoint's replica pods only — it does not expose them beyond the pod boundary.

## Sidecar contract

Any HTTP server that implements the following two endpoints can be used as a custom router image:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/_custom_router/set-backends` | Receive the current backend list: `{"backends": ["scheme://host:port", ...]}` |
| `GET`  | `/_custom_router/health` | Readiness probe — return 200 when ready to serve traffic |

Every other request path is treated as a user request to be proxied.

## Enabling the custom router

Add a top-level `customRouter` field to your endpoint creation payload with a Docker image reference:

```bash
curl -X POST "https://api.endpoints.huggingface.cloud/v2/endpoint/YOUR_NAMESPACE" \
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

The [endpoints-custom-routers repository](https://github.com/huggingface/endpoints-custom-routers) provides a ready-to-use router that addresses the most common burst traffic problem: requests queued on the original replica while freshly scaled-up replicas sit idle.

### How it routes

- Incoming requests are pushed onto an in-memory FIFO queue.
- A dispatcher picks the replica with the lowest [EWMA](https://en.wikipedia.org/wiki/Exponential_smoothing) latency that is still under a configurable threshold.
- Replicas that have never been tried are treated as latency 0 and picked first, so new capacity is used immediately on scale-up.
- If all replicas are above the threshold, the dispatcher holds the request in the queue until a replica becomes available or the per-request timeout elapses.
- After each response, the measured end-to-end latency (time to last byte) is fed back into the EWMA for that replica.

### Configuration

All tunables are environment variables, passed via the `env` field in the `customRouter` config:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUSTOM_ROUTER_LATENCY_THRESHOLD` | `3.0` | Max EWMA latency (seconds) before a replica is skipped |
| `CUSTOM_ROUTER_QUEUE_MAX_SIZE` | `1000` | Maximum requests held in the queue |
| `CUSTOM_ROUTER_QUEUE_TIMEOUT` | `1200` | Seconds a request may wait before being dropped with 503 |
| `CUSTOM_ROUTER_EWMA_ALPHA` | `0.3` | EWMA smoothing factor — higher means more reactive to recent latency |
| `CUSTOM_ROUTER_STATE_LOG_INTERVAL` | `30` | Seconds between periodic backend-state log lines |
| `CUSTOM_ROUTER_PORT` | `3000` | Listening port (must be 3000 to satisfy the platform contract) |

**Setting the threshold.** The threshold determines when a backend is considered loaded and new requests should queue at the router instead. The right value depends on whether your model benefits from batching:

- **If batching does not increase your throughput** (or if you prioritize latency over throughput), set a low threshold. Once a backend's EWMA exceeds it, the router stops sending more requests there and queues them at the edge instead. Each backend handles one request at a time, which is exactly what you want when batching brings no benefit.
- **If batching does increase your throughput** (e.g. LLMs), you may want a higher threshold so that multiple requests can accumulate on a backend and be processed together. This trades some per-request latency for better overall throughput.

**Diffusion / image generation** (e.g. ~40 s per request, no batching benefit):

The default threshold (3.0 s) works well. After the first completed request the EWMA is ~40 s, which immediately exceeds 3 s and causes new arrivals to queue at the router. Each backend handles one request at a time, which is fine since batching would not help throughput. Only tune queue depth and timeout:

```json
"customRouter": {
  "tag": "huggingface/queued-least-latency:latest",
  "env": {
    "CUSTOM_ROUTER_QUEUE_MAX_SIZE": "200",
    "CUSTOM_ROUTER_QUEUE_TIMEOUT": "300"
  }
}
```

**LLM** (batching increases throughput):

LLM backends can process several concurrent requests and batching improves throughput — but at a latency cost. With the default 3.0 s threshold, the backend would be flagged as loaded after its very first request and forced to handle requests one at a time. Raise the threshold to allow backends to accumulate concurrent requests; how high depends on how much latency you are willing to trade for throughput gains:

```json
"customRouter": {
  "tag": "huggingface/queued-least-latency:latest",
  "env": {
    "CUSTOM_ROUTER_LATENCY_THRESHOLD": "65.0",
    "CUSTOM_ROUTER_QUEUE_MAX_SIZE": "2000"
  }
}
```

### Backpressure

When the system is overloaded the router applies backpressure in two ways:

- **Queue full**: when the queue reaches `CUSTOM_ROUTER_QUEUE_MAX_SIZE`, the oldest waiting request is dropped with `503 Service Unavailable`.
- **Request timeout**: when a request has been waiting longer than `CUSTOM_ROUTER_QUEUE_TIMEOUT` seconds, it is dropped with `503`.

### Metrics

The sidecar exposes Prometheus metrics at `GET /_custom_router/metrics`:

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

Any HTTP server that implements the two-endpoint contract works. A minimal skeleton in Go:

```go
// POST /_custom_router/set-backends
// Called by the platform on every replica set change.
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

<Tip>
The `queued-least-latency` router is a starting point, not a requirement. If its routing strategy does not fit your use case — for example you need KV-cache-aware routing, proactive scale-up signals, sticky sessions, or federation across providers — implementing your own router only requires satisfying the two-endpoint contract (`/_custom_router/set-backends` and `/_custom_router/health`). Any HTTP server that does so can be dropped in as the `customRouter` image.
</Tip>

## Summary

| Scenario | Recommended approach |
|----------|----------------------|
| Burst traffic + autoscaling | Use `queued-least-latency` — queued requests drain to new replicas on scale-up |
| Heterogeneous replica performance | Use `queued-least-latency` — EWMA routing avoids slow replicas |
| Diffusion / no batching benefit | Keep `CUSTOM_ROUTER_LATENCY_THRESHOLD` at default (3.0 s) — one request at a time per backend |
| LLM / batching increases throughput | Raise `CUSTOM_ROUTER_LATENCY_THRESHOLD` to allow concurrent requests per backend |
| Sticky sessions, proactive scaling, federation | Fork the router and implement your own strategy |
