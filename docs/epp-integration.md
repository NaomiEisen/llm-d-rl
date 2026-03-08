# EPP Integration for llm-d-rl

This document explains how to use the llm-d inference scheduler (EPP) with the llm-d-rl rollout controller for intelligent load balancing during RL training.

## Overview

The rollout controller can route generation requests through Envoy+EPP instead of using simple round-robin engine selection. This provides:

- **Load-aware routing**: Avoid overloaded engines
- **Active request tracking**: Route to engines with fewer in-flight requests
- **Session affinity**: Multi-turn conversations go to the same engine (optional)

## Architecture

```
Training Framework
    ↓ POST /v1/generate
Rollout Controller
    ↓ (if --use-envoy)
Envoy Gateway
    ↓ ext-proc call
EPP (Endpoint Picker Plugin)
    ↓ Analyzes load, active requests
    ↓ Returns optimal engine
Envoy
    ↓ Forwards to selected engine
vLLM Engine
```

## Configuration

### EPP Configuration for RL Training

The EPP configuration at `deploy/config/epp-rl-config.yaml` is optimized for RL training:

- **No KV cache dependency**: Compatible with cache clearing during weight updates
- **Load-based scoring**: Routes based on queue depth and active requests
- **No P/D disaggregation**: Single-phase generation only

**Scorers used:**
1. `active-request-scorer` (weight: 60) - Prefers engines with fewer active requests
2. `load-aware-scorer` (weight: 40) - Considers queue depth

### Rollout Controller Flags

```bash
--use-envoy              # Enable Envoy+EPP routing (default: false)
--envoy-url=<url>        # Envoy gateway URL (default: http://envoy-gateway:8080)
```

## Usage

### Mode 1: Direct Engine Selection (Default)

```bash
# Original behavior - controller picks engines directly
rollout-controller \
  --engines=http://engine-1:8000,http://engine-2:8000
```

### Mode 2: Envoy+EPP Routing

```bash
# Route through Envoy+EPP for intelligent load balancing
rollout-controller \
  --engines=http://engine-1:8000,http://engine-2:8000 \
  --use-envoy \
  --envoy-url=http://envoy-gateway:8080
```

**Note**: When using `--use-envoy`, the `--engines` flag is still used for:
- Weight management (direct NCCL coordination)
- Engine lifecycle (sleep/wake/pause/resume)
- Health checks

Only generation requests (`POST /v1/generate`) are routed through Envoy+EPP.

## Deployment

### Prerequisites

1. **vLLM engines** running with OpenAI-compatible API
2. **Envoy gateway** configured with ext-proc pointing to EPP
3. **EPP service** running with RL-optimized configuration

### Kubernetes Deployment

See `deploy/kubernetes/with-epp.yaml` for a complete deployment manifest including:
- vLLM engines
- Envoy gateway
- EPP service
- Rollout controller

```bash
kubectl apply -f deploy/kubernetes/with-epp.yaml
```

## Weight Updates and EPP

**Important**: EPP does NOT participate in weight management. The rollout controller handles:

- `POST /v1/weights/init` - Direct to engines
- `POST /v1/weights/update` - Direct NCCL coordination
- `POST /v1/engines/sleep` - Direct to engines
- `POST /v1/engines/wake` - Direct to engines

EPP only handles routing for `POST /v1/generate` requests.

### Weight Update Flow

```
1. Training generates rollouts (routed via EPP)
2. Controller calls POST /v1/engines/sleep (direct to engines)
   - Engines clear KV cache
3. Controller calls POST /v1/weights/update (direct NCCL)
   - Trainer broadcasts weights via NCCL
4. Controller calls POST /v1/engines/wake (direct to engines)
5. Training generates more rollouts (routed via EPP)
```

EPP's load-based scorers don't maintain state that needs invalidation, so weight updates don't affect routing decisions.

## Session Affinity (Optional)

If your RL training has multi-turn conversations, you can enable session affinity:

1. **Update EPP config** to include `session-affinity-scorer`
2. **Pass session IDs** in your generate requests:

```python
response = requests.post(
    "http://rollout-controller:8090/v1/generate",
    json={
        "prompt_token_ids": tokens,
        "session_id": f"episode-{episode_id}",  # ← Add this
        "sampling_params": {...}
    }
)
```

## Monitoring

### Check EPP Routing Decisions

EPP exposes metrics at `http://epp-service:9090/metrics`:

```
# Active requests per engine
epp_active_requests{engine="engine-1"} 5
epp_active_requests{engine="engine-2"} 2

# Queue depth per engine
epp_queue_depth{engine="engine-1"} 10
epp_queue_depth{engine="engine-2"} 3
```

### Check Rollout Controller Status

```bash
curl http://rollout-controller:8090/v1/pool/status
```

Returns:
```json
{
  "phase": "Serving",
  "total_engines": 2,
  "ready_engines": 2,
  "weight_version": 5,
  "engines": [
    {"id": "engine-0", "address": "http://engine-1:8000", "ready": true},
    {"id": "engine-1", "address": "http://engine-2:8000", "ready": true}
  ]
}
```

## Troubleshooting

### Generation requests fail with "no engines available"

**Cause**: EPP or Envoy is not reachable

**Solution**:
```bash
# Check Envoy is running
kubectl get pods -l app=envoy-gateway

# Check EPP is running
kubectl get pods -l app=epp

# Test Envoy directly
curl http://envoy-gateway:8080/v1/models
```

### Requests timeout

**Cause**: EPP is taking too long to make routing decisions

**Solution**: Check EPP logs for errors:
```bash
kubectl logs -l app=epp
```

### Weight updates fail

**Cause**: Weight updates bypass Envoy and go directly to engines

**Solution**: Ensure `--engines` flag points to actual engine URLs, not Envoy

## Future Enhancements

### KV Cache Awareness (Planned)

To enable KV cache-aware routing in the future:

1. **Implement "clear all blocks" event** in vLLM
2. **Update EPP config** to use `precise-prefix-cache-scorer`
3. **Notify EPP** of weight updates to invalidate cache state

This will enable routing based on KV cache locality for better throughput.

## References

- [llm-d-inference-scheduler](https://github.com/llm-d/llm-d-inference-scheduler)
- [EPP Architecture](https://github.com/llm-d/llm-d-inference-scheduler/blob/main/docs/architecture.md)
- [Envoy ext-proc](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter)