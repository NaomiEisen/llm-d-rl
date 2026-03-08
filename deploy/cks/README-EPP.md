# Deploying llm-d-rl with EPP Integration

This guide explains how to deploy the llm-d-rl rollout controller with Envoy+EPP for intelligent load balancing.

## Prerequisites

1. Kubernetes cluster with GPU nodes
2. HuggingFace token (for model downloads)
3. `kubectl` configured to access your cluster

## Quick Start

### 1. Create namespace and secrets

```bash
kubectl create namespace llm-d-rl

# Create HuggingFace token secret
kubectl create secret generic hf-token \
  --namespace=llm-d-rl \
  --from-literal=token=hf_YOUR_TOKEN_HERE
```

### 2. Deploy everything

```bash
kubectl apply -f with-epp.yaml
```

This deploys:
- 2x vLLM engines (StatefulSet)
- EPP service (for routing decisions)
- Envoy gateway (with ext-proc to EPP)
- Rollout controller (with Envoy routing enabled)

### 3. Wait for pods to be ready

```bash
# Wait for vLLM engines (may take 2-5 minutes for model download)
kubectl -n llm-d-rl wait --for=condition=ready pod -l app=vllm-engine --timeout=600s

# Wait for EPP
kubectl -n llm-d-rl wait --for=condition=ready pod -l app=epp --timeout=120s

# Wait for Envoy
kubectl -n llm-d-rl wait --for=condition=ready pod -l app=envoy-gateway --timeout=120s

# Wait for rollout controller
kubectl -n llm-d-rl wait --for=condition=ready pod -l app=rollout-controller --timeout=120s
```

### 4. Test the deployment

```bash
# Port-forward to rollout controller
kubectl -n llm-d-rl port-forward svc/rollout-controller 8090:8090 &

# Check pool status
curl http://localhost:8090/v1/pool/status

# Test generation
curl -X POST http://localhost:8090/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_token_ids": [1, 2, 3, 4, 5],
    "sampling_params": {
      "max_tokens": 100,
      "temperature": 0.7
    }
  }'
```

## Architecture

```
Training Framework
    ↓
Rollout Controller (port 8090)
    ↓ (generation requests)
Envoy Gateway (port 8080)
    ↓ (ext-proc gRPC)
EPP (port 9002)
    ↓ (routing decision based on load)
Envoy forwards to selected engine
    ↓
vLLM Engine 0 or 1 (port 8000)
```

**Weight management bypasses Envoy:**
```
Rollout Controller
    ↓ (direct NCCL)
vLLM Engines (all)
```

## Components

### vLLM Engines

- **Replicas**: 2 (StatefulSet)
- **Ports**: 8000 (HTTP)
- **Resources**: 1 GPU, 4-8 CPU, 32-64Gi memory per pod
- **Services**:
  - `vllm-engine` (headless)
  - `vllm-engine-0` (individual)
  - `vllm-engine-1` (individual)

### EPP (Endpoint Picker Plugin)

- **Image**: `ghcr.io/llm-d/llm-d-inference-scheduler/epp:latest`
- **Ports**: 9002 (gRPC), 9090 (metrics)
- **Config**: Load-based scoring (no KV cache dependency)
- **Resources**: 500m-1 CPU, 512Mi-1Gi memory

### Envoy Gateway

- **Image**: `envoyproxy/envoy:v1.28-latest`
- **Ports**: 8080 (HTTP), 9901 (admin)
- **Config**: ext-proc filter pointing to EPP
- **Resources**: 500m-1 CPU, 512Mi-1Gi memory

### Rollout Controller

- **Image**: `ghcr.io/llm-d/llm-d-rl:latest`
- **Ports**: 8000 (HTTP)
- **Flags**:
  - `--engines=http://vllm-engine-0:8000,http://vllm-engine-1:8000`
  - `--use-envoy` (enables EPP routing)
  - `--envoy-url=http://envoy-gateway:8080`
- **Resources**: 500m-1 CPU, 512Mi-1Gi memory

## Monitoring

### Check EPP metrics

```bash
kubectl -n llm-d-rl port-forward svc/epp 9090:9090 &
curl http://localhost:9090/metrics | grep epp_
```

### Check Envoy admin

```bash
kubectl -n llm-d-rl port-forward svc/envoy-gateway 9901:9901 &
curl http://localhost:9901/stats | grep ext_proc
```

### Check vLLM engine logs

```bash
kubectl -n llm-d-rl logs vllm-engine-0
kubectl -n llm-d-rl logs vllm-engine-1
```

### Check rollout controller logs

```bash
kubectl -n llm-d-rl logs -l app=rollout-controller
```

## Scaling

### Scale vLLM engines

```bash
kubectl -n llm-d-rl scale statefulset vllm-engine --replicas=4
```

**Important**: After scaling, update:
1. Envoy config to include new engine endpoints
2. Rollout controller `--engines` flag

### Scale EPP

```bash
kubectl -n llm-d-rl scale deployment epp --replicas=2
```

EPP is stateless and can be scaled horizontally.

## Troubleshooting

### Generation requests fail

**Check Envoy can reach EPP:**
```bash
kubectl -n llm-d-rl exec -it deployment/envoy-gateway -- curl http://epp:9002
```

**Check Envoy can reach engines:**
```bash
kubectl -n llm-d-rl exec -it deployment/envoy-gateway -- curl http://vllm-engine-0:8000/v1/models
```

### Weight updates fail

**Check rollout controller can reach engines directly:**
```bash
kubectl -n llm-d-rl exec -it deployment/rollout-controller -- curl http://vllm-engine-0:8000/health
```

### EPP not making routing decisions

**Check EPP logs:**
```bash
kubectl -n llm-d-rl logs -l app=epp
```

**Check Envoy ext-proc stats:**
```bash
kubectl -n llm-d-rl port-forward svc/envoy-gateway 9901:9901
curl http://localhost:9901/stats | grep ext_proc
```

## Customization

### Change EPP configuration

Edit the `epp-config` ConfigMap in `with-epp.yaml`:

```yaml
data:
  config.yaml: |
    # Add session affinity
    plugins:
      - type: session-affinity-scorer
      - type: active-request-scorer
        parameters:
          requestTimeout: 300
      # ...
```

Then apply:
```bash
kubectl apply -f with-epp.yaml
kubectl -n llm-d-rl rollout restart deployment/epp
```

### Disable Envoy routing

To go back to direct engine selection:

```bash
kubectl -n llm-d-rl set env deployment/rollout-controller USE_ENVOY=false
```

Or update the deployment args to remove `--use-envoy`.

## Performance Tuning

### EPP timeout

Increase if EPP is slow:
```yaml
# In Envoy config
grpc_service:
  timeout: 5s  # Increase from default 1s
```

### Request timeout

Increase for long-running generations:
```yaml
# In EPP config
plugins:
  - type: active-request-scorer
    parameters:
      requestTimeout: 600  # 10 minutes
```

### Envoy connection pool

For high throughput:
```yaml
# In Envoy config
clusters:
  - name: vllm_engines
    max_requests_per_connection: 100
```

## Next Steps

- See [docs/epp-integration.md](../../docs/epp-integration.md) for detailed integration guide
- See [examples/](../../examples/) for training loop examples
- Monitor EPP metrics to tune scorer weights