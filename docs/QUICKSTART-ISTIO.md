# Quick Start: Deploy Rollout Controller with Existing llm-d Stack

This guide shows how to deploy your rollout controller alongside an existing llm-d deployment that already has Istio gateway and EPP.

## Prerequisites

✅ llm-d stack already deployed with:
- Istio Ingress Gateway (`infra-kv-events-inference-gateway-istio`)
- EPP (`gaie-kv-events-epp`)
- vLLM engines (`ms-kv-events-llm-d-modelservice-decode`)

## Step 1: Update the Deployment YAML

Edit `deploy/cks/rollout-controller-with-istio.yaml`:

```yaml
metadata:
  namespace: llm-d-precise  # ← Change to YOUR llm-d namespace

containers:
- name: controller
  image: ghcr.io/llm-d/llm-d-rl:latest  # ← Change to YOUR image
  args:
  # Update engine names to match YOUR vLLM pods
  - --engines=http://ms-kv-events-llm-d-modelservice-decode-0:8000,...
  # Update Istio gateway name to match YOUR deployment
  - --envoy-url=http://infra-kv-events-inference-gateway-istio:80
```

## Step 2: Build and Push Your Image

```bash
cd /Users/naomieisenstark/Documents/GitHub/llm-d-rl

# Build
make docker-build

# Push to your registry
docker tag llm-d-rl:latest ghcr.io/YOUR_ORG/llm-d-rl:latest
docker push ghcr.io/YOUR_ORG/llm-d-rl:latest
```

## Step 3: Deploy

```bash
# Deploy the rollout controller
kubectl apply -f deploy/cks/rollout-controller-with-istio.yaml

# Wait for it to be ready
kubectl -n llm-d-precise wait --for=condition=ready pod -l app=rollout-controller --timeout=120s

# Check status
kubectl -n llm-d-precise get pods -l app=rollout-controller
```

## Step 4: Test

```bash
# Port-forward to the rollout controller
kubectl -n llm-d-precise port-forward svc/rollout-controller 8090:8090 &

# Check health
curl http://localhost:8090/v1/health

# Check pool status
curl http://localhost:8090/v1/pool/status

# Test generation (will route through Istio+EPP)
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

## How It Works

```
Training Framework
    ↓ POST /v1/generate
Rollout Controller (your new pod)
    ↓ Forwards to Istio
infra-kv-events-inference-gateway-istio (existing)
    ↓ Calls EPP via ext-proc
gaie-kv-events-epp (existing)
    ↓ Returns routing decision
Istio Gateway forwards to
ms-kv-events-llm-d-modelservice-decode-X (existing)
```

## What Gets Routed Through Istio

✅ **Generation requests** (`POST /v1/generate`)
- Routed via Istio+EPP
- Intelligent load balancing

❌ **Weight management** (`POST /v1/weights/*`)
- Direct to engines
- No Istio/EPP involvement

❌ **Engine lifecycle** (`POST /v1/engines/*`)
- Direct to engines
- No Istio/EPP involvement

## Troubleshooting

### Pod won't start

```bash
# Check logs
kubectl -n llm-d-precise logs -l app=rollout-controller

# Check events
kubectl -n llm-d-precise get events --sort-by='.lastTimestamp'
```

### Generation requests fail

```bash
# Check Istio gateway is reachable
kubectl -n llm-d-precise exec -it deployment/rollout-controller -- \
  curl http://infra-kv-events-inference-gateway-istio:80/v1/models

# Check EPP is reachable
kubectl -n llm-d-precise exec -it deployment/rollout-controller -- \
  curl http://gaie-kv-events-epp:9002
```

### Weight updates fail

```bash
# Check direct engine access
kubectl -n llm-d-precise exec -it deployment/rollout-controller -- \
  curl http://ms-kv-events-llm-d-modelservice-decode-0:8000/health
```

## Monitoring

### Check Istio routing

```bash
# Istio gateway logs
kubectl -n llm-d-precise logs -l app=inference-gateway-istio

# EPP logs
kubectl -n llm-d-precise logs -l inferencepool=gaie-kv-events-epp -c epp
```

### Check EPP metrics

```bash
kubectl -n llm-d-precise port-forward svc/gaie-kv-events-epp 9090:9090 &
curl http://localhost:9090/metrics | grep epp_
```

## Cleanup

```bash
# Remove just the rollout controller
kubectl delete -f deploy/cks/rollout-controller-with-istio.yaml

# llm-d stack remains untouched
```

## Next Steps

1. **Test with your training loop** - See `examples/` directory
2. **Monitor EPP routing decisions** - Check metrics
3. **Tune EPP configuration** - If needed, update EPP ConfigMap
4. **Scale engines** - Add more vLLM replicas as needed

## Summary

✅ **No new infrastructure needed** - Uses existing llm-d stack
✅ **Minimal deployment** - Just your rollout controller
✅ **Easy to test** - Deploy alongside existing workload
✅ **Easy to remove** - Delete one deployment

Your rollout controller now benefits from llm-d's intelligent routing! 🎉