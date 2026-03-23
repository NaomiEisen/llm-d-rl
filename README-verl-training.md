# veRL PPO Training with llm-d

## 1. Install Prerequisites (KubeRay)

```bash
export NAMESPACE=<your-namespace>
bash deploy/verl/prereqs.sh
```

This installs the KubeRay operator + CRDs (if not installed already) into your namespace, which manages the Ray cluster lifecycle.

## 2. Deploy the Ray Cluster

```bash
kubectl apply -f deploy/verl/raycluster.yaml
```

Wait for the cluster to be ready:

```bash
kubectl get raycluster verl-cluster -w
# STATUS should reach "ready"
```

Copy the training config files to the Ray head pod:

```bash
HEAD_POD=$(kubectl get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

kubectl cp deploy/verl/ppo_llmd.yaml    ${HEAD_POD}:/tmp/ppo_llmd.yaml
kubectl cp deploy/verl/ppo_vanilla.yaml ${HEAD_POD}:/tmp/ppo_vanilla.yaml
```

## 3. Run Training

Make sure the llm-d deployment is running before starting training with the llm-d backend (see [README-epp.md](README-epp.md)).

Verify Ray is running:

```bash
kubectl get raycluster verl-cluster
```

Exec into the head pod:

```bash
HEAD_POD=$(kubectl get pods -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec -ti ${HEAD_POD} -- bash
```

### Training with llm-d

Uses the llm-d controller for rollout — vLLM runs in dedicated pods, training GPU is dedicated to FSDP.

```bash
python -m verl.trainer.main_ppo \
    --config-path /tmp \
    --config-name ppo_llmd \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    data.train_files=/tmp/verl/data/gsm8k/train.parquet \
    data.val_files=/tmp/verl/data/gsm8k/test.parquet
```

### Training without llm-d (vanilla veRL)

Uses veRL's built-in vLLM rollout — vLLM and FSDP share the same GPU.

```bash
python -m verl.trainer.main_ppo \
    --config-path /tmp \
    --config-name ppo_vanilla \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    data.train_files=/tmp/verl/data/gsm8k/train.parquet \
    data.val_files=/tmp/verl/data/gsm8k/test.parquet
```
