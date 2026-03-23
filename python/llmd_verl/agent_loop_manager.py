"""AgentLoopManager replacement that routes generation through llm-d.

Drop-in replacement for veRL's AgentLoopManager.  Instead of spawning
Ray-managed vLLM processes, it sends HTTP requests to the llm-d Go
controller which load-balances across its pool of vLLM pods.

Usage
-----
In your veRL config:

    actor_rollout_ref:
      rollout:
        agent:
          agent_loop_manager_class: "llmd_verl.agent_loop_manager.LlmdAgentLoopManager"
        # sampling params still read from the same place:
        temperature: 1.0
        top_p: 1.0
        top_k: -1
        max_tokens: 512     # read from response_length

DataProto contract
------------------
Input (prompts):
    batch["input_ids"]      [B, max_prompt_len]  padded prompt token IDs
    batch["attention_mask"] [B, max_prompt_len]  1=real token, 0=padding
    meta_info["validate"]   bool  (optional) — use greedy sampling
    meta_info["do_sample"]  bool  (optional, False → greedy)

Output (required by fit() downstream):
    batch["prompts"]        [B, prompt_len]                original prompt ids
    batch["responses"]      [B, response_len]              generated ids (right-padded)
    batch["input_ids"]      [B, prompt_len + response_len] full sequence
    batch["attention_mask"] [B, prompt_len + response_len] 1=real, 0=pad
    batch["position_ids"]   [B, prompt_len + response_len] incremental
    batch["response_mask"]  [B, response_len]              1=generated, 0=pad
    meta_info["timing"]     dict  timing metrics
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from tensordict import TensorDict

from verl.utils.model import compute_position_id_with_mask

from .client import RolloutControllerClient
from .config import LlmdRolloutConfig

logger = logging.getLogger(__name__)

# Maximum concurrent HTTP requests to the controller.
# The controller and vLLM pods handle batching internally, so send all
# prompts in parallel — bounded only to avoid exhausting file descriptors.
_MAX_CONCURRENT = 128


def _extract_llmd_config(verl_config) -> LlmdRolloutConfig:
    """Pull llm-d settings from veRL's rollout config."""
    rollout = verl_config.actor_rollout_ref.rollout
    kw = rollout.get("custom", {}).get("llmd", {})
    return LlmdRolloutConfig(**kw)


class LlmdAgentLoopManager:
    """veRL AgentLoopManager that uses llm-d for sequence generation.

    veRL's ray_trainer.py constructs this via create():

        manager = LlmdAgentLoopManager.create(
            config=config,
            worker_group=actor_rollout_wg,    # ignored — llm-d owns inference GPUs
            rollout_resource_pool=pool,        # ignored
        )

    and calls:
        output = manager.generate_sequences(prompts_dataproto)
        replicas = manager.rollout_replicas   # empty list — no veRL-managed pods
    """

    def __init__(self, config, client: RolloutControllerClient, tokenizer=None):
        self.config = config
        self.client = client
        self.tokenizer = tokenizer
        # Empty — LlmdVerlCheckpointEngineManager ignores replicas anyway
        self.rollout_replicas: list = []

    # ------------------------------------------------------------------
    # veRL interface — construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        config,
        worker_group=None,
        rollout_resource_pool=None,
        reward_loop_worker_handles=None,
    ):
        """Create manager.  worker_group / resource_pool are ignored."""
        llmd_config = _extract_llmd_config(config)
        client = RolloutControllerClient(llmd_config)

        # Load tokenizer so we can tokenize raw_prompt from non_tensor_batch.
        # veRL's RLHFDataset puts chat messages in non_tensor_batch["raw_prompt"]
        # rather than pre-tokenized input_ids.
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            model_path = config.actor_rollout_ref.model.path
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            logger.debug("[LLMD] Loaded tokenizer from %s", model_path)
        except Exception as exc:
            logger.warning("[LLMD] Could not load tokenizer: %s", exc)

        return cls(config, client, tokenizer)

    # ------------------------------------------------------------------
    # veRL interface — generation
    # ------------------------------------------------------------------

    def generate_sequences(self, prompts) -> object:
        """Generate responses for a batch of prompts via llm-d HTTP API.

        Args:
            prompts: DataProto with batch["input_ids"] and batch["attention_mask"].

        Returns:
            DataProto with all fields required by veRL's fit() loop.
        """
        t_start = time.perf_counter()

        rollout_cfg = self.config.actor_rollout_ref.rollout
        is_validate = prompts.meta_info.get("validate", False)
        do_sample = prompts.meta_info.get("do_sample", True)

        # Sampling parameters — mirror veRL's AgentLoopWorker logic
        if is_validate or not do_sample:
            temperature = rollout_cfg.val_kwargs.get("temperature", 0.0) if is_validate else 0.0
            top_p = rollout_cfg.val_kwargs.get("top_p", 1.0) if is_validate else 1.0
            top_k = rollout_cfg.val_kwargs.get("top_k", -1) if is_validate else -1
        else:
            temperature = rollout_cfg.temperature
            top_p = rollout_cfg.top_p
            top_k = rollout_cfg.top_k

        max_tokens = rollout_cfg.response_length

        # ------------------------------------------------------------------
        # 1. Tokenize raw_prompt chat messages → token IDs + formatted text
        #
        # veRL's RLHFDataset returns chat messages in non_tensor_batch["raw_prompt"]
        # (not pre-tokenized).  We apply the chat template to get both the
        # formatted text (sent to vLLM via HTTP) and token IDs (used to build
        # the padded output tensors that the training loop expects).
        # ------------------------------------------------------------------
        assert self.tokenizer is not None, (
            "LlmdAgentLoopManager requires a tokenizer. "
            "Set actor_rollout_ref.model.path in your config."
        )
        pad_token_id = prompts.meta_info.get("pad_token_id", None)
        eos_token_id = prompts.meta_info.get("eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        raw_prompts = prompts.non_tensor_batch["raw_prompt"]  # np.ndarray of message-lists
        B = len(raw_prompts)
        prompt_token_ids: list[list[int]] = []
        prompt_texts: list[str] = []
        prompt_lengths: list[int] = []
        for messages in raw_prompts:
            ids = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
            )
            text = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_token_ids.append(ids)
            prompt_texts.append(text)
            prompt_lengths.append(len(ids))

        # Build padded input tensor for the output DataProto
        max_prompt_len = max(prompt_lengths)
        input_ids_tensor = torch.full((B, max_prompt_len), pad_token_id, dtype=torch.long)
        attn_mask_tensor = torch.zeros(B, max_prompt_len, dtype=torch.long)
        for i, ids in enumerate(prompt_token_ids):
            n = len(ids)
            input_ids_tensor[i, :n] = torch.tensor(ids, dtype=torch.long)
            attn_mask_tensor[i, :n] = 1

        # ------------------------------------------------------------------
        # 2. Send all prompts concurrently to llm-d controller
        # ------------------------------------------------------------------
        t_gen_start = time.perf_counter()

        def _generate_one(idx_ids_text):
            idx, ids, text = idx_ids_text
            resp = self.client.generate(
                prompt=text,
                prompt_token_ids=ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=([self.tokenizer.decode([eos_token_id])] if eos_token_id is not None and self.tokenizer is not None else []),
            )
            return idx, resp

        results: list[tuple[int, dict]] = [None] * B
        max_workers = min(B, _MAX_CONCURRENT)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_generate_one, (i, ids, prompt_texts[i])): i
                for i, ids in enumerate(prompt_token_ids)
            }
            for future in as_completed(futures):
                idx, resp = future.result()
                results[idx] = (idx, resp)

        t_gen_end = time.perf_counter()

        # Tokenize the text responses returned by the controller.
        # vLLM's /v1/completions returns text only — the controller has no
        # tokenizer, so tokenization is done here with the loaded tokenizer.
        assert self.tokenizer is not None, (
            "LlmdAgentLoopManager: tokenizer is required to tokenize text responses. "
            "Set actor_rollout_ref.model.path in your config."
        )
        response_token_ids: list[list[int]] = []
        for r in results:
            text = r[1].get("text", "")
            ids = self.tokenizer.encode(text, add_special_tokens=False) if text else []
            response_token_ids.append(ids)

        if response_token_ids:
            sample_text = self.tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
            print(f"[LLMD DEBUG] response len={len(response_token_ids[0])} tokens | START: {repr(sample_text[:200])} | END: {repr(sample_text[-300:])}", flush=True)

        # ------------------------------------------------------------------
        # 3. Pack responses back into DataProto tensors
        # ------------------------------------------------------------------
        max_prompt_len = input_ids_tensor.shape[1]
        max_resp_len = max(max((len(r) for r in response_token_ids), default=0), 1) if response_token_ids else 1

        # response tensors (right-padded with 0)
        resp_ids = torch.zeros(B, max_resp_len, dtype=torch.long)
        resp_mask = torch.zeros(B, max_resp_len, dtype=torch.long)
        for i, tokens in enumerate(response_token_ids):
            n = len(tokens)
            if n > 0:
                resp_ids[i, :n] = torch.tensor(tokens, dtype=torch.long)
                resp_mask[i, :n] = 1

        # full sequence = original padded prompt + response
        full_ids = torch.cat([input_ids_tensor, resp_ids], dim=1)       # [B, P+R]
        full_mask = torch.cat([attn_mask_tensor, resp_mask], dim=1)     # [B, P+R]
        position_ids = compute_position_id_with_mask(full_mask)            # [B, P+R]

        # ------------------------------------------------------------------
        # 4. Build output DataProto
        # ------------------------------------------------------------------
        from verl.protocol import DataProto

        out_batch = TensorDict(
            {
                "prompts":        input_ids_tensor,   # [B, P]
                "responses":      resp_ids,            # [B, R]
                "input_ids":      full_ids,            # [B, P+R]
                "attention_mask": full_mask,           # [B, P+R]
                "position_ids":   position_ids,        # [B, P+R]
                "response_mask":  resp_mask,           # [B, R]
            },
            batch_size=[B],
        )

        t_end = time.perf_counter()
        timing = {
            "generate_sequences": t_end - t_start,
            "llmd/http_generate":  t_gen_end - t_gen_start,
        }

        out_non_tensor = dict(prompts.non_tensor_batch)

        # veRL's training loop unconditionally iterates non_tensor_batch["multi_modal_inputs"]
        # (ray_trainer.py:1380). Add empty-dict entries for text-only batches so it
        # doesn't KeyError and finds nothing to process (same as what the standard
        # AgentLoopWorker does for non-vision samples).
        if "multi_modal_inputs" not in out_non_tensor:
            out_non_tensor["multi_modal_inputs"] = np.array([{} for _ in range(B)], dtype=object)

        # The trainer's extract_reward() unconditionally reads batch["rm_scores"].
        # Provide a zero tensor — actual scoring is done later by the reward
        # manager (e.g. rule-based rewards in core_algos.py for GSM8K).
        out_batch["rm_scores"] = torch.zeros_like(resp_ids, dtype=torch.float32)

        output = DataProto(
            batch=out_batch,
            non_tensor_batch=out_non_tensor,
            meta_info={**prompts.meta_info, "timing": timing},
        )

        logger.info(
            "generate_sequences: B=%d prompt_len=[%d..%d] resp_len=[%d..%d] t=%.2fs",
            B,
            min(prompt_lengths), max(prompt_lengths),
            min(len(r) for r in response_token_ids),
            max_resp_len,
            t_end - t_start,
        )

        return output

    # ------------------------------------------------------------------
    # veRL interface — stubs for optional methods called in fit()
    # ------------------------------------------------------------------

    def clear_kv_cache(self) -> None:
        """No-op: llm-d controller manages KV cache lifecycle."""

    def start_profile(self, **kwargs) -> None:
        """No-op: profiling not supported via HTTP API."""

    def stop_profile(self) -> None:
        """No-op: profiling not supported via HTTP API."""