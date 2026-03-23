"""Trainer-side NCCL checkpoint engine for llm-d-rl inference pods.

Registers as "llmd" backend in veRL's CheckpointEngineRegistry so that
trainer Ray actors broadcast weights directly to llm-d-managed inference
pods via torch.distributed NCCL, without routing through veRL's rollout
workers.

Uses vLLM's StatelessProcessGroup + PyNcclCommunicator so the trainer and
vLLM use the same NCCL initialization protocol (store-based unique_id
exchange via broadcast_obj).

veRL's flow when backend="llmd":
    engine_workers.py:
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param)

    LlmdNcclCheckpointEngine.send_weights():
        rank 0 -> pynccl.broadcast(tensor, src=0) for each param
        rank 1..N -> consume the generator (FSDP AllGather still happens)
"""

from __future__ import annotations

import logging
import socket
import time
from typing import AsyncGenerator, Generator

import torch
from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry

logger = logging.getLogger(__name__)


def _get_local_ip() -> str:
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)


@CheckpointEngineRegistry.register("llmd")
class LlmdNcclCheckpointEngine(CheckpointEngine):
    """CheckpointEngine that runs on trainer Ray actors and broadcasts to inference pods.

    Only trainer rank 0 participates in the NCCL group with inference pods.
    Other FSDP ranks participate in AllGather (to materialize full params)
    but do not broadcast.

    Engine-agnostic: uses torch.distributed for NCCL rendezvous and broadcast.
    Compatible with any inference engine (vLLM, sglang, etc.) managed by the
    llm-d Go controller.

    Registered as "llmd" backend in CheckpointEngineRegistry.
    """

    def __init__(self, bucket_size: int, is_master: bool = False, **kwargs) -> None:
        from .config import LlmdRolloutConfig
        self.is_master = is_master
        cfg = LlmdRolloutConfig(**kwargs)
        self.bucket_size = bucket_size  # accepted for interface compat, not used
        self.controller_url = cfg.controller_url
        self.master_port = cfg.master_port
        self.nccl_timeout_s = cfg.nccl_timeout_s

        self.rank: int | None = None
        self.world_size: int | None = None
        self._pynccl = None  # PyNcclCommunicator, set in init_process_group

        # Param metadata (names/dtypes/shapes) cached after collect_param_metadata()
        self._metadata_only: bool = False
        self._cached_metadata: dict | None = None

    # ------------------------------------------------------------------
    # CheckpointEngine interface
    # ------------------------------------------------------------------

    def prepare(self) -> dict | None:
        """Return local IP:port from every FSDP worker.

        Called on every FSDP worker before rank assignment. The manager
        collects all results and uses metadata[0] (rank-0 worker's IP:port)
        as the NCCL rendezvous address; other workers' metadata is ignored.
        """
        ip = _get_local_ip()
        return {"master_address": ip, "master_port": self.master_port}

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
    ) -> tuple[dict, dict]:
        """Build NCCL topology: only trainer rank 0 joins the llm-d group.

        rollout_world_size here represents the number of inference pods
        (passed in from LlmdCheckpointEngineManager, not from veRL's
        rollout workers).
        """
        master_meta = metadata[0]  # rank 0's IP:port
        total_world_size = 1 + rollout_world_size  # rank 0 + inference pods

        trainer_kwargs = {
            # rank 0 gets slot 0 in the NCCL group; others get -1 (skip)
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [total_world_size] * trainer_world_size,
            "master_metadata": [master_meta] * trainer_world_size,
        }
        # Inference pods join via HTTP → Go controller, not via veRL worker group
        rollout_kwargs: dict = {}
        return trainer_kwargs, rollout_kwargs

    def init_process_group(
        self,
        rank: int,
        world_size: int,
        master_metadata: dict,
    ) -> None:
        """Initialize NCCL group.  rank<0 means this worker is not rank 0."""
        self.rank = rank
        self.world_size = world_size

        if rank < 0:
            # Non-zero FSDP worker: skip NCCL group, just set rank
            return

        master_address = master_metadata["master_address"]
        master_port = master_metadata["master_port"]

        logger.info(
            "LlmdNcclCheckpointEngine: rank=0 NCCL rendezvous at %s:%d, world_size=%d",
            master_address, master_port, world_size,
        )

        # Use vLLM's StatelessProcessGroup so the trainer and vLLM pods use
        # the same NCCL init protocol: store-based broadcast_obj for unique_id
        # exchange, then PyNcclCommunicator for actual tensor transfer.
        import os
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")

        logger.info(
            "CUDA available=%s initialized=%s device_count=%s current_device=%s",
            torch.cuda.is_available(),
            torch.cuda.is_initialized(),
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
            torch.cuda.current_device() if torch.cuda.is_available() else "N/A",
        )
        logger.info(
            "CUDA_VISIBLE_DEVICES=%s NCCL_SOCKET_IFNAME=%s NCCL_DEBUG=%s",
            os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
            os.environ.get("NCCL_SOCKET_IFNAME", "not set"),
            os.environ.get("NCCL_DEBUG", "not set"),
        )

        # Pin vLLM to use the nvidia pip NCCL (same library as the vLLM engine pod).
        # Both sides must use the same NCCL version for ncclCommInitRank to succeed.
        _site = os.path.dirname(torch.__file__)
        _nvidia_nccl = os.path.normpath(
            os.path.join(_site, "..", "nvidia", "nccl", "lib", "libnccl.so.2")
        )
        if os.path.exists(_nvidia_nccl):
            os.environ["VLLM_NCCL_SO_PATH"] = _nvidia_nccl
            try:
                import vllm.envs as _vllm_envs
                _vllm_envs.VLLM_NCCL_SO_PATH = _nvidia_nccl
            except Exception:
                pass
            print(f"[LLMD DEBUG] Pinned VLLM_NCCL_SO_PATH={_nvidia_nccl}", flush=True)
        else:
            print("[LLMD DEBUG] nvidia pip NCCL not found, using default", flush=True)

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        print(f"[LLMD DEBUG] TRAINER rank=0 creating StatelessProcessGroup: host={master_address} port={master_port} world_size={world_size}", flush=True)
        pg = StatelessProcessGroup.create(
            host=master_address,
            port=master_port,
            rank=0,
            world_size=world_size,
            store_timeout=int(self.nccl_timeout_s),
        )
        print(f"[LLMD DEBUG] TRAINER rank=0 StatelessProcessGroup created, wrapping broadcast_obj", flush=True)

        # Monkey-patch broadcast_obj to trace the unique_id exchange
        _orig_broadcast_obj = pg.broadcast_obj
        def _debug_broadcast_obj(obj, src):
            import pickle
            print(f"[LLMD DEBUG] TRAINER broadcast_obj ENTER: src={src} type={type(obj).__name__} hex={pickle.dumps(obj).hex()[:32]}", flush=True)
            result = _orig_broadcast_obj(obj, src)
            print(f"[LLMD DEBUG] TRAINER broadcast_obj EXIT: src={src}", flush=True)
            return result
        pg.broadcast_obj = _debug_broadcast_obj

        print(f"[LLMD DEBUG] TRAINER calling PyNcclCommunicator (world_size={world_size}) — will block in ncclCommInitRank until vLLM joins", flush=True)
        self._pynccl = PyNcclCommunicator(pg, device=torch.cuda.current_device())
        print(f"[LLMD DEBUG] TRAINER PyNcclCommunicator DONE — ncclCommInitRank + warmup all_reduce completed", flush=True)

        logger.info("LlmdNcclCheckpointEngine: NCCL group established")

    def finalize(self) -> None:
        """Release NCCL resources (called after each update in rebuild_group mode)."""
        self._pynccl = None
        self.rank = None
        torch.cuda.empty_cache()

    async def send_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None]
    ) -> None:
        """Broadcast all parameters to inference pods via NCCL.

        Rank 0 broadcasts; other ranks consume the generator (required to
        complete FSDP AllGather on non-rank-0 workers).
        """
        if self.rank is None:
            raise RuntimeError("init_process_group() must be called before send_weights()")

        if self.rank < 0:
            # Drain generator so FSDP AllGather completes on this worker
            for _name, _tensor in weights:
                pass
            return

        # Rank 0: optionally collect metadata, then broadcast
        start = time.perf_counter()
        count = 0
        stream = torch.cuda.current_stream()
        names, dtypes, shapes = [], [], []
        for name, tensor in weights:
            if self._cached_metadata is None:
                names.append(name)
                dtypes.append(str(tensor.dtype).replace("torch.", ""))
                shapes.append(list(tensor.shape))
            if self._metadata_only:
                count += 1
                continue
            t = tensor.contiguous()
            if not t.is_cuda:
                t = t.cuda()
            self._pynccl.broadcast(t, src=0, stream=stream)
            count += 1
        if self._cached_metadata is None and names:
            self._cached_metadata = {
                "param_names": names,
                "param_dtypes": dtypes,
                "param_shapes": shapes,
            }
        if not self._metadata_only:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        logger.info("send_weights: %s %d tensors in %.3fs",
                    "collected metadata for" if self._metadata_only else "broadcast",
                    count, elapsed)

    def set_metadata_only(self, enabled: bool) -> None:
        """Enable/disable metadata-only mode (skips NCCL broadcast, captures param info)."""
        self._metadata_only = enabled

    def get_cached_metadata(self) -> dict | None:
        """Return cached param metadata after collect_param_metadata() completes."""
        return self._cached_metadata

    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        raise NotImplementedError(
            "Inference pods receive weights via the Go controller, not this engine"
        )
