"""Co-locate a HuggingFace training policy with a vLLM rollout engine on
two GPUs, and stream updated weights between them via NCCL.

This follows the canonical vLLM RLHF pattern (see
``vllm/examples/offline_inference/rlhf.py`` + ``rlhf_utils.py``):

* The vLLM ``LLM`` is created with ``worker_extension_cls`` pointing to
  :class:`WorkerExtension` (defined below). vLLM imports that class in the
  worker subprocess and mixes it into the ``Worker``, so its methods can be
  invoked on the worker via :py:meth:`vllm.LLM.collective_rpc`.
* On the first weight sync we create a *stateless* NCCL process group of
  size 2 between the trainer process (rank 0) and the vLLM worker process
  (rank 1). The trainer-side handshake is launched in a background thread
  while we kick off the worker-side handshake via ``collective_rpc`` -- both
  must call into NCCL concurrently for the rendezvous to succeed.
* Every subsequent sync iterates the policy ``state_dict``: tell the worker
  the next ``(name, dtype, shape)`` it should expect, then broadcast the
  GPU tensor over NCCL. The worker hands the received tensor straight to
  ``model.load_weights`` (which knows the per-arch fused-weight remapping).

This works with vLLM V1 (which spawns the model worker in a subprocess);
no Ray dependency needed.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from vllm import LLM


# ---------------------------------------------------------------------------
# Worker-side extension (executed inside the vLLM engine subprocess).
# ---------------------------------------------------------------------------


def _stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """Create a 2-rank NCCL group without polluting ``torch.distributed``.

    Lifted verbatim from the vLLM RLHF example.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)


class WorkerExtension:
    """Mixed into the vLLM ``Worker`` via ``worker_extension_cls``.

    The class lives in its own importable module so that the worker
    subprocess can locate it by fully-qualified name (we pass
    ``"cs336_alignment.vllm_utils.WorkerExtension"`` to ``LLM(...)``).
    """

    # Set on first ``init_weight_update_group`` call.
    model_update_group: object | None = None

    def init_weight_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
    ) -> None:
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset
        self.model_update_group = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device,
        )

    def update_weight(self, name: str, dtype: str, shape) -> None:
        torch_dtype = getattr(torch, dtype)
        weight = torch.empty(tuple(shape), dtype=torch_dtype, device="cuda")
        self.model_update_group.broadcast(  # type: ignore[union-attr]
            weight, src=0, stream=torch.cuda.current_stream(),
        )
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight


# ---------------------------------------------------------------------------
# Trainer-side helpers.
# ---------------------------------------------------------------------------


@contextmanager
def _scoped_cuda_visible_devices(device: str):
    """Pin ``CUDA_VISIBLE_DEVICES`` while the vLLM subprocess forks.

    vLLM 0.8+ removed the ``device=`` kwarg from ``LLM`` -- the engine binds
    to whichever GPU is visible at construction time. We restrict visibility
    to a single index for the duration of the constructor so the worker
    inherits it, then restore the original env so the trainer's CUDA
    contexts on other GPUs in the same parent process keep working.
    """
    idx = device.split(":", 1)[1] if device.startswith("cuda:") else device
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "bfloat16",
    max_model_len: int | None = None,
    enforce_eager: bool = False,
) -> "LLM":
    """Construct a single-GPU vLLM engine pinned to ``device`` with the
    weight-update worker extension wired in.
    """
    # Force the V1 engine -- default in 0.8.5 but be explicit.
    os.environ.setdefault("VLLM_USE_V1", "1")

    from vllm import LLM

    with _scoped_cuda_visible_devices(device):
        llm = LLM(
            model=model_id,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            worker_extension_cls="cs336_alignment.vllm_utils.WorkerExtension",
        )
    return llm


# ---------------------------------------------------------------------------
# NCCL handle cached on the LLM so we only set it up once.
# ---------------------------------------------------------------------------


def _ensure_weight_update_group(llm: "LLM", trainer_device: torch.device):
    """Lazy-init the trainer<->worker NCCL pg and stash it on ``llm``."""
    cached = getattr(llm, "_weight_update_group", None)
    if cached is not None:
        return cached

    # vLLM provides helpers to find a free port on the trainer host.
    from vllm.utils import get_ip, get_open_port

    master_address = get_ip()
    master_port = get_open_port()
    world_size = 2  # trainer (rank 0) + single vLLM worker (rank 1)

    # Both sides must call into NCCL concurrently for the rendezvous to
    # complete, so launch the worker handshake in a background thread.
    rpc_error: list[BaseException] = []

    def _kick_worker():
        try:
            llm.collective_rpc(
                "init_weight_update_group",
                args=(master_address, master_port, 1, world_size),
            )
        except BaseException as e:  # noqa: BLE001
            rpc_error.append(e)

    t = threading.Thread(target=_kick_worker, daemon=True)
    t.start()
    trainer_pg = _stateless_init_process_group(
        master_address, master_port, 0, world_size, trainer_device,
    )
    t.join()
    if rpc_error:
        raise rpc_error[0]

    llm._weight_update_group = trainer_pg  # type: ignore[attr-defined]
    return trainer_pg


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: "LLM") -> None:
    """Stream ``policy``'s weights into the vLLM worker over NCCL.

    Per parameter we ``collective_rpc("update_weight", ...)`` so the worker
    posts a matching ``recv``, then ``broadcast(p, src=0)`` over the
    stateless pynccl group. Tensors travel GPU-to-GPU; no IPC pickling.
    """
    # All trainer-side params must live on the same CUDA device for the
    # broadcast stream to make sense.
    first = next(policy.parameters())
    trainer_device = first.device
    if trainer_device.type != "cuda":
        raise RuntimeError(
            f"policy is on {trainer_device}; vLLM weight sync requires CUDA."
        )

    pg = _ensure_weight_update_group(llm, trainer_device)

    for name, p in policy.named_parameters():
        dtype_name = str(p.dtype).split(".")[-1]  # "torch.bfloat16" -> "bfloat16"
        # Async-fire the worker's matching recv, then broadcast.
        rpc_err: list[BaseException] = []

        def _post_recv(_name=name, _dtype=dtype_name, _shape=tuple(p.shape)):
            try:
                llm.collective_rpc("update_weight", args=(_name, _dtype, _shape))
            except BaseException as e:  # noqa: BLE001
                rpc_err.append(e)

        t = threading.Thread(target=_post_recv, daemon=True)
        t.start()
        pg.broadcast(p.data, src=0, stream=torch.cuda.current_stream())
        t.join()
        if rpc_err:
            raise rpc_err[0]
