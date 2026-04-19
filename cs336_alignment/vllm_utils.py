"""Co-locate a HuggingFace training policy with a vLLM rollout engine on
the *same* GPU, and stream updated weights between them via CUDA IPC.

Pattern (from vLLM's ``examples/offline_inference/rlhf_utils.py``,
``ColocateWorkerExtension``):

* The vLLM ``LLM`` is created with ``worker_extension_cls`` pointing to
  :class:`WorkerExtension` (defined below). vLLM imports that class in the
  worker subprocess and mixes it into the ``Worker``, so its methods can
  be invoked from the main process via :py:meth:`vllm.LLM.collective_rpc`.
* For weight sync we use **CUDA IPC**: the trainer turns each parameter
  into a ``(rebuild_fn, args)`` IPC handle (via
  ``torch.multiprocessing.reductions.reduce_tensor``); the worker
  reconstructs a tensor pointing at the same GPU memory and feeds it to
  ``model.load_weights``. No NCCL setup, no inter-process broadcast --
  this is what colocation is for.
* :func:`sleep_engine` / :func:`wake_engine` wrap vLLM's
  ``LLM.sleep(level=1)`` / ``LLM.wake_up()`` so the trainer can reclaim
  the GPU between rollouts. ``enable_sleep_mode=True`` must have been
  passed to ``init_vllm`` for these to work.

Works with vLLM V1 (default in 0.8.5+); no Ray dependency.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from vllm import LLM


# ---------------------------------------------------------------------------
# Worker-side extension (runs inside the vLLM engine subprocess).
# ---------------------------------------------------------------------------


class WorkerExtension:
    """Mixed into the vLLM ``Worker`` via ``worker_extension_cls``.

    Must live in an importable module so the worker subprocess can locate
    it by fully-qualified name (we pass
    ``"cs336_alignment.vllm_utils.WorkerExtension"`` to ``LLM(...)``).
    """

    def report_device_uuid(self) -> str:
        """Return the worker GPU's CUDA UUID. The trainer keys IPC
        handles by UUID so that, in principle, multiple workers each
        receive the right shard.
        """
        from vllm.platforms import current_platform

        return current_platform.get_device_uuid(self.device.index)

    def update_weights_from_ipc_handles(self, ipc_handles: dict) -> None:
        """Apply weights shipped from the trainer process via CUDA IPC.

        ``ipc_handles`` is ``{device_uuid: {param_name: (rebuild_fn, args)}}``.
        We pick the entry for our GPU, reconstruct each tensor (which
        re-attaches to the same GPU memory the trainer wrote to), and hand
        ``(name, tensor)`` pairs to ``model.load_weights`` -- vLLM applies
        any per-architecture remapping (fused QKV, etc.) for us.
        """
        from vllm.platforms import current_platform

        device_uuid = current_platform.get_device_uuid(self.device.index)
        handles = ipc_handles[device_uuid]
        device_id = self.device.index
        weights = []
        for name, (func, args) in handles.items():
            # ``args`` from ``reduce_tensor`` includes the source process's
            # device id at index 6; remap to our device id.
            list_args = list(args)
            list_args[6] = device_id
            tensor = func(*list_args)
            weights.append((name, tensor))
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Trainer-side helpers.
# ---------------------------------------------------------------------------


@contextmanager
def _scoped_cuda_visible_devices(device: str):
    """Temporarily restrict ``CUDA_VISIBLE_DEVICES`` while vLLM constructs
    its engine subprocess. vLLM 0.8+ removed the ``device=`` kwarg from
    ``LLM(...)``; the worker binds to whichever GPU is visible at fork
    time. We restore the env after construction so the trainer's CUDA
    contexts on other GPUs (if any) keep working.
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
    gpu_memory_utilization: float = 0.45,
    dtype: str = "bfloat16",
    max_model_len: int | None = None,
    enforce_eager: bool = False,
    enable_sleep_mode: bool = True,
) -> "LLM":
    """Construct a single-GPU vLLM engine on ``device`` with our worker
    extension wired in.

    ``enable_sleep_mode=True`` (default) lets the trainer call
    :func:`sleep_engine` to free vLLM's GPU memory between rollouts and
    :func:`wake_engine` to bring it back -- this is what makes single-GPU
    colocation with the trainable policy practical.
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
            enable_sleep_mode=enable_sleep_mode,
            worker_extension_cls="cs336_alignment.vllm_utils.WorkerExtension",
        )
    return llm


def sleep_engine(llm: "LLM", level: int = 1) -> None:
    """Park vLLM weights on CPU and free its GPU KV cache.

    Call this *after* a rollout so the trainer can use the GPU memory for
    activations / optimizer state. Pair with :func:`wake_engine` before
    the next rollout.
    """
    llm.sleep(level=level)


def wake_engine(llm: "LLM") -> None:
    """Reallocate vLLM's GPU buffers (weights + KV) after a previous
    :func:`sleep_engine`. Always followed by
    :func:`load_policy_into_vllm_instance` so the freshly-allocated weight
    buffers get the latest trainer params.
    """
    llm.wake_up()


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: "LLM") -> None:
    """Push the trainer ``policy``'s weights into the colocated vLLM worker
    via CUDA IPC handles -- no copy, no NCCL.
    """
    from torch.multiprocessing.reductions import reduce_tensor

    # Look up the worker GPU's UUID once (cached on the LLM after the
    # first call, since this never changes for a given engine).
    device_uuid = getattr(llm, "_worker_device_uuid", None)
    if device_uuid is None:
        device_uuid = llm.collective_rpc("report_device_uuid")[0]
        llm._worker_device_uuid = device_uuid  # type: ignore[attr-defined]

    handles: dict[str, tuple] = {}
    for name, p in policy.named_parameters():
        # ``reduce_tensor`` returns ``(rebuild_fn, rebuild_args)``; the
        # rebuild_args carry a CUDA IPC handle referencing the same GPU
        # memory ``p.data`` lives in.
        handles[name] = reduce_tensor(p.detach())
    llm.collective_rpc(
        "update_weights_from_ipc_handles", args=({device_uuid: handles},),
    )
