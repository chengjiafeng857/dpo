import torch


def resolve_torch_dtype(precision: str) -> torch.dtype | None:
    precision = (precision or "").lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    return None
