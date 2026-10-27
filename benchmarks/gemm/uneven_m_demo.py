"""Uneven per-group m demo: shows the [TE] --format stdout tag with the
bracketed per-group m list (the DeepGemm "distribution" field).
"""
import torch
import transformer_engine.pytorch as te

dev = "cuda"
torch.manual_seed(0)

num_gemms, in_f, out_f = 8, 2096, 1536
splits = [256, 512, 1024, 2048, 512, 1024, 256, 4096]  # uneven per-group m
x = torch.randn(sum(splits), in_f, dtype=torch.bfloat16, device=dev)
m_splits = torch.tensor(splits, dtype=torch.int32, device=dev)
m = te.GroupedLinear(num_gemms, in_f, out_f, bias=False,
                     params_dtype=torch.bfloat16).to(dev)

y = m(x, m_splits)
torch.cuda.synchronize()
print("OK, y shape:", tuple(y.shape))
