<h1 align="center">TransformerEngine-for-SAIL</h1>

**TransformerEngine-for-SAIL** is a PPU-adapted kernel library and generator based on TransformerEngine v2.16. It provides runtime dependency, backend kernel, and build workflow adaptations for T-Head AI accelerator chips. It keeps Transformer Engine's high-performance kernel generation capabilities and integrates PPU platform optimizations for running large language models and multimodal models on PPU devices.

This document only covers the basic installation, verification, and usage workflow. For Attention Backend, JIT/AOT build tuning, known issues, and model-specific instructions, see the TransformerEngine-for-SAIL User Guide.

## Highlights

* Easy-to-use modules for building Transformer layers with FP8 support
* Optimizations (e.g. fused kernels) for Transformer models
* Support for FP8 on ZhenWu M890P PPUs
* Support for optimizations across all precisions (FP16, BF16) on ZhenWu ppu001 architecture generations and later

## Examples

```python
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048

# Initialize model and inputs.
model = te.Linear(in_features, out_features, bias=True)
inp = torch.randn(hidden_size, in_features, device="cuda")

# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

# Enable autocasting for the forward pass
with te.autocast(enabled=True, recipe=fp8_recipe):
    out = model(inp)

loss = out.sum()
loss.backward()
```

## Installation

### System Requirements

Before installing TransformerEngine-for-SAIL v2.16, make sure the SAIL SDK and required runtime components are available in your environment.

- SAIL SDK v2.2.0 or later
- Python 3.12
- PyTorch-for-SAIL 2.10.0 or later

For supported operating systems, CUDA Wrapper versions, and the full dependency list, see the TransformerEngine-for-SAIL User Guide.

### Installation Methods

#### Docker (Recommended)

TransformerEngine-for-SAIL is pre-installed in the matching SAIL runtime Docker image, so no extra installation is required. Using this image is recommended because it avoids manual setup of the base runtime environment.

#### Install from PyPI

```bash
pip install transformer_engine==2.16.0 -i https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple  --no-deps --force
```

#### Source Installation

```bash
git clone https://github.com/flytiger-eco/TransformerEngine-for-sail.git -b v2.16 --recursive
cd TransformerEngine-for-sail
export NVTE_FRAMEWORK=pytorch
export NVTE_WITH_NCCL_EP=0
export NVTE_CUDA_ARCHS="80;89"
pip3 install -v --no-build-isolation .
```

#### Environment Variables

These environment variables can be set before installation to customize the build process:

* **CUDA_PATH**: Path to CUDA installation
* **CUDNN_PATH**: Path to cuDNN installation
* **CXX**: Path to C++ compiler
* **NVTE_FRAMEWORK**: Comma-separated list of frameworks to build for (e.g., ``pytorch,jax``)
* **MAX_JOBS**: Limit number of parallel build jobs (default varies by system)
* **NVTE_BUILD_THREADS_PER_JOB**: Control threads per build job
* **NVTE_CUDA_ARCHS**: Semicolon-separated list of CUDA compute architectures to compile for (e.g., ``80;90`` for A100 and H100). If not set, automatically determined based on CUDA version. Setting this can significantly reduce build time and binary size.

## Security
For security information, please see [SECURITY.md](SECURITY.md).

## Disclaimer

- This software is provided for development and debugging purposes. Users assume all risks associated with its use.
- Users are responsible for managing data generated during use and complying with applicable security and compliance requirements.

## License

For the TransformerEngine-for-SAIL license, please see the [LICENSE](LICENSE) file.

## Acknowledgments

TransformerEngine-for-SAIL is developed based on the community open-source TransformerEngine project. We thank the TransformerEngine team and the open-source community for their contributions, and welcome developers to contribute code, documentation, and tests to TransformerEngine-for-SAIL.

