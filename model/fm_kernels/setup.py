from __future__ import annotations

from pathlib import Path

from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # type: ignore
except ImportError:  # pragma: no cover - torch optional
    CUDAExtension = None
    BuildExtension = None

ROOT = Path(__file__).parent
KERNEL_SOURCES = [str(path) for path in (ROOT / "csrc" / "kernels").glob("*.cu")]

ext_modules = []
if CUDAExtension and KERNEL_SOURCES:
    ext_modules.append(
        CUDAExtension(
            name="fm_kernels._ops",
            sources=KERNEL_SOURCES,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    )

cmdclass = {"build_ext": BuildExtension} if BuildExtension and ext_modules else {}

setup(
    name="fm_kernels",
    version="0.0.1",
    packages=["fm_kernels"],
    package_dir={"fm_kernels": "."},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
