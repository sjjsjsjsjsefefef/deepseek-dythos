from setuptools import setup, find_packages

setup(
    name="deepseek-dythos",
    version="3.6.0",
    description="DeepSeek Dythos — Recurrent-Depth Multimodal MoE with 1-Hour 120fps Video",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.7.0",
        "torchvision>=0.22.0",
        "torchaudio>=2.7.0",
        "transformers>=4.51.0",
        "accelerate>=0.29.0",
        "datasets>=2.19.0",
        "einops>=0.8.0",
        "timm>=1.0.0",
        "numpy>=1.26.0",
    ],
)
