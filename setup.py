from setuptools import setup, find_packages

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    name="ChinnuAi",
    version="1.0.1",
    description="Chinnu AI: Quantum-inspired chatbot framework with deep learning integration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="N V R K SAI KAMESH YADAVALLI",
    author_email="saikamesh.y@gmail.com",
    url="https://github.com/Kalasaikamesh944/ChinnuAi.git",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
        "termcolor",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
