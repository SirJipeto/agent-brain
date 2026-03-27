from setuptools import setup, find_packages

setup(
    name="agent-brain",
    version="0.1.0",
    description="Associative memory system for AI agents with Neo4j GraphRAG",
    author="SirJipeto",
    author_email="",
    url="https://github.com/SirJipeto/agent-brain",
    packages=find_packages(),
    install_requires=[
        "neo4j>=5.0",
    ],
    extras_require={
        "embeddings": [
            "sentence-transformers>=2.0.0",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="ai agent memory neo4j graphrag rag",
)
