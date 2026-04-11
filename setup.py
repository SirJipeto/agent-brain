from setuptools import setup, find_packages

setup(
    name="agent-brain",
    version="0.2.0",
    description="Associative memory system for AI agents with Neo4j GraphRAG",
    author="SirJipeto",
    author_email="",
    url="https://github.com/SirJipeto/agent-brain",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "neo4j>=5.0",
        "spacy>=3.5.0",
        "opentelemetry-api>=1.15.0",
    ],
    extras_require={
        # Embedding providers — user chooses at install time
        "local": ["sentence-transformers>=2.0.0"],
        "openai": ["openai>=1.0.0"],
        "ollama": ["httpx>=0.24.0"],
        # Convenience: install all providers
        "all-embeddings": [
            "sentence-transformers>=2.0.0",
            "openai>=1.0.0",
            "httpx>=0.24.0",
        ],
        # Development
        "dev": [
            "pytest>=7.0",
            "pytest-mock>=3.10",
            "ruff>=0.1.0",
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
