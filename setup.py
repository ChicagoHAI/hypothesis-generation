import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

import tomllib

with open("pyproject.toml", "rb") as f:
    pyproject_data = tomllib.load(f)
    requirements = pyproject_data["project"]["dependencies"]

setuptools.setup(
    name="hypogenic",
    version="0.3.3",
    author="Haokun Liu, Mingxuan Li, Chenfei Yuan, Yangqiaoyu Zhou, Tejes Srivastava, Chenhao Tan",
    author_email="haokunliu@uchicago.edu",
    description="A package for generating and evaluating hypotheses.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "hypogenic_generation=hypogenic_cmd.generation:main",
            "hypogenic_inference=hypogenic_cmd.inference:main",
        ],
    },
)
