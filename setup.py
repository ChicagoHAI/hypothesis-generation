import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="hypogenic",
    version="0.2.2",
    author="Haokun Liu, Mingxuan Li, Chenfei Yuan, Yangqiaoyu Zhou, Tejes Srivastava",
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
    # Note: doc2json must be installed separately via:
    # pip install git+https://github.com/allenai/s2orc-doc2json@71c022ed4bed3ffc71d22c2ac5cdbc133ad04e3c
)
