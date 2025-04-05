from setuptools import setup, find_packages

setup(
    name="blackjack_ai",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "seaborn",
        "tqdm",
    ],
    python_requires=">=3.7",
)