from setuptools import setup, find_packages

setup(
    name="topofluid2d",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyvoro",
        "shapely",
        "matplotlib",
        "taichi",
    ],
    author="arthifact",
    description="A 2D fluid simulation using Taichi with Voronoi diagram-based topology optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/arthifact/TopoFluid2D",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
