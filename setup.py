from setuptools import setup, find_packages

setup(
    name="cursor",
    version="0.1.0",
    author="Yu Zhao",
    author_email="yu.zhao@ed.ac.uk",
    description=r"Cursor Moving",
    packages=find_packages(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yuzhaouoe/GUI-Cursor-Moving",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)