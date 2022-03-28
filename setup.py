import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup_args = dict(
    name="scpyutils",
    version="0.0.2",
    author="Sam Cohan",
    author_email="samcohan007@gmail.com",
    keywords=[
        "Python Utilities",
        "Generic Utilities",
    ],
    description="Collection of various utilities used by Sam Cohan and may be others?",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sam-cohan/scpyutils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

install_requires = [
    "dill",
    "joblib",
    "pandas",
    "scipy",
]

if __name__ == "__main__":
    setuptools.setup(**setup_args, install_requires=install_requires)
