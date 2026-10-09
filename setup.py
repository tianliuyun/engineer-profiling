from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="engineer-profiling",
    version="0.1.0",
    description="伙伴工程师能力画像与培训推荐：文本分类 + 聚类 + 个性化推荐，赋能 to B 伙伴能力管理",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="tianliuyun",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scikit-learn>=1.3",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)