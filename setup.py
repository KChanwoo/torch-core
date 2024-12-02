from setuptools import setup, find_packages

setup(
    name='torch-core',
    version='0.8',
    description='main core of machine learning module by Chanwoo Gwon',
    author='Chanwoo Gwon',
    author_email='arknell@yonsei.ac.kr',
    url='https://github.com/KChanwoo/torch-core.git',
    install_requires=[
        "torch",
        "timm",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "torchvision",
        "tqdm",
        "lightning",
        "tensorboard"
    ],
    packages=find_packages(exclude=['docs', 'tests*']),
    keywords=['ai', 'pytorch'],
    python_requires='>=3',
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.7'
    ]
)
