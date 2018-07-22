import setuptools

setuptools.setup(
    name='rkNN_imputer',
    version="1.2",
    author='Lorenzo Beretta',
    author_email='lorberimm@hotmail.com',
    description='Ranking and kNN imputation algorithm',
    long_description='An imputation method based on ranking and nearest neighbor (kNN)',
    long_description_content_type="text/markdown",
    url='https://github.com/LBcoding/ranking-kNN-imputation',
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
