# -*- coding: utf-8 -*-
"""
@author: Lorenzo Beretta
"""

from setuptools import setup, find_packages


setup(
    name='rkNN_imputer',
    version="1.2",
    author='Loernzo Beretta',
    author_email='lorberimm@hotmail.com',
    packages=find_packages(),
    url='https://github.com/LBcoding/ranking-kNN-imputation',
    license='License :: OSI Approved :: MIT License',
    description='Ranking and kNN imputation algorithm',
    long_description='An imputation method based on ranking and nearest neighbor (kNN).',
    zip_safe=False,
    install_requires=['numpy', 'scipy', 'scikit-learn', 'skrebate'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.x',
        'Topic :: Utilities'
    ],
    keywords=['machine learning', 'missing data', 'imputation'],
    include_package_data=True,
)
