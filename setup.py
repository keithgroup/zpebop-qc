#!/usr/bin/env python3

"""The setup script."""

from setuptools import setup, find_packages

requirements = ['numpy']

setup_requirements = []

test_requirements = requirements.append([])

setup(
    install_requires=requirements,
    extras_require={},
    include_package_data=True,
    package_data={},
    packages=find_packages(include=['zpebop1', 'zpebop2' ,'zpebop1.*','zpebop2.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    zip_safe=False,
    version='1.0.0',
    entry_points={
        'console_scripts': ['zpebop1=zpebop1.cli:main','zpebop2=zpebop2.cli:main']
    }
)
