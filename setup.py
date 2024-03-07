import os
from os import path
import git
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

setup(
    name='safe_rl_energy',
    version='1.0',
    keywords='safe_rl_energy',
    description='SafeEnergyATACOM library',
    license='MIT',
    url='https://github.com/kirillin/generalized_atacom_envs/tree/master',
    packages=['safe_rl'],
    install_requires=requires_list,
    include_package_data=True,
)
