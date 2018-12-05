from setuptools import setup, find_packages

setup(
    name='modAL',
    version='0.3.4',
    author='Tivadar Danka',
    author_email='85a5187a@opayq.com',
    description='A modular active learning framework for Python3',
    license='MIT',
    url='https://modAL-python.github.io/',
    packages=['modAL', 'modAL.models', 'modAL.utils'],
    classifiers=['Development Status :: 4 - Beta'],
    install_requires=['numpy>=1.13', 'scikit-learn>=0.18', 'scipy>=0.18'],
)
