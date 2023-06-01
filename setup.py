from setuptools import find_packages, setup

setup(
    name='modAL-python',
    version='0.4.2',
    author='Tivadar Danka',
    author_email='85a5187a@opayq.com',
    description='A modular active learning framework for Python3',
    license='MIT',
    url='https://modAL-python.github.io/',
    packages=['modAL', 'modAL.models', 'modAL.utils'],
    classifiers=['Development Status :: 4 - Beta'],
    install_requires=['numpy', 'scikit-learn>=0.18',
                      'scipy>=0.18', 'pandas>=1.1.0', 'skorch==0.9.0'],
)
