from setuptools import setup

setup(
    name='modAL',
    version='0.2.0',
    author='Tivadar Danka',
    author_email='85a5187a@opayq.com',
    description='Modular Active Learning framework for Python3',
    license='MIT',
    url='https://cosmic-cortex.github.io/modAL',
    packages=['modAL', 'modAL.utils'],
    classifiers=['Development Status :: 4 - Beta'],
    install_requires=['numpy>=1.13', 'scikit-learn>=0.18', 'scipy>=0.18'],
)
