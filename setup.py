from setuptools import setup

setup(
    name='GPX',
    version='0.0.3',
    packages=['gp_explainer'],
    url='https://github.com/leauferreira/GpX',
    license='MIT',
    author='Leonardo Augusto Ferreira',
    author_email='leauferreira@cpdee.ufmg.br',
    description='GPX - Genetic Programming Explainer',
    python_requires='>=3.5',
    install_requires=[
        'numpy~=1.18.4',
        'gplearn~=0.4.1',
        'pydotplus~=2.0.2',
    ],
)
