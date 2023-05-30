from setuptools import setup

setup(
    name='GPX',
    version='0.0.6.18',
    packages=['gp_explainer', 'explainer', 'neighborhood', "translate", 'explain'],
    package_data={'pydm': ['data/*.csv']},
    url='https://github.com/leauferreira/GpX',
    license='MIT',
    author='Leonardo Augusto Ferreira',
    author_email='leauferreira@cpdee.ufmg.br',
    description='GPX - Genetic Programming Explainer',
    python_requires='>=3.5',
    install_requires=[
        'numpy >= 1.21.2',
        'pydotplus~=2.0.2',
        'sympy~=1.12'
    ],
)
