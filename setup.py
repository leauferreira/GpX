from setuptools import setup

setup(
    name='GPX',
    version='0.0.3',
    packages=['gp_explainable', 'py_data_manager'],
    package_data={'py_data_manager': ['data/*.csv']},
    url='https://github.com/leauferreira/GpX',
    license='MIT',
    author='Leonardo Augusto Ferreira',
    author_email='leauferreira@cpdee.ufmg.br',
    description='Genetic Programming Explainable - Provide interpretability on black-box Machine learning models by '
                'Genetic Programming ',
    python_requires='>=3.5',
    install_requires=[
        'numpy~=1.18.4',
        'gplearn~=0.4.1',
        'pandas~=1.0.4',
        'pydotplus~=2.0.2',
    ],
)
