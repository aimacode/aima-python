from setuptools import setup

setup(
    name='aimaPy',
    version='1.0',
    description='Python code for the book Artificial Intelligence: A Modern Approach.',
    long_description='Python code for the book Artificial Intelligence: A Modern Approach.',
    author='Peter Norvig',
    author_email='peter@norvig.com',
    url='https://github.com/aimacode/aima-python',
    license="MIT",
    platforms="all",
    packages=['aimaPy'],
    include_package_data=True,

    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)