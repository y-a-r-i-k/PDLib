from setuptools import setup, find_packages


def readme():
  with open('README.md', 'r') as f:
    return f.read()


setup(
  name='PyDeepLib',
  version='0.0.1',
  author='yarik_g',
  author_email='gusevyaroslaveggg666@gmail.com',
  description='A simple library for deep learning',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/y-a-r-i-k/PDLib',
  packages=find_packages(),
  install_requires=['requests>=2.25.1', 'numpy'],
  classifiers=[
    'Programming Language :: Python :: 3.10',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  keywords='example python',
  project_urls={
    'Documentation': 'https://github.com/y-a-r-i-k/PDLib'
  },
  python_requires='>=3.9'
)