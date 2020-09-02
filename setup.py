
from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()
    
setup(
    name="dsatools",
    packages=find_packages(),
    version="0.1.3",
    description=""" The library with a most popular Digital Signal Analysis (DSA) tools """,
    long_description=readme,
    author="Mikhail Ronkin",
	author_email="mvRonkin@gmail.com",
	url = 'https://github.com/MVRonkin/dsatools',
	download_url = "https://github.com/MVRonkin/dsatools/archive/547abfbb9e452703bb83f90c4d66a350ee4e2ae3.zip",
	keywords = ['Digital Signal Analysis', 
			'Signal Processing', 
			'Digital Signal Processing',
			'Digital Spectrum Processing',
			'Digital Spectrum Analysis'],
			
    license="MIT",
    install_requires=[
          'numpy',
          'scipy',
		  'matplotlib'
      ],
)