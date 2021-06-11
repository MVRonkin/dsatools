
from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()
    
setup(
    name="dsatools",
    packages=find_packages(),
    version="0.1.54",
    description=""" The library with a most popular tools for Digital Signal Analysis (DSA)  """,
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
	
	classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
    ],
	
    install_requires=[
          'numpy',
          'scipy',
	  'matplotlib'
      ],
)
