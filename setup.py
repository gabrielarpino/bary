from setuptools import setup

setup(name='bary',
      version='0.1',
      description='Project comparing Barycenters of Gaussian Processes',
      url='https://github.com/gabrielarpino/bary',
      author='Gabriel Arpino, Wessel Bruinsma',
      author_email='gabriel.arpino@mail.utoronto.ca, wessel.p.bruinsma@gmail.com',
      license='MIT',
      packages=['bary'],
      install_requires=[
	  'tensorflow',
	  'numpy',
        'gpflow',
        'matplotlib',
      ],
      zip_safe=False)
