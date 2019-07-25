from setuptools import setup

setup(name='pyaca',
      version='0.1',
      description='scripts accompanying the book "An Introduction to Audio Content Analysis" by Alexander Lerch',
      long_description='Really, the funniest around.',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
      ],
      keywords='audio analysis features pitch key extraction music onset beat detection descriptors',
      url='https://github.com/alexanderlerch/pyACA',
      author='Alexander Lerch',
      author_email='info@AudioContentAnalysis.org',
      license='MIT',
      packages=['pyaca'],
      install_requires=[
          'numpy',
          'scipy',
          'math',
          'matplotlib',
      ],
      zip_safe=False)