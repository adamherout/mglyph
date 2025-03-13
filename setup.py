from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='mglyph',
    version='0.5.2',    
    description='The Malleable Glyph package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://tmgc.fit.vutbr.cz/',
    author='Adam Herout, Vojtech Bartl',
    author_email='herout@vutbr.cz, ibartl@fit.vut.cz',
    license='MIT',
    packages=['mglyph'],
    package_dir={'mglyph': 'src'},
    install_requires=[
                    'skia-python',
                    'colour',
                    'numpy',
                    'matplotlib'
                    ],
    python_requires='>=3.7',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3'
    ],
)