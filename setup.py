from setuptools import setup

setup(
    name='mglyph',
    version='0.5.1',    
    description='The Malleable Glyph package',
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