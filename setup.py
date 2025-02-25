from setuptools import setup

setup(
    name='mglyph',
    version='1.0.0',    
    description='The MGlyph package',
    url='https://tmgc.fit.vutbr.cz/',
    author='Vojtech Bartl, Adam Herout, ',
    author_email='ibartl@fit.vut.cz, herout@fit.vut.cz',
    license='MIT',
    packages=['mglyph'],
    package_dir={'mglyph': 'src'},
    install_requires=[
                    'skia-python',
                    'colour'
                    ],
    python_requires='>=3.7',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3'
    ],
)