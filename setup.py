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
    install_requires=[
                      'numpy',
                      'skia-python',
                      'colour'
                      ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: Microsoft :: Windows',        
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3'
    ],
)