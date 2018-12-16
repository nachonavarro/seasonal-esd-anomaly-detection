import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="sesd",
    version="0.1.5",
    author="Nacho Navarro",
    author_email="nachonavarroasv@gmail.com",
    description="Anomaly detection algorithm implemented at Twitter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["sesd"],
    url="https://github.com/nachonavarro/seasonal-esd-anomaly-detection",
    install_requires=[
          'numpy',
          'scipy',
          'statsmodels'
    ],
    tests_require=["pytest"],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)