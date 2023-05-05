from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='Wallpaper_defect_classification',
    version='0.1',
    packages=find_packages(where='model'),
    package_dir={'': 'model'},
    py_modules={splitext(basename(path))[0] for path in glob('model/*.py')}
)