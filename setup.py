from setuptools import find_packages, setup


setup(
    name='Wallpaper_defect_classification',
    version='0.1',
    packages=find_packages(where='.'),
    install_requires = [
        'torchmetrics'
    ]   
)