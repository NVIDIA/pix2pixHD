"""
Installs scripts:
   - pix2pixhd-train
   - pix2pixhd-test
   - pix2pixhd-encode-features
   - pix2pixhd-precompute-feature-maps
"""
import codecs
from setuptools import setup, find_packages

with codecs.open('README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    name='pix2pixhd',
    version='1.0',
    description='Synthesizing and manipulating 2048x1024 images with conditional GANs',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, Bryan Catanzaro',
    author_email='tingchunw@nvidia.com, mingyul@nvidia.com, jan@jankautz.com, junyanz@cs.cmu.edu, bcatanzaro@acm.org',
    url='https://github.com/NVIDIA/pix2pixHD',
    license='BSD',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().split('\n'),
    entry_points={
        'console_scripts': [
            'pix2pixhd-train=pix2pixhd.train:main',
            'pix2pixhd-test=pix2pixhd.test:main',
            'pix2pixhd-encode-features=pix2pixhd.encode_features:main',
            'pix2pixhd-precompute-feature-maps=pix2pixhd.precompute_feature_maps:main',
        ]
    },
)
