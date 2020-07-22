from setuptools import find_packages, setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='meiyume',
    url='https://github.com/travel-code-sleep/meiyume_master_source_codes',
    author='Amit Prusty',
    author_email='dcsamit@nus.edu.sg',
    # Needed to actually package something
    packages=find_packages(),
    namespace_packages=['meiyume'],
    # Needed for dependencies
    # install_requires=['numpy', 'logging', 'pandas', 'selenium', 'missingno', 'spacy',
    #                   'fastai', 'torch', 'torchvision', 'unidecode', 'boto3',
    #                   'matplotlib', 'tldextract', 'tqdm', 'plotly', 'seaborn',
    #                   'sklearn', 'swifter', 'keras', 'tensorflow', 'scipy', 'pyarrow',
    #                   ],
    # *strongly* suggested for in-house use
    version='0.7',
    # The license can be anything you like
    license='Private',
    description='Contains all codes for data scraping, cleaning and machine/deep learning algorithms.',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)
