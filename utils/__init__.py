"""
utils
=========
The utils module for miscellaneous helper functions.

Modules:
    - EmoPainDataset: Handles dataset import from the raw dataframe of EmoPainDataset collected from the .mat file.
"""
__version__ = '1.0.0'

import logging
logging.basicConfig(level=logging.INFO)
logging.info("Initializing mypackage...")

from .EmoPainDataset import KinematicsDataset, function1
__all__ = ['Class1', 'function1', 'Class2']  # Limits what is imported with "from mypackage import *"

CONFIG = {
    'version': __version__,
    'author': 'Xiangdong Xie'
}
