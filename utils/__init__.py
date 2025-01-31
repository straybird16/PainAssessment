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
logging.info("Initializing utils package...")

from .EmoPainDataset import KinematicsDataset, SlidingWindowDataset
from .models import KinematicsTransformer

""" 
__all__ = [ # Limits what is imported
    'KinematicsDataset',
    'SlidingWindowDataset',
    'KinematicsTransformer',
    ]  """

CONFIG = {
    'version': __version__,
    'author': 'Xiangdong Xie'
}
