"""
Forms a list of available features by checking for all the Feature*.py files present in the package.

  Args:
    type: type of Features (valid values: 'all', 'spectral' 'temporal')

  Returns:
    features:  list of strings

"""

import glob
import os

import pyACA


def getFeatureList(feature_type='all'):

    if feature_type == 'all':
        feature_type = ''
    elif feature_type == 'spectral':
        feature_type = 'Spectral'
    elif feature_type == 'temporal':
        feature_type = 'Time'

    package_loc = os.path.dirname(pyACA.__file__)
    modules = sorted(glob.glob(os.path.join(package_loc, 'Feature' + feature_type + '*.py')))
    features = [os.path.basename(feature)[len('Feature'):-len('.py')] for feature in modules]
    return features
