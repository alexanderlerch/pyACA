"""
Forms a list of available features by checking for all the Feature*.py files present in the package.

  Args:
    feature_type: (optional) type of features (valid values: 'all', 'spectral' 'temporal')

  Returns:
    features:  list of strings

"""

import glob
import os

import pyACA


def getFeatureList(feature_type='all'):

    feature_type = feature_type.lower()
    if feature_type == 'all':
        feature_type = ''
    elif feature_type == 'spectral':
        feature_type = 'Spectral'
    elif feature_type == 'temporal':
        feature_type = 'Time'
    else:
        print('Invalid feature type')
        return []

    package_loc = os.path.dirname(pyACA.__file__)
    modules = sorted(glob.glob(os.path.join(package_loc, 'Feature' + feature_type + '*.py')))
    features = [os.path.basename(feature)[len('Feature'):-len('.py')] for feature in modules]
    return features
