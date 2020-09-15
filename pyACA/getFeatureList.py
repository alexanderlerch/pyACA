"""
Forms a list of available features by checking for all the Feature*.py files present in the package.

  Returns:
    features  list of strings

"""

from importlib import util as imputil
import glob
import os


def getFeatureList():
    spec = imputil.find_spec('pyACA')
    modules_loc = spec.submodule_search_locations[0]
    modules = sorted(glob.glob(os.path.join(modules_loc, 'Feature*.py')))
    features = [os.path.basename(feature)[len('Feature'):-len('.py')] for feature in modules]
    return features
