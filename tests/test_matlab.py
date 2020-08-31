"""
Runs MATLAB scripts through python and cross checks results of python modules with corresponding MATLAB code.

Requirements:
    - matlab.engine must be installed. (https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
    - The engine requires python 3.6 or less.
    - Need to specify path to MATLAB code.

More info:
    - https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
    - https://www.mathworks.com/matlabcentral/answers/415682-how-to-run-m-file-in-python

"""


from importlib import util as imputil
matlab_spec = imputil.find_spec("matlab")  # importlib.util.find_spec works for python3 >= v3.4
if matlab_spec is not None:
    import matlab.engine

import os
import logging
import unittest
import numpy as np
import numpy.testing as npt

import pyACA

# Initialize this according to the path on your system:
path_to_matlab_code = '/Users/kaushal/Kaushal/Repositories/ACA-Code'


class TestFeaturesWithMatlab(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    def __init__(self, *args, **kwargs):
        super(TestFeaturesWithMatlab, self).__init__(*args, **kwargs)
        self.matlab_engine = None
        path_valid = os.path.exists(path_to_matlab_code)
        if matlab_spec is not None and path_valid:
            self.matlab_engine = matlab.engine.start_matlab()
            self.matlab_engine.cd(path_to_matlab_code)
        else:
            if matlab_spec is None:
                self.logger.warning("matlab package not found")
            if not path_valid:
                self.logger.warning("Invalid path to Matlab code")

    def __del__(self):
        if self.matlab_engine is not None:
            self.matlab_engine.quit()

    # def test_spectral_decrease(self):
    #
    #     if self.matlab_engine is None:
    #         self.skipTest("Matlab engine not available")
    #
    #     # WhiteNoise input
    #     X_py = np.random.uniform(-1, 1, size=(1025, 16))
    #     X_m = matlab.double(X_py.tolist())
    #     fs = 44100
    #
    #     vsd_py = pyACA.FeatureSpectralDecrease(X_py, fs)
    #     vsd_m = self.matlab_engine.FeatureSpectralDecrease(X_m, fs, nargout=1)
    #
    #     if vsd_py.shape == (1, 1):
    #         vsd_py = vsd_py.item()
    #     vsd_m = np.asfarray(vsd_m)
    #
    #     # npt.assert_almost_equal(vsd_py, vsd_m, decimal=7) //TODO: Use numpy testing instead?
    #     self.assertTrue(((vsd_py - vsd_m) < 1e-7).all(), "SD: MATLAB crosscheck test (Noise input) failed")  # TODO: what should be the precision?
    #

    def test_all_features(self):

        if self.matlab_engine is None:
            self.skipTest("Matlab engine not available")

        # WhiteNoise input
        X_py = np.random.uniform(-1, 1, size=(1025, 5))
        X_m = matlab.double(X_py.tolist())
        fs = 44100

        for feature in pyACA.features:
            self.logger.info('Testing feature:' + feature)
            vsd_py, t_py = pyACA.computeFeature(feature, X_py, fs)
            vsd_m, t_m = self.matlab_engine.ComputeFeature(feature, X_m, fs, nargout=2)
            vsd_m = np.asarray(vsd_m)

            #npt.assert_almost_equal(vsd_py, vsd_m, decimal=5, err_msg="MATLAB crosscheck test (Noise input) failed for " + feature) #TODO: Use numpy testing instead?
            self.assertTrue(((vsd_py - vsd_m) < 1e-7).all(), "MATLAB crosscheck test (Noise input) failed for " + feature)  # TODO: what should be the precision?
