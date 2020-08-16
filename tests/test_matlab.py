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

import unittest
import numpy as np
import matlab.engine

import pyACA

# Initialize this according to the path on your system:
path_to_matlab_code = '/Users/kaushal/Kaushal/Repositories/ACA-Code'


class TestFeaturesWithMatlab(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestFeaturesWithMatlab, self).__init__(*args, **kwargs)
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(path_to_matlab_code)

    def __del__(self):
        self.eng.quit()

    def test_spectral_decrease(self):
        X_py = np.zeros(1025)
        X_m = self.eng.zeros(1025, 1)
        fs = 4

        # zero input
        vsd_py = pyACA.FeatureSpectralDecrease(X_py, fs)                    # From python
        vsd_m = self.eng.FeatureSpectralDecrease(X_m, fs, nargout=1)        # From MATLAB
        self.assertEqual(vsd_py, vsd_m, "SD 1: Zero input incorrect")

        # one peak input
        X_py[512] = 1
        X_m[512] = [1.0]
        vsd_py = pyACA.FeatureSpectralDecrease(X_py, fs)
        vsd_m = self.eng.FeatureSpectralDecrease(X_m, fs)
        self.assertEqual(vsd_py, vsd_m, "SD 2: Delta input incorrect")

        # flat spec input
        X_py = 2*np.ones((1025, 1))
        X_m = matlab.double(X_py.tolist())
        vsd_py = pyACA.FeatureSpectralDecrease(X_py, fs)
        vsd_m = self.eng.FeatureSpectralDecrease(X_m, fs)
        self.assertEqual(vsd_py, vsd_m, "SD 3: Flat input incorrect")

        # i/o dimensions
        X = np.ones((1025, 4))
        X_m = matlab.double(X_py.tolist())
        vsd_py = pyACA.FeatureSpectralDecrease(X_py, fs)
        vsd_m = self.eng.FeatureSpectralDecrease(X_m, fs)
        self.assertEqual(vsd_py, vsd_m, "SD 4: output vector dimension incorrect")
