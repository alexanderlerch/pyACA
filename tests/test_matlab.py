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
import sys
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
        init_engine = True

        if 'matlab.engine' not in sys.modules:
            self.logger.warning("matlab package not found")
            init_engine = False
        if not path_valid:
            self.logger.warning("Invalid path to Matlab code")
            init_engine = False

        if init_engine:
            self.matlab_engine = matlab.engine.start_matlab()
            self.matlab_engine.cd(path_to_matlab_code)

    def __del__(self):
        if self.matlab_engine is not None:
            self.matlab_engine.quit()

    def test_all_features(self):

        if self.matlab_engine is None:
            self.skipTest("Matlab engine not available")

        # WhiteNoise input
        fs = 44100
        x_py = np.random.uniform(-1, 1, size=(fs//2, 1))  # 0.5 sec
        x_m = matlab.double(x_py.tolist())

        for feature in pyACA.getFeatureList():
            self.logger.info('Testing feature:' + feature)
            v_out_py, t_py = pyACA.computeFeature(feature, x_py, fs)
            # Note: fs must be float when passing to Matlab
            v_out_m, t_m = self.matlab_engine.ComputeFeature(feature, x_m, float(fs), nargout=2)

            v_out_py = v_out_py.squeeze()
            v_out_m = np.asarray(v_out_m).squeeze()

            with self.subTest(msg=feature):
                npt.assert_almost_equal(v_out_py, v_out_m, decimal=7, err_msg="MATLAB crosscheck test failed for " + feature)


    def test_mel_spectrogram(self):

        if self.matlab_engine is None:
            self.skipTest("Matlab engine not available")

        # WhiteNoise input
        fs = 44100
        x_py = np.random.uniform(-1, 1, size=(fs//2, 1))  # 0.5 sec
        x_m = matlab.double(x_py.tolist())

        self.logger.info('Testing computeMelSpectrogram')
        M_py, fc_py, t_py = pyACA.computeMelSpectrogram(x_py, fs)
        M_m, fc_m, t_m = self.matlab_engine.ComputeMelSpectrogram(x_m, float(fs), nargout=3)

        M_py = M_py.squeeze()
        M_m = np.asarray(M_m).squeeze()

        with self.subTest(msg='computeMelSpectrogram'):
            npt.assert_almost_equal(M_py, M_m, decimal=7, err_msg="MATLAB crosscheck test failed for computeMelSpectrogram" )
