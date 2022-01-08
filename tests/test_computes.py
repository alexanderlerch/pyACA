import unittest
import numpy as np
import numpy.testing as npt

import pyACA


class TestTools(unittest.TestCase):

    def test_specgram(self):
        f = 400
        fs = 40000
        iBlockLength = 1024
        iHopLength = 512
        t = np.arange(0, fs - 1) / fs
        x = np.sin(2 * np.pi * f * t)

        [X, f, t] = pyACA.computeSpectrogram(x, fs, None, iBlockLength, iHopLength)

        # spectrum length
        self.assertEqual(X.shape[0], (iBlockLength // 2 + 1), "SP 1: number of frequency bins incorrect")

        # frequency vector
        self.assertEqual(f[0], 0, "SP 2: frequency vector incorrect")
        npt.assert_almost_equal(f[1], fs / iBlockLength, decimal=7, err_msg="SP 3: frequency vector incorrect")

        # content
        npt.assert_almost_equal(X[10][10] / X[20][10], 4154.15, decimal=2, err_msg="SP 4: magnitude spectrum incorrect")
        npt.assert_almost_equal(X[115][10] / X[116][10], 1.029, decimal=2, err_msg="SP 5: magnitude spectrum incorrect")
        
        # non-windowed content (freq on bin)
        f0 = 4
        fs = 16
        A = 0.5
        iBlockLength = 16
        iHopLength = 16
        t = np.arange(0, fs) / fs
        x = A*np.sin(2 * np.pi * f0 * t)

        [X, f, t] = pyACA.computeSpectrogram(x, fs, np.ones(iBlockLength), iBlockLength, iHopLength, bNormalize=False)

        npt.assert_almost_equal(X[np.int_(f0)][0], A, decimal=7, err_msg="SP 6: magnitude spectrum incorrect")
        npt.assert_almost_equal(np.sum(X), A, decimal=7, err_msg="SP 7: magnitude spectrum incorrect")

        [X, f, t] = pyACA.computeSpectrogram(x, fs, np.ones(iBlockLength), iBlockLength, iHopLength, bNormalize=True)

        npt.assert_almost_equal(X[np.int_(f0)][0], 1, decimal=7, err_msg="SP 6: magnitude spectrum incorrect")
