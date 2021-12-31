import unittest
import numpy as np
import numpy.testing as npt

import pyACA


class TestTools(unittest.TestCase):

    def test_block_audio(self):
        iBlockLength = 20
        iHopLength = 10
        fs = 1
        numSamples = 101
        x = np.arange(0, numSamples)

        [xb, time] = pyACA.ToolBlockAudio(x, iBlockLength, iHopLength, fs)
        xb = np.squeeze(xb)

        # check dimensions
        targetNumBlocks = np.ceil(numSamples / iHopLength).astype(int)
        dim = xb.shape
        self.assertEqual(dim[0], targetNumBlocks, "TB 1: number of blocks incorrect")
        self.assertEqual(dim[1], iBlockLength, "TB 2: block length incorrect")

        # block content
        self.assertEqual(xb[targetNumBlocks - 2][0], numSamples - 11, "TB 3: block content incorrect")

        # time stamps
        self.assertEqual(time[0], 0, "TB 4: time stamp incorrect")
        self.assertEqual(time[1], iHopLength / fs, "TB 5: time stamp incorrect")

        fs = 40000
        iBlockLength = 1024
        iHopLength = 512
        numSamples = 40000
        x = np.arange(0, numSamples)
        [xb, time] = pyACA.ToolBlockAudio(x, iBlockLength, iHopLength, fs)
        targetNumBlocks = np.ceil(numSamples / iHopLength).astype(int)
        dim = xb.shape
        self.assertEqual(dim[0], targetNumBlocks, "TB 6: number of blocks incorrect")
        self.assertEqual(dim[1], iBlockLength, "TB 7: block length incorrect")

    def test_specgram(self):
        f = 400
        fs = 40000
        iBlockLength = 1024
        iHopLength = 512
        t = np.arange(0, fs - 1) / fs
        x = np.sin(2 * np.pi * f * t)

        [X, f, t] = pyACA.computeSpectrogram(x, fs, None, iBlockLength, iHopLength,)

        # spectrum length
        self.assertEqual(X.shape[0], (iBlockLength // 2 + 1), "SP 1: number of frequency bins incorrect")

        # frequency vector
        self.assertEqual(f[0], 0, "SP 2: frequency vector incorrect")
        npt.assert_almost_equal(f[1], fs / iBlockLength, decimal=7, err_msg="SP 3: frequency vector incorrect")

        # content
        npt.assert_almost_equal(X[10][10] / X[20][10], 4154.15, decimal=2, err_msg="SP 4: magnitude spectrum incorrect")
        npt.assert_almost_equal(X[115][10] / X[116][10], 1.029, decimal=2, err_msg="SP 5: magnitude spectrum incorrect")
