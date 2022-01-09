import unittest
import numpy as np
import numpy.testing as npt

import pyACA


class TestComputes(unittest.TestCase):

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
        x = A * np.sin(2 * np.pi * f0 * t)

        [X, f, t] = pyACA.computeSpectrogram(x, fs, np.ones(iBlockLength), iBlockLength, iHopLength, bNormalize=False)

        npt.assert_almost_equal(X[np.int_(f0)][0], A, decimal=7, err_msg="SP 6: magnitude spectrum incorrect")
        npt.assert_almost_equal(np.sum(X), A, decimal=7, err_msg="SP 7: magnitude spectrum incorrect")

        [X, f, t] = pyACA.computeSpectrogram(x, fs, np.ones(iBlockLength), iBlockLength, iHopLength, bNormalize=True)

        npt.assert_almost_equal(X[np.int_(f0)][0], 1, decimal=7, err_msg="SP 6: magnitude spectrum incorrect")

    def test_melspecgram(self):
        f = 400
        f_s = 40000
        iBlockLength = 1024
        iHopLength = 512
        t = np.arange(0, f_s - 1) / f_s
        x = np.sin(2 * np.pi * f * t)

        iNumMelBands = 128
        [M, f, t] = pyACA.computeMelSpectrogram(x, f_s, None, True, iBlockLength, iHopLength, iNumMelBands)
        self.assertEqual(M.shape[0], iNumMelBands, "MSP 1: number of frequency bins incorrect")
        
        iNumMelBands = 256
        [M, f, t] = pyACA.computeMelSpectrogram(x, f_s, None, True, iBlockLength, iHopLength, iNumMelBands)
        self.assertEqual(M.shape[0], iNumMelBands, "MSP 2: number of frequency bins incorrect")
        
        iNumMelBands = 64
        [M, f, t] = pyACA.computeMelSpectrogram(x, f_s, None, True, iBlockLength, iHopLength, iNumMelBands)
        self.assertEqual(M.shape[0], iNumMelBands, "MSP 3: number of frequency bins incorrect")

    def test_chords(self):
        fSeriesOfIntervals = 2**(np.array([[7, 12, 14, 7, 10],
                                           [4, 9, 11, 4, 6],
                                           [0, 5, 7, 0, 3]]) / 12)
        fBaseFreq = 440
        fFreq = fBaseFreq * fSeriesOfIntervals
        f_s = 44100

        # generate audio with cadence plus an additional minor chord
        t = np.arange(0, f_s) / f_s
        x = np.zeros([1, 0])
        for n in range(fFreq.shape[1]):
            x_tmp = np.zeros([1, len(t)])
            for f in range(fFreq.shape[0]):
                x_tmp += np.sin(2 * np.pi * fFreq[f, n] * t)
            x = np.concatenate((x, x_tmp), axis=1)

        x = x.T

        cChordLabel, aiChordIdx, t, P_E = pyACA.computeChords(x, f_s)

        gtchords = np.array([9, 2, 4, 9, 12])
        # shift time stamps to account for block middle
        t = t[2:len(t)]
        for n in range(len(t)):
            if t[n] < 1:
                gt = gtchords[0]
            elif t[n] < 2:
                gt = gtchords[1]
            elif t[n] < 3:
                gt = gtchords[2]
            elif t[n] < 3.99:
                gt = gtchords[3]
            elif t[n] > 4.05:
                gt = gtchords[4]
            else:
                continue

            self.assertEqual(aiChordIdx[0, n], gt, "CH 1: detected chord incorrect")
            self.assertEqual(aiChordIdx[1, n], gt, "CH 2: detected chord incorrect")

    def test_fingerprint(self):

        x = np.zeros(24000)
        f_s = 8000
        SubFingerprint, tf = pyACA.computeFingerprint(x, f_s)

        self.assertEqual(SubFingerprint.shape[0], 32, "FP 1: Subfingerprint length incorrect")
        self.assertEqual(SubFingerprint.shape[1], 234, "FP 2: number of Subfingerprints incorrect")

        # not sure what a good test for fingerprinting would be...
 