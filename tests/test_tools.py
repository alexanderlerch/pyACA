import unittest
import numpy as np
import numpy.testing as npt

import pyACA


class TestTools(unittest.TestCase):

    def test_blockaudio(self):
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


    def test_freq2midi2freq(self):

        # check concert pitch
        npt.assert_almost_equal(69, pyACA.ToolFreq2Midi(440), decimal=7, err_msg="FMIDI 1: frequency to pitch conversion incorrect")
        npt.assert_almost_equal(440, pyACA.ToolMidi2Freq(69), decimal=7, err_msg="FMIDI 2: pitch to frequency conversion incorrect")

        # generate high resolution pitch vector and corresponding frequencies
        midi = np.arange(0,1280)/10
        hz = 2**((midi-69)/12) * 440

        midiout = pyACA.ToolFreq2Midi(hz)
        hzout = pyACA.ToolMidi2Freq(midi)

        # find maximum deviation
        diffmidi = np.abs(midiout-midi).max()
        diffhz = np.abs(hzout-hz).max()

        npt.assert_almost_equal(diffmidi, 0, decimal=7, err_msg="FMIDI 3: frequency to pitch conversion incorrect")
        npt.assert_almost_equal(diffhz, 0, decimal=7, err_msg="FMIDI 4: pitch to frequency conversion incorrect")


    def test_freq2mel2freq(self):

        # check reference point at 1000Hz
        npt.assert_almost_equal(1000, pyACA.ToolFreq2Mel(1000), decimal=7, err_msg="FMEL 1: frequency to pitch conversion incorrect")
        npt.assert_almost_equal(1000, pyACA.ToolMel2Freq(1000), decimal=7, err_msg="FMEL 2: pitch to frequency conversion incorrect")

        mel = np.arange(0,3000)

        # check back and forth conversion
        npt.assert_almost_equal(mel, pyACA.ToolFreq2Mel(pyACA.ToolMel2Freq(mel)), decimal=7, err_msg="FMEL 3: pitch to frequency to pitch conversion incorrect")
