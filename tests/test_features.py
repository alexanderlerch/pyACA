import unittest
import numpy as np
import numpy.testing as npt

import pyACA


class TestFeatures(unittest.TestCase):

    def test_spectral_centroid(self):
        X = np.zeros(1025)
        fs = 4

        # zero input
        vsc = pyACA.FeatureSpectralCentroid(X, fs)
        self.assertEqual(vsc, 0, "SC 1: Zero input incorrect")

        # one peak input
        X[512] = 1
        vsc = pyACA.FeatureSpectralCentroid(X, fs)
        self.assertEqual(vsc, 1, "SC 2: Delta input incorrect")

        # flat spec input
        X = 2*np.ones(1025)
        vsc = pyACA.FeatureSpectralCentroid(X, fs)
        self.assertEqual(vsc, 1, "SC 3: Flat input incorrect")

        # i/o dimensions
        X = np.ones([1025, 4])
        vsc = pyACA.FeatureSpectralCentroid(X, fs)
        self.assertEqual(len(np.squeeze(vsc)), 4, "SC 4: output vector dimension incorrect")

    def test_spectral_crest(self):
        X = np.zeros(1025)
        fs = 4

        # zero input
        vtsc = pyACA.FeatureSpectralCrestFactor(X, fs)
        self.assertEqual(vtsc, 0, "TSC 1: Zero input incorrect")

        # one peak input
        X[512] = 1
        vtsc = pyACA.FeatureSpectralCrestFactor(X, fs)
        self.assertEqual(vtsc, 1, "TSC 2: Delta input incorrect")

        # flat spec input
        X = 2*np.ones(1025)
        vtsc = pyACA.FeatureSpectralCrestFactor(X, fs)
        self.assertEqual(vtsc, 1/len(X), "TSC 3: Flat input incorrect")

        # i/o dimensions
        X = np.ones([1025, 4])
        vtsc = pyACA.FeatureSpectralCrestFactor(X, fs)
        self.assertEqual(len(np.squeeze(vtsc)), 4, "TSC 4: output vector dimension incorrect")

    def test_spectral_decrease(self):
        X = np.zeros(1025)
        fs = 4

        # zero input
        vsd = pyACA.FeatureSpectralDecrease(X, fs)
        self.assertEqual(vsd, 0, "SD 1: Zero input incorrect")

        # one peak input
        X[512] = 1
        vsd = pyACA.FeatureSpectralDecrease(X, fs)
        self.assertEqual(vsd, 1.0/512, "SD 2: Delta input incorrect")

        # flat spec input
        X = 2*np.ones(1025)
        vsd = pyACA.FeatureSpectralDecrease(X, fs)
        self.assertEqual(vsd, 0, "SD 3: Flat input incorrect")

        # i/o dimensions
        X = np.ones([1025, 4])
        vsd = pyACA.FeatureSpectralDecrease(X, fs)
        self.assertEqual(len(np.squeeze(vsd)), 4, "SD 4: output vector dimension incorrect")

    def test_spectral_flatness(self):
        X = np.zeros(1025)
        fs = 4

        # zero input
        vtf = pyACA.FeatureSpectralFlatness(X, fs)
        self.assertEqual(vtf, 0, "TF 1: Zero input incorrect")

        # one peak input
        X[512] = 1
        vtf = pyACA.FeatureSpectralFlatness(X, fs)
        self.assertEqual(vtf, 0, "TF 2: Delta input")

        # flat spec input
        X = 2*np.ones(1025)
        vtf = pyACA.FeatureSpectralFlatness(X, fs)
        self.assertEqual(vtf, 1, "TF 3: Flat input incorrect")

        # i/o dimensions
        X = np.ones([1025, 4])
        vtf = pyACA.FeatureSpectralFlatness(X, fs)
        self.assertEqual(len(np.squeeze(vtf)), 4, "TF 4: output vector dimension incorrect")

    def test_novelty_flux(self):
        X = np.ones([5, 2])
        X[:, 1] = 0
        vnf = pyACA.NoveltyFlux(X, -1)
        self.assertEqual(vnf[1], 0, "NF 1: hwr incorrect")

        X = np.ones([5, 2])
        X[:, 0] = 0
        vnf = pyACA.NoveltyFlux(X, -1)
        npt.assert_almost_equal(vnf[1], np.sqrt(5) / 5, decimal=7, err_msg="NF 2: hwr incorrect")

    def test_rms(self):

        # zero test
        iBlockLength = 4096
        iHopLength = 2048
        x = np.zeros(iBlockLength)
        fs = 4096

        vrms, t = pyACA.FeatureTimeRms(x, iBlockLength, iHopLength, fs)
        self.assertEqual(vrms[0], -100, "RMS 1: feature value incorrect")

        # sine test
        f0 = 4
        fs = 256
        A = 1
        iBlockLength = 128
        iHopLength = 64
        t = np.arange(0, fs) / fs
        x = A*np.sin(2 * np.pi * f0 * t)

        vrms, t = pyACA.FeatureTimeRms(x, iBlockLength, iHopLength, fs)
        self.assertEqual(t[1]-t[0], .25, "RMS 2: time stamps incorrect")
        npt.assert_almost_equal(vrms[1], 20*np.log10(np.sqrt(2)/2), decimal=7, err_msg="RMS 3: feature value incorrect")

    def test_zerocrossingrate(self):
        fs = 44100
        f = 440.
        iBlockLength = 882
        iHopLength = 441
    
        x = np.sin(2*np.pi * np.arange(fs*1)*f/fs)
    
        # sine input
        vzcr, t = pyACA.FeatureTimeZeroCrossingRate(x, iBlockLength, iHopLength, fs)

        npt.assert_almost_equal(vzcr[1], f*2/fs, decimal=2, err_msg="ZCR 1: feature value incorrect")

    def test_spectral_flux(self):
        X = np.zeros(1025)
        fs = 1

        # zero input
        vsf = pyACA.FeatureSpectralFlux(X, fs)
        self.assertEqual(vsf, 0, "SF 1: Zero input incorrect")

        # one input
        X = np.ones([5, 2])
        vsf = pyACA.FeatureSpectralFlux(X, -1)
        npt.assert_almost_equal(vsf[1], 0, decimal=7, err_msg="SF 2: value incorrect")

        X = np.ones([5, 2])
        X[:, 0] = 0
        vsf = pyACA.FeatureSpectralFlux(X, -1)
        npt.assert_almost_equal(vsf[1], np.sqrt(5) / 5, decimal=7, err_msg="SF 3: value incorrect")


if __name__ == '__main__':
    unittest.main()
