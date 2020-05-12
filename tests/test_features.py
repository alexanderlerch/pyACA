import unittest
import numpy as np

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
        X = np.ones([1025,4])
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
        X = np.ones([1025,4])
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
        X = np.ones([1025,4])
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
        X = np.ones([1025,4])
        vtf = pyACA.FeatureSpectralFlatness(X, fs)
        self.assertEqual(len(np.squeeze(vtf)), 4, "TF 4: output vector dimension incorrect")

    #def test_spectral_flux(self):
    #    X = np.zeros(1025)
    #    fs = 4

    #    # zero input
    #    vsf = pyACA.FeatureSpectralFlatness(X, fs)
    #    self.assertEqual(vsf, 0, "TF 1: Zero input incorrect")

    #    # one peak input
    #    X[512] = 1
    #    vsf = pyACA.FeatureSpectralFlatness(X, fs)
    #    self.assertEqual(vsf, 0, "TF 2: Delta input")

    #    # flat spec input
    #    X = 2*np.ones(1025)
    #    vsf = pyACA.FeatureSpectralFlatness(X, fs)
    #    self.assertEqual(vsf, 1, "TF 3: Flat input incorrect")

    #    # i/o dimensions
    #    X = np.ones([1025,4])
    #    vsf = pyACA.FeatureSpectralFlatness(X, fs)
    #    self.assertEqual(len(np.squeeze(vsf)), 4, "TF 4: output vector dimension incorrect")


if __name__ == '__main__':
    unittest.main()
