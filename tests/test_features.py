import unittest
import numpy as np

import pyACA

class TestFeatures(unittest.TestCase):

    def test_spectral_centroid(self):
        X = np.zeros(1025)
        fs = 4

        # zero input
        vsc = pyACA.FeatureSpectralCentroid(X, fs)
        self.assertEqual(vsc, 0, "SC 1: Zero input should have output 0")

        # one peak input
        X[512] = 1
        vsc = pyACA.FeatureSpectralCentroid(X, fs)
        self.assertEqual(vsc, 1, "SC 2: Delta input should have output 1")

        # flat spec input
        X = 2*np.ones(1025)
        vsc = pyACA.FeatureSpectralCentroid(X, fs)
        self.assertEqual(vsc, 1, "SC 3: Flat input should have output 1")

        # i/o dimensions
        X = np.ones([1025,4])
        vsc = pyACA.FeatureSpectralCentroid(X, fs)
        self.assertEqual(len(np.squeeze(vsc)), 4, "SC 4: output vector dimension incorrect")


    def test_spectral_crest(self):
        X = np.zeros(1025)
        fs = 4

        # zero input
        vtsc = pyACA.FeatureSpectralCrestFactor(X, fs)
        self.assertEqual(vtsc, 0, "TSC 1: Zero input should have output 0")

        # one peak input
        X[512] = 1
        vtsc = pyACA.FeatureSpectralCrestFactor(X, fs)
        self.assertEqual(vtsc, 1, "TSC 2: Delta input should have output 1")

        # flat spec input
        X = 2*np.ones(1025)
        vtsc = pyACA.FeatureSpectralCrestFactor(X, fs)
        self.assertEqual(vtsc, 1/len(X), "TSC 3: Flat input should have output 1")

        # i/o dimensions
        X = np.ones([1025,4])
        vtsc = pyACA.FeatureSpectralCrestFactor(X, fs)
        self.assertEqual(len(np.squeeze(vtsc)), 4, "TSC 4: output vector dimension incorrect")


if __name__ == '__main__':
    unittest.main()
