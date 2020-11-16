import unittest
import numpy as np
import numpy.testing as npt

import pyACA


class TestShape(unittest.TestCase):

    def test_compute_feature_num_blocks(self):
        blockLength = 256   # Using smaller blocksize to reduce time required to test (~7.3 sec)
        hopLength = 128
        fs = 44100
        inputData = np.random.uniform(-1, 1, size=(2*blockLength, 1))

        for inputSize in range(1, 2*blockLength):
            x = inputData[:inputSize]
            expectedNumBlocks = self.calcNumBlocks(x.size+blockLength, blockLength, hopLength)
            for feature in pyACA.getFeatureList('all'):
                with self.subTest(msg=feature + ':' + str(inputSize)):
                    out, t = pyACA.computeFeature(feature, x, fs, iBlockLength=blockLength, iHopLength=hopLength)
                    npt.assert_equal(out.shape[-1], expectedNumBlocks)


    def test_spectral_features_for_1d_input(self):
        fs = 44100
        X = np.zeros(1025)

        features = pyACA.getFeatureList('spectral')
        features.remove('SpectralMfccs')
        features.remove('SpectralPitchChroma')

        for feature in features:
            with self.subTest(msg=feature):
                featureFunc = getattr(pyACA, "Feature" + feature)
                out = featureFunc(X, fs)
                npt.assert_equal(out.shape, ())

        with self.subTest(msg='SpectralMfccs'):
            out = pyACA.FeatureSpectralMfccs(X, fs)
            npt.assert_equal(out.shape, (13,))

        with self.subTest(msg='SpectralPitchChroma'):
            out = pyACA.FeatureSpectralPitchChroma(X, fs)
            npt.assert_equal(out.shape, (12,))


    def test_spectral_features_for_2d_input(self):
        fs = 44100
        features = pyACA.getFeatureList('spectral')
        features.remove('SpectralMfccs')
        features.remove('SpectralPitchChroma')

        # Test for 2d input with only one block
        X = np.zeros((1025, 1))

        for feature in features:
            with self.subTest(msg=feature):
                featureFunc = getattr(pyACA, "Feature" + feature)
                out = featureFunc(X, fs)
                npt.assert_equal(out.shape, (1,))

        with self.subTest(msg='SpectralMfccs'):
            out = pyACA.FeatureSpectralMfccs(X, fs)
            npt.assert_equal(out.shape, (13, 1))

        with self.subTest(msg='SpectralPitchChroma'):
            out = pyACA.FeatureSpectralPitchChroma(X, fs)
            npt.assert_equal(out.shape, (12, 1))

        # Test for 2d input with only multiple block
        X = np.zeros((1025, 16))

        for feature in features:
            with self.subTest(msg=feature):
                featureFunc = getattr(pyACA, "Feature" + feature)
                out = featureFunc(X, fs)
                npt.assert_equal(out.shape, (16,))

        with self.subTest(msg='SpectralMfccs'):
            out = pyACA.FeatureSpectralMfccs(X, fs)
            npt.assert_equal(out.shape, (13, 16))

        with self.subTest(msg='SpectralPitchChroma'):
            out = pyACA.FeatureSpectralPitchChroma(X, fs)
            npt.assert_equal(out.shape, (12, 16))


    def calcNumBlocks(self, inputSize, blockLen, hopLen):
        return np.floor((inputSize - blockLen) / hopLen + 1)
