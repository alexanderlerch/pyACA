import unittest
import numpy as np
import numpy.testing as npt

import pyACA


class TestShape(unittest.TestCase):

    def test_compute_feature_num_blocks(self):
        blockLength = 4096
        hopLength = 2048
        fs = 44100
        x = np.random.uniform(-1, 1, size=(fs//2, 1))  # 0.5 sec

        # Num blocks considering the extra block appended in computeFeature
        expectedNumBlocks = self.calcNumBlocks(x.size+blockLength, blockLength, hopLength)

        for feature in pyACA.getFeatureList('all'):
            with self.subTest(msg=feature):
                out, t = pyACA.computeFeature(feature, x, fs, iBlockLength=blockLength, iHopLength=hopLength)
                npt.assert_equal(out.shape[-1], expectedNumBlocks)


    def test_spectral_shape_spectrum_input(self):
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


    def test_spectral_shape_spectrogram_single_block_input(self):
        X = np.zeros((1025, 1))
        fs = 44100

        features = pyACA.getFeatureList('spectral')
        features.remove('SpectralMfccs')
        features.remove('SpectralPitchChroma')

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


    def test_spectral_shape_spectrogram_multiple_blocks_input(self):
        X = np.zeros((1025, 16))
        fs = 44100

        features = pyACA.getFeatureList('spectral')
        features.remove('SpectralMfccs')
        features.remove('SpectralPitchChroma')

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
