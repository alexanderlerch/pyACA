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


    def calcNumBlocks(self, inputSize, blockLen, hopLen):
        return np.floor((inputSize - blockLen) / hopLen + 1)
