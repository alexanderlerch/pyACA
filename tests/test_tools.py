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
        [xb, t] = pyACA.ToolBlockAudio(x, iBlockLength, iHopLength, fs)
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

    def test_freq2midi2freq(self):

        # check concert pitch
        npt.assert_almost_equal(69, pyACA.ToolFreq2Midi(440), decimal=7, err_msg="FMIDI 1: frequency to pitch conversion incorrect")
        npt.assert_almost_equal(440, pyACA.ToolMidi2Freq(69), decimal=7, err_msg="FMIDI 2: pitch to frequency conversion incorrect")

        # generate high resolution pitch vector and corresponding frequencies
        midi = np.arange(0, 1280)/10
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

        mel = np.arange(0, 3000)

        # check back and forth conversion
        npt.assert_almost_equal(mel, pyACA.ToolFreq2Mel(pyACA.ToolMel2Freq(mel)), decimal=7, err_msg="FMEL 3: pitch to frequency to pitch conversion incorrect")

    def test_knn(self):

        train_data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [.8, .9, .8], [1.8, 2.0, 1.9], [3, 3, 3]]).T
        train_label1 = np.array([0, 1, 2, 0, 1, 2])
        train_label2 = train_label1 + 5
        test_data = np.array([[10, 10, 10], [2, 2, 2], [1.1, 0.95, 1.3], [0, 0, 0], [1.5, 1.5, 1.5]]).T
        ground_truth = np.array([2, 1, 0, 0, 1])

        est_class = pyACA.ToolSimpleKnn(test_data, train_data, train_label2, 1)

        # dimensions test
        self.assertEqual(len(est_class), len(ground_truth), "KNN 1: incorrect dimensions")

        # label test
        self.assertEqual(min(est_class), 5+min(ground_truth), "KNN 2: incorrect labels")
        self.assertEqual(max(est_class), 5+max(ground_truth), "KNN 3: incorrect labels")

        # content test
        est_class = pyACA.ToolSimpleKnn(test_data, train_data, train_label1, 1)
        self.assertEqual(sum(abs(est_class - ground_truth)), 0, "KNN 4: incorrect result")
 
        est_class = pyACA.ToolSimpleKnn(test_data, train_data, train_label1, 2)
        self.assertEqual(sum(abs(est_class - ground_truth)), 0, "KNN 5: incorrect result")
 
        est_class = pyACA.ToolSimpleKnn(test_data, train_data, train_label1, 5)
        self.assertEqual(sum(abs(est_class - ground_truth)), 0, "KNN 6: incorrect result")

        # different dimensionality and labels
        train_data = np.array([[1, 0], [1, 2], [-1, 2], [2.1, 1]]).T
        train_label = np.array([0, 1, 1, 0])
        test_data = np.array([[0, 0]]).T
        ground_truth1 = np.array([0])
        ground_truth3 = np.array([1])
        ground_truth4 = np.array([0])

        # content test
        est_class = pyACA.ToolSimpleKnn(test_data, train_data, train_label, 1)
        self.assertEqual(sum(abs(est_class - ground_truth1)), 0, "KNN 7: incorrect result")

        est_class = pyACA.ToolSimpleKnn(test_data, train_data, train_label, 3)
        self.assertEqual(sum(abs(est_class - ground_truth3)), 0, "KNN 8: incorrect result")

        est_class = pyACA.ToolSimpleKnn(test_data, train_data, train_label, 4)
        self.assertEqual(sum(abs(est_class - ground_truth4)), 0, "KNN 9: incorrect result")

    def test_loocv(self):

        data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 0, 5, 4, 3, 8, 7, 6]])
        gt = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        [avg_acc, accuracies, confmat] = pyACA.ToolLooCrossVal(data, gt)

        self.assertEqual(len(accuracies)-len(gt), 0, "CV 1: incorrect result dimensions")

    def test_pca(self):
        V1 = np.array([[1, 2, 3, 4], [0.5, 1, 1.5, 2]])
        V2 = -V1
        V = np.concatenate((V1, V2), axis=1)

        # dimensions
        u_pc, T, ev = pyACA.ToolPca(V)

        self.assertEqual(u_pc.shape[0], V.shape[0], "PCA 1: component dimensions incorrect")
        self.assertEqual(u_pc.shape[1], V.shape[1], "PCA 2: component dimensions incorrect")
        self.assertEqual(T.shape[0], V.shape[0], "PCA 3: transformation matrix dimensions incorrect")
        self.assertEqual(T.shape[1], V.shape[0], "PCA 4: transformation matrix dimensions incorrect")
        self.assertEqual(ev.shape[0], V.shape[0], "PCA 5: eigenvalue dimensions incorrect")

        # only one component
        npt.assert_almost_equal(ev[1], 0, decimal=7, err_msg="PCA 6: incorrect eigenvalue")

        fScale = 0.5
        V1 = np.array([[-2, -1, 1, 2], [-2, -1, 1, 2]])
        V2 = fScale * np.vstack((-V1[0, :], V1[1, :]))
        V = np.concatenate((V1, V2), axis=1)

        # two perfectly orthogonal components
        u_pc, T, ev = pyACA.ToolPca(V)

        npt.assert_almost_equal(ev[0], ev[1] / fScale**2, decimal=7, err_msg="PCA 7: incorrect eigenvalues")
        npt.assert_almost_equal(np.abs(np.max(T)), np.abs(np.min(T)), decimal=7, err_msg="PCA 8: incorrect transformation matrix")
        npt.assert_almost_equal(np.sum(np.abs(u_pc[0, :])), np.sum(np.abs(u_pc[1, :])) / fScale, decimal=7, err_msg="PCA 9: incorrect component scaling")
        npt.assert_almost_equal(np.sum(u_pc), 0, decimal=7, err_msg="PCA 10: incorrect component mean")

    def test_feature_selection(self):
        # generate 3D features (dim 1 not separable, dim 2 separable, dim 3 noise
        np.random.seed(11)
        sep = np.array([[.1, .2, .3, .4]])
        offset = .12
        f1 = np.concatenate((.2*(sep + .5*offset), sep - .5*offset), axis=1)
        f2 = np.concatenate((sep - 2*offset, sep + 2*offset), axis=1)
        f3 = 0.3 * np.random.rand(1, f1.shape[1])
        V = np.concatenate((f1, f2, f3), axis=0)
        # assign class labels (last one is wrong)
        classIdx = np.array([0, 0, 0, 0, 1, 1, 1, 0])

        # dimensions
        featIdx, acc = pyACA.ToolSeqFeatureSel(V, classIdx, 2)
        self.assertEqual(len(featIdx), 2, "FeS 1: output dimensions incorrect")
        self.assertEqual(len(featIdx), 2, "FeS 2: output dimensions incorrect")

        # selected features
        featIdx, acc = pyACA.ToolSeqFeatureSel(V, classIdx)
        self.assertEqual(len(featIdx), 3, "FeS 3: output dimensions incorrect")
        self.assertEqual(featIdx[0], 1, "FeS 4: selected features incorrect")
        self.assertEqual(featIdx[1], 0, "FeS 5: selected features incorrect")
        self.assertEqual(featIdx[2], 2, "FeS 6: selected features incorrect")

    def test_hann(self):

        iBlockLength = np.asarray([2, 16, 128, 1024, 16384])

        for b in iBlockLength:
            w = pyACA.ToolComputeHann(b)
            self.assertEqual(len(w), b, "HN 1: window dimension incorrect")

        # note that the window should be periodic
        npt.assert_almost_equal(w[0], 0, decimal=7, err_msg="HN 2: window does not start with 0")
        npt.assert_almost_equal(np.max(w), 1, decimal=7, err_msg="HN 3: window maximum incorrect")
        npt.assert_almost_equal(w[int(iBlockLength[-1]/4)], .5, decimal=7, err_msg="HN 4: window shape incorrect")

    def test_instfreq(self):
        iBlockLength = 1024
        iHopLength = 128
        f_s = 48000
        fFreqRes = f_s/iBlockLength

        # select freqs to generate
        bins = iBlockLength / np.asarray([32, 8, 4])
        fFreq = fFreqRes * (bins + np.asarray([.5, .25, 0]))

        # generate audio
        t = np.arange(0, iBlockLength + iHopLength) / f_s
        x = np.zeros([len(fFreq), len(t)])
        for i, f in enumerate(fFreq):
            x[i, :] = np.sin(2 * np.pi * f * t)

        X = np.zeros([2, iBlockLength]).astype(complex)
        w = pyACA.ToolComputeHann(iBlockLength)
        X[0, :] = np.fft.fft(np.sum(x[:, 0:iBlockLength], axis=0) * w) * 2 / iBlockLength
        X[1, :] = np.fft.fft(np.sum(x[:, iHopLength:iHopLength+iBlockLength], axis=0) * w) * 2 / iBlockLength

        iSpecDim = np.int_(iBlockLength / 2 + 1)
        X = X[:, 0:iSpecDim]
        f_I = pyACA.ToolInstFreq(X, iHopLength, f_s)

        for i, f in enumerate(fFreq):
            npt.assert_almost_equal(f_I[bins[i].astype(int)], f, decimal=4, err_msg="IF 1: incorrect result")

    def test_gmm(self):
        mu = np.array([[1, 2],
                       [-1, -2]])
        sigma = np.array([[[3, .2],
                           [.2, 2]],

                          [[2, 0],
                           [0, 1]]])
        N = np.array([2000,
                      1000])

        np.random.seed(11)
        points = []
        for i in range(len(mu)):
            x = np.random.multivariate_normal(mu[i], sigma[i], N[i])
            points.append(x)
        V = np.concatenate(points).T

        mu_hat, sigma_hat, state = pyACA.ToolGmm(V, 2)

        diffm0 = np.min(np.array([np.sum(np.abs(mu[0] - mu_hat[:, 0])), np.sum(np.abs(mu[0] - mu_hat[:, 1]))]))
        diffm1 = np.min(np.array([np.sum(np.abs(mu[1] - mu_hat[:, 0])), np.sum(np.abs(mu[1] - mu_hat[:, 1]))]))
        npt.assert_almost_equal(diffm0, 0, decimal=1, err_msg="GMM 1: incorrect result")
        npt.assert_almost_equal(diffm1, 0, decimal=1, err_msg="GMM 2: incorrect result")

        diffs0 = np.min(np.array([np.max(np.abs(sigma[0] - sigma_hat[0])), np.max(np.abs(sigma[0] - sigma_hat[1]))]))
        diffs1 = np.min(np.array([np.max(np.abs(sigma[1] - sigma_hat[0])), np.max(np.abs(sigma[1] - sigma_hat[1]))]))
        npt.assert_almost_equal(diffs0, 0, decimal=1, err_msg="GMM 3: incorrect result")
        npt.assert_almost_equal(diffs1, 0, decimal=1, err_msg="GMM 4: incorrect result")
