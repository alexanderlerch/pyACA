import unittest
import numpy as np
import numpy.testing as npt

import pyACA


class TestTools(unittest.TestCase):

    def test_nmf(self):
        np.random.seed(42)
        X = np.random.rand(128, 6) / 20
        X[np.arange(4, 128, 4), 0:2] = 1
        X[np.arange(7, 128, 7), 2:4] = 1
        X[:, 4:6] = 0.7 * X[:, 0:2] + 0.3 * X[:, 2:4]
        W, H, err = pyACA.ToolSimpleNmf(X, 2)

        npt.assert_almost_equal(W[8, 1], np.mean(W[np.arange(4, 128, 4), 1]), decimal=3, err_msg="NMF 1: dictionary incorrect")
        npt.assert_almost_equal(W[12, 1], np.mean(W[np.arange(4, 128, 4), 1]), decimal=3, err_msg="NMF 2: dictionary incorrect")
        npt.assert_almost_equal(W[124, 1], np.mean(W[np.arange(4, 128, 4), 1]), decimal=3, err_msg="NMF 3: dictionary incorrect")

        npt.assert_almost_equal(W[21, 0], np.mean(W[np.arange(7, 128, 7), 0]), decimal=3, err_msg="NMF 4: dictionary incorrect")
        npt.assert_almost_equal(W[84, 0], np.mean(W[np.arange(7, 128, 7), 0]), decimal=3, err_msg="NMF 5: dictionary incorrect")
        npt.assert_almost_equal(W[105, 0], np.mean(W[np.arange(7, 128, 7), 0]), decimal=3, err_msg="NMF 6: dictionary incorrect")

        self.assertEqual(np.max(np.diff(err)) < 0, True, "NMF 7: loss incorrect")

        for n in range(3):
            npt.assert_almost_equal(H[1, 2*n]-H[1, 2*n+1], 0, decimal=0, err_msg="NMF 8: activation incorrect")
            npt.assert_almost_equal(H[0, 2*n]-H[0, 2*n+1], 0, decimal=0, err_msg="NMF 9: activation incorrect")
        self.assertEqual(np.mean(H[0, 0:2]) < 1, True, "NMF 10: activation incorrect")
        self.assertEqual(np.mean(H[1, 2:4]) < 1, True, "NMF 11: activation incorrect")
        self.assertEqual(np.mean(H[1, 0:2]) > np.mean(H[1, 4:6]), True, "NMF 12: activation incorrect")
        self.assertEqual(np.mean(H[0, 2:4]) > np.mean(H[0, 4:6]), True, "NMF 13: activation incorrect")

    def test_resample(self):
        fs_in = 8000
        fs_out = 44100
        fFreq = 100
        t_in = np.arange(0, fs_in) / fs_in
        x = np.sin(2 * np.pi * fFreq * t_in)

        [x_out, t_out] = pyACA.ToolResample(x, fs_out, fs_in)
        x_gt = np.sin(2 * np.pi * fFreq * t_out)

        # check output sample rate
        npt.assert_almost_equal(1 / (t_out[1]-t_out[0]), fs_out, decimal=7, err_msg="RS 1: sample rate incorrect")

        # check output samples
        npt.assert_almost_equal(np.mean(np.abs(x_gt - x_out)[10:-10]), 0, decimal=3, err_msg="RS 2: interpolation incorrect")
        
        fs_in = 48000
        fs_out = 7003
        fFreq = 100
        t_in = np.arange(0, fs_in) / fs_in
        x = np.sin(2 * np.pi * fFreq * t_in)

        [x_out, t_out] = pyACA.ToolResample(x, fs_out, fs_in)
        x_gt = np.sin(2 * np.pi * fFreq * t_out)

        # check output sample rate
        npt.assert_almost_equal(1 / (t_out[1]-t_out[0]), fs_out, decimal=7, err_msg="RS 3: sample rate incorrect")

        # check output samples
        npt.assert_almost_equal(np.mean(np.abs(x_gt - x_out)[10:-10]), 0, decimal=3, err_msg="RS 4: interpolation incorrect")

    def test_viterbi(self):
        # states: healthy: 0, fever: 1
        # obs: normal: 0, cold: 1, dizzy: 2
        # V = np.array([0, 1, 2])
        
        # start prob: healthy: 0.6, fever: 0.4
        p_s = np.array([0.6, 0.4])
        
        # emission prob: normal|healthy: 0.5, cold|healthy: 0.4, dizzy|healthy: 0.1
        #                normal|fever: 0.1, cold|fever: 0.3, dizzy|fever: 0.6
        P_E = np.array([[0.5, 0.4, 0.1],
                       [0.1, 0.3, 0.6]])
        # trans prob: healthy->healthy: 0.7, healthy->fever: 0.3, fever->healthy: 0.4, fever->fever: 0.6
        P_T = np.array([[0.7, 0.3],
                        [0.4, 0.6]])

        p, P_res = pyACA.ToolViterbi(P_E, P_T, p_s)

        npt.assert_almost_equal(np.sum(np.abs(p - np.array([0, 0, 1]))), 0, decimal=7, err_msg="V 1: state sequence incorrect")
        npt.assert_almost_equal(np.sum(np.abs(P_res - np.array([[0.3000, 0.0840, 0.0059], [0.04, 0.0270, 0.0151]]))), 0, decimal=4, err_msg="V 1: state sequence incorrect")

        p, P_res = pyACA.ToolViterbi(P_E, P_T, p_s, True)

        npt.assert_almost_equal(np.sum(np.abs(p - np.array([0, 0, 1]))), 0, decimal=7, err_msg="V 1: state sequence incorrect")
        npt.assert_almost_equal(np.sum(np.abs(P_res - np.log(np.array([[0.3000, 0.0840, 0.0059], [0.04, 0.0270, 0.0151]])))), 0, decimal=2, err_msg="V 1: state sequence incorrect")

    def test_kmeans(self):
        mu = np.array([[-5, 5],
                       [5, -5]])
        iNumObs = 32
        phase = np.arange(0, iNumObs)*2*np.pi / iNumObs
        r = np.array([.1, .5])

        # generate data points for two clusters
        cluster1 = np.zeros([2, 2*iNumObs])
        cluster2 = np.zeros([2, 2*iNumObs])

        cluster1[:, 0:iNumObs] = mu[:, [0]] + r[0] * np.squeeze(np.array([[np.exp(1j*phase).real], [np.exp(1j*phase).imag]]))
        cluster1[:, iNumObs:2*iNumObs] = mu[:, [0]] + r[1] * np.squeeze(np.array([[np.exp(1j*phase).real], [np.exp(1j*phase).imag]]))

        cluster2[:, 0:iNumObs] = mu[:, [1]] + r[0] * np.squeeze(np.array([[np.exp(1j*phase).real], [np.exp(1j*phase).imag]]))
        cluster2[:, iNumObs:2*iNumObs] = mu[:, [1]] + r[1] * np.squeeze(np.array([[np.exp(1j*phase).real], [np.exp(1j*phase).imag]]))

        V = np.concatenate((cluster1, cluster2), axis=1)

        [clusterIdx, state] = pyACA.ToolSimpleKmeans(V, 2)

        self.assertEqual(np.sum(np.diff(clusterIdx[0:2*iNumObs])), 0, "KM 1: block content incorrect")
        self.assertEqual(np.sum(np.diff(clusterIdx[2*iNumObs:-1])), 0, "KM 2: block content incorrect")
        self.assertEqual(np.abs(clusterIdx[0]-clusterIdx[-1]), 1, "KM 3: block content incorrect")

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

    def test_freq2bin2freq(self):

        iUpsample = 10
        iFftLength = 256
        f_s = 8000
        bins = np.arange(0, 1281)/iUpsample
        fftres = f_s / iFftLength / iUpsample

        hzout = pyACA.ToolBin2Freq(bins, iFftLength, f_s)

        npt.assert_almost_equal(hzout[1]-hzout[0], fftres, decimal=7, err_msg="FBIN 1: frequency resolution incorrect")
        self.assertEqual(len(hzout), len(bins), "FBIN 2: output dimension incorrect")
        npt.assert_almost_equal(hzout[0], 0, decimal=7, err_msg="FBIN 3: frequency values incorrect")
        npt.assert_almost_equal(hzout[-1], f_s * 0.5, decimal=7, err_msg="FBIN 4: frequency values incorrect")

        # check back and forth conversion
        npt.assert_almost_equal(bins, pyACA.ToolFreq2Bin(pyACA.ToolBin2Freq(bins, iFftLength, f_s), iFftLength, f_s), decimal=7, err_msg="FBIN 5: bin to frequency to bin conversion incorrect")

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
