![GitHub top language](https://img.shields.io/github/languages/top/alexanderlerch/pyACA)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyACA)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/alexanderlerch/pyACA)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6310329.svg)](https://doi.org/10.5281/zenodo.6310329)
![GitHub issues](https://img.shields.io/github/issues-raw/alexanderlerch/pyACA)
![GitHub last commit](https://img.shields.io/github/last-commit/alexanderlerch/pyACA)
![GitHub](https://img.shields.io/github/license/alexanderlerch/pyACA)

# pyACA
Python scripts accompanying the book ["An Introduction to Audio Content Analysis"](https://www.AudioContentAnalysis.org). The source code shows example implementations of basic approaches, features, and algorithms for music audio content analysis.

All implementations are also available in:
* [Matlab: ACA-Code](https://github.com/alexanderlerch/ACA-Code)
* [C++: libACA](https://github.com/alexanderlerch/libACA)


## functionality
The top-level functions are (alphabetical):
> - [`computeBeatHisto`](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/computeBeatHisto.py): calculates a simple beat histogram
> - [`computeChords`](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/computeChords.py): simple chord recognition
> - [`computeFeature`](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/computeFeature.py): calculates instantaneous features 
> - [`computeFingerprint`](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/computeFingerprint.py): audio fingerprint extraction 
> - [`computeKey`](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/computeKey.py): calculates a simple key estimate
> - [`computeMelSpectrogram`](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/computeMelSpectrogram.py): computes a mel spectrogram
> - [`computeNoveltyFunction`](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/computeNoveltyFunction.py): simple onset detection
> - [`computePitch`](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/computePitch.py): calculates a fundamental frequency estimate
> - [`computeSpectrogram`](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/computeSpectrogram.py): computes a magnitude spectrogram

The names of the additional functions follow the following 
conventions:
> - `Feature`*: instantaneous features
> - `Pitch`*: pitch tracking approach
> - `Novelty`*: novelty function computation
> - `Tool`*: additional helper functions and basic algorithms such as 
>   - [Blocking](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolBlockAudio.py) of audio into overlapping blocks
>   - [Pre-processing](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolPreprocAudio.py) audio
>   - Conversion ([freq2bark](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolFreq2Bark.py), [freq2mel](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolFreq2Mel.py), [freq2midi](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolFreq2Midi.py), [mel2freq](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/Mel2Freq.py), [midi2freq](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolMidi2Freq.py))
>   - Filterbank ([Gammatone](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolGammatoneFb.py))
>   - [Gaussian Mixture Model](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolGmm.py)
>   - [Principal Component Analysis](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolPca.py)
>   - [Feature Selection](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolSeqFeatureSel.py)
>   - [Dynamic Time Warping](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolSimpleDtw.py)
>   - [K-Means Clustering](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolSimpleKmeans.py)
>   - [K Nearest Neighbor classification](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolSimpleKnn.py)
>   - [Non-Negative Matrix Factorization](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolSimpleNmf.py)
>   - [Viterbi](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolViterbi.py) algorithm

## documentation
The latest full documentation of this package can be found at [https://alexanderlerch.github.io/pyACA](https://alexanderlerch.github.io/pyACA).

## design principles
Please note that the provided code examples are only _intended to showcase 
algorithmic principles_ – they are not entirely suitable for practical usage without 
parameter optimization and additional algorithmic tuning. Rather, they intend to show how to implement audio analysis solutions and to facilitate algorithmic understanding to enable the reader to design and implement their own analysis approaches. 

### minimal dependencies
The _required dependencies_ are reduced to a minimum, more specifically to only [numpy](https://numpy.org/) and [scipy](https://scipy.org/), for the following reasons:
* accessibility, i.e., clear algorithmic implementation from scratch without obfuscation by using 3rd party implementations,
* maintainability through independence of 3rd party code. 
This design choice brings, however, some limitations; for instance, reading of non-RIFF audio files is not supported and the machine learning models are very simple.  

### readability
Consistent variable naming and formatting, as well as the choice for simple implementations allow for easier parsing.
The readability of the source code will sometimes come at the cost of lower performance.

### cross-language comparability
All code is matched exactly with [Matlab implementations](https://www.github.com/alexanderlerch/ACA-Code) and the equations in the book. This also means that the python code might **violate typical python style conventions** in order to be consistent.

## related repositories and links
The python source code in this repository is matched with corresponding source code in the [Matlab repository](https://www.github.com/alexanderlerch/ACA-Code).

Other, _related repositories_ are
* [ACA-Slides](https://www.github.com/alexanderlerch/ACA-Slides): slide decks for teaching and learning audio content analysis
* [ACA-Plots](https://www.github.com/alexanderlerch/ACA-Plots): Matlab scripts for generating all plots in the book and slides

The _main entry point_ to all book-related information is [AudioContentAnalysis.org](https://www.AudioContentAnalysis.org)

## getting started
### installation
```console
pip install pyACA 
```

### code examples

**example 1**: computation and plot of the _Spectral Centroid_

```python
import pyACA
import matplotlib.pyplot as plt 

# file to analyze
cPath = "c:/temp/test.wav"

# extract feature
[v, t] = pyACA.computeFeatureCl(cPath, "SpectralCentroid")

# plot feature output
plt.plot(t,np.squeeze(v))
```
**example 2**: Computation of two features (here: _Spectral Centroid_ and _Spectral Flux_)

```python
import pyACA

# read audio file
cPath = "c:/temp/test.wav"
[f_s, afAudioData] = pyACA.ToolReadAudio(cPath)

# compute feature
[vsc, t] = pyACA.computeFeature("SpectralCentroid", afAudioData, f_s)
[vsf, t] = pyACA.computeFeature("SpectralFlux", afAudioData, f_s)
```


