![GitHub top language](https://img.shields.io/github/languages/top/alexanderlerch/pyACA)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyACA)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/alexanderlerch/pyACA)
![GitHub issues](https://img.shields.io/github/issues-raw/alexanderlerch/pyACA)
[![CodeFactor](https://www.codefactor.io/repository/github/alexanderlerch/pyaca/badge)](https://www.codefactor.io/repository/github/alexanderlerch/pyaca)
![GitHub last commit](https://img.shields.io/github/last-commit/alexanderlerch/pyACA)
![GitHub](https://img.shields.io/github/license/alexanderlerch/pyACA)

# pyACA
Python scripts accompanying the book "An Introduction to Audio Content 
Analysis" (www.AudioContentAnalysis.org). The source code shows example implementations of basic approaches, features, and algorithms for music audio content analysis.

## Functionality

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
>  - [Blocking](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolBlockAudio.py) of audio into overlapping blocks
>  - [Pre-processing](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolPreprocAudio.py) audio
>  - Conversion ([freq2bark](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolFreq2Bark.py), [freq2mel](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolFreq2Mel.py), [freq2midi](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolFreq2Midi.py), [mel2freq](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/Mel2Freq.py), [midi2freq](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolMidi2Freq.py))
>  - Filterbank ([Gammatone](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolGammatoneFb.py))
>  - [Gaussian Mixture Model](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolGmm.py)
>  - [Principal Component Analysis](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolPca.py)
>  - [Feature Selection](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolSeqFeatureSel.py)
>  - [Dynamic Time Warping](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolSimpleDtw.py)
>  - [K-Means Clustering](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolSimpleKmeans.py)
>  - [K Nearest Neighbor classification](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolSimpleKnn.py)
>  - [Non-Negative Matrix Factorization](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolSimpleNmf.py)
>  - [Viterbi](https://github.com/alexanderlerch/pyACA/blob/master/pyACA/ToolViterbi.py) algorithm


## Design principles
Please note that the provided code examples are only _intended to showcase 
algorithmic principles_ – they are not suited for practical usage without 
parameter optimization and additional algorithmic tuning. Rather, they intend to show how to implement audio analysis solutions from scratch and to facilitate algorithmic understanding to enable the reader to design and implement their own analysis approaches. Furthermore,
the python code might **violate typical python style conventions** in order to
be consistent with the Matlab code (see below).
### Readability
### Cross-language comparability
### Dependencies
The _required dependencies_ are reduced to a minimum to improve maintainability and increase accessibility. More specifically, the only common dependencies are [numpy](https://numpy.org/) and [scipy](https://scipy.org/). This design choice brings some limitations; for instance, reading of non-RIFF audio files is not supported.  

## Related repositories and links
The python source code in this repository is matched with corresponding source code in the [Matlab repository](https://www.github.com/alexanderlerch/ACA-Code).

Other, _related repositories_ are
* [ACA-Slides](https://www.github.com/alexanderlerch/ACA-Slides): slide decks for teaching and learning audio content analysis
* [ACA-Plots](https://www.github.com/alexanderlerch/ACA-Plots): Matlab scripts for generating all plots in the book and slides

The _main entry point_ to all book-related information is [AudioContentAnalysis.org](https://www.AudioContentAnalysis.org)

## Getting started
### Installation
**TODO double check - these are only place holders!**

For developers working on local clone, `cd` to the repo and replace `pyACA` with `.`. 

```console
pip install pyACA 
```
**Running tests**
```
pip install .[tests]
pytest tests/ --cov=pyACA
```

### Code examples

**Example 1**: Computation and plot of the _Spectral Centroid_

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
**Example 2**: Computation of two features (here: _Spectral Centroid_ and _Spectral Flux_)

```python
import pyACA

# read audio file
cPath = "c:/temp/test.wav"
[f_s, afAudioData] = pyACA.ToolReadAudio(cPath)

# compute feature
[vsc, t] = pyACA.computeFeature("SpectralCentroid", afAudioData, f_s)
[vsf, t] = pyACA.computeFeature("SpectralFlux", afAudioData, f_s)
```


