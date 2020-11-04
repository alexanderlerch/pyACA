#
name = "pyACA"
from .computeBeatHisto import computeBeatHisto
from .computeFeature import computeFeature
from .computeKey import computeKey
from .computeNoveltyFunction import computeNoveltyFunction
from .computePitch import computePitch
from .computeBeatHisto import computeBeatHistoCl
from .computeFeature import computeFeatureCl
from .computeKey import computeKeyCl
from .computeNoveltyFunction import computeNoveltyFunctionCl
from .computePitch import computePitchCl
from .ToolSimpleDtw import ToolSimpleDtw
from .ToolReadAudio import ToolReadAudio
from .ToolPreprocAudio import ToolPreprocAudio
from .ToolBlockAudio import ToolBlockAudio
from .FeatureSpectralCentroid import FeatureSpectralCentroid
from .FeatureSpectralCrestFactor import FeatureSpectralCrestFactor
from .FeatureSpectralDecrease import FeatureSpectralDecrease
from .FeatureSpectralFlatness import FeatureSpectralFlatness
from .FeatureSpectralFlux import FeatureSpectralFlux
from .FeatureSpectralKurtosis import FeatureSpectralKurtosis
from .FeatureSpectralMfccs import FeatureSpectralMfccs
from .FeatureSpectralPitchChroma import FeatureSpectralPitchChroma
from .FeatureSpectralRolloff import FeatureSpectralRolloff
from .FeatureSpectralSkewness import FeatureSpectralSkewness
from .FeatureSpectralSlope import FeatureSpectralSlope
from .FeatureSpectralSpread import FeatureSpectralSpread
from .FeatureSpectralTonalPowerRatio import FeatureSpectralTonalPowerRatio
from .FeatureTimeAcfCoeff import FeatureTimeAcfCoeff
from .FeatureTimeMaxAcf import FeatureTimeMaxAcf
from .FeatureTimePeakEnvelope import FeatureTimePeakEnvelope
from .FeatureTimeRms import FeatureTimeRms
from .FeatureTimeStd import FeatureTimeStd
from .FeatureTimeZeroCrossingRate import FeatureTimeZeroCrossingRate
from .NoveltyFlux import NoveltyFlux
from .NoveltyHainsworth import NoveltyHainsworth
from .NoveltyLaroche import NoveltyLaroche
from .PitchSpectralAcf import PitchSpectralAcf
from .PitchSpectralHps import PitchSpectralHps
from .PitchTimeAcf import PitchTimeAcf
from .PitchTimeAmdf import PitchTimeAmdf
from .PitchTimeAuditory import PitchTimeAuditory
from .PitchTimeZeroCrossings import PitchTimeZeroCrossings
from .getFeatureList import getFeatureList

features = [
    "SpectralCentroid",
    "SpectralCrestFactor",
    "SpectralDecrease",
    "SpectralFlatness",
    "SpectralFlux",
    "SpectralKurtosis",
    "SpectralMfccs",
    "SpectralPitchChroma",
    "SpectralRolloff",
    #"SpectralSkewness",
    "SpectralSlope",
    "SpectralTonalPowerRatio",
    #"TimeAcfCoeff",
    #"TimeMaxAcf",
    #"TimePeakEnvelope",
    #"TimeRms",
    #"TimeStd",
    #"TimeZeroCrossingRate"
]