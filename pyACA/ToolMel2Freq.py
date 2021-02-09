"""
helper function: convert Mel to Hz scale

  Args:
    # fMel: Mel frequency
    # cModel: The name of the model ('fant' [default], 'shaughnessy', 'umesh')

  Returns:
    # Frequency in Hertz
"""

def ToolMel2Freq(fMel, cModel='fant'):
    cModel = cModel.lower()
    if cModel == 'fant':
        return acaFant_I(fMel)
    elif cModel == 'shaughnessy':
        return acaShaughnessy_I(fMel)
    elif cModel == 'umesh':
        return acaUmesh_I(fMel)
    else:
        print('Invalid model type')


def acaFant_I(m):
    # mel = 1000 * log2(1 + f / 1000);
    return 1000 * (2 ** (m / 1000) - 1)


def acaShaughnessy_I(m):
    # mel = 2595 * log10(1 + f / 700);
    return 700 * (10 ** (m / 2595) - 1)


def acaUmesh_I(m):
    # mel = f / (2.4e-4 * f + 0.741);
    return m * 0.741 / (1 - m * 2.4e-4)
