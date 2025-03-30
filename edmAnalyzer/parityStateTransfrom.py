import numpy as np
import itertools
def parityStateTransform(channelName= ['N','E','B']):
    """
    Note that the matrix returned from parityStateTransform should be applied in this way:

    $
    \phi_{P} = \mathbf{transformMatrix} \times\phi_{S}
    $

    while all vectors should be column vectors
    """

    def stateLabel(channelName=channelName):
        length = len(channelName)
        elements = [+1, -1]
        combinations = itertools.product(elements, repeat=length)
        return [tuple(combination) for combination in combinations]
    def parityLabel(channelName=channelName):
        parityChannelName = ['nr']
        combinations = [()]
        for r in range(1, len(channelName) + 1):
            for combination in itertools.combinations(channelName, r):
                parityChannelName.append(''.join(map(str,combination)))
                combinations.append(combination)
        return parityChannelName, combinations
    parityLabels, parityLabelsCombinations = parityLabel(channelName=channelName)
    stateLabels = stateLabel(channelName=channelName)
    transformMatrix = []
    for flippedIndex in parityLabelsCombinations:
        scalingFactor = 1.0 / len(stateLabels)
        p = scalingFactor * np.ones((2 ** (len(channelName))))
        for eachFlippedIndex in flippedIndex:
            for i in range(len(stateLabels)):
                p[i] = p[i]*(-1 if stateLabels[i][channelName.index(eachFlippedIndex)] == -1 else 1)
        transformMatrix.append(p)
    prefix = ''
    for i in channelName:
        prefix = prefix+i
    prefix = prefix+'='
    for i in range(len(stateLabels)):
        stateLabels[i] = stateLabels[i]
    return np.array(transformMatrix), parityLabels, stateLabels

def combine_switches(list1, list2):
    """
    list1: superblock switches
    list2: block switches
    """
    def multiply_symbols(symbol1, symbol2):
        if symbol1 == 'nr':
            return symbol2
        elif symbol2 == 'nr':
            return symbol1
        return symbol2 + symbol1
    
    # Create the 2D matrix by applying the multiplication rule
    matrix = [[multiply_symbols(a, b) for b in list2] for a in list1]
    
    # Flatten the 2D matrix into a 1D list
    flattened_list = [item for sublist in matrix for item in sublist]
    
    return flattened_list, matrix