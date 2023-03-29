import numpy as np
import torch

def initializeEpsilonWeightsMask(text,epsilon, noRows, noCols):
    # generate an epsilon based Erdos Renyi sparse weights mask
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)
    mask_weights = np.random.rand(noRows, noCols)
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    sparsity = 1-noParameters/(noRows * noCols)
    print("Epsilon Sparse Initialization ",text,": Epsilon ",epsilon,"; Sparsity ",sparsity,"; NoParameters ",noParameters,"; NoRows ",noRows,"; NoCols ",noCols,"; NoDenseParam ",noRows*noCols)
    print (" OutDegreeBottomNeurons %.2f ± %.2f; InDegreeTopNeurons %.2f ± %.2f" % (np.mean(mask_weights.sum(axis=1)),np.std(mask_weights.sum(axis=1)),np.mean(mask_weights.sum(axis=0)),np.std(mask_weights.sum(axis=0))))
    return [noParameters, mask_weights.transpose()]

def initializeSparsityLevelWeightMask(text,sparsityLevel,noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    prob=sparsityLevel
    mask_weights = np.random.rand(noRows, noCols)
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    sparsity = 1-noParameters/(noRows * noCols)
    epsilon = int((prob*(noRows * noCols)-1)/(noRows + noCols))
    print("Sparsity Level Initialization ",text,": Computed Epsilon ",epsilon,"; Sparsity ",sparsity,"; NoParameters ",noParameters,"; NoRows ",noRows,"; NoCols ",noCols,"; NoDenseParam ",noRows*noCols)
    print (" OutDegreeBottomNeurons %.2f ± %.2f; InDegreeTopNeurons %.2f ± %.2f" % (np.mean(mask_weights.sum(axis=1)),np.std(mask_weights.sum(axis=1)),np.mean(mask_weights.sum(axis=0)),np.std(mask_weights.sum(axis=0))))
    return [noParameters, mask_weights.transpose()]

def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx

def changeConnectivitySET(weights, noWeights, initMask, zeta, theta, lastTopologyChange, iteration):
    print(f'# weights before connectivity change {noWeights}')
    # change Connectivity
    # remove zeta largest negative and smallest positive weights
    weights = weights * initMask
    values = np.sort(weights.ravel())
    firstZeroPos = find_first_pos(values, 0)
    lastZeroPos = find_last_pos(values, 0)
    largestNegative = values[int((1 - zeta) * firstZeroPos)]
    smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + zeta * (values.shape[0] - lastZeroPos)))]
    rewiredWeights = weights.copy()
    rewiredWeights[rewiredWeights > smallestPositive] = 1
    rewiredWeights[rewiredWeights < largestNegative] = 1
    rewiredWeights[rewiredWeights != 1] = 0

    # add random weights
    nrAdd = 0
    if not lastTopologyChange:
        noRewires = (noWeights - np.sum(rewiredWeights)) * theta
        while (nrAdd < noRewires):
            i = np.random.randint(0, rewiredWeights.shape[0])
            j = np.random.randint(0, rewiredWeights.shape[1])
            if (rewiredWeights[i, j] == 0):
                rewiredWeights[i, j] = 1
                nrAdd += 1

    ascStats=[iteration, nrAdd, noWeights, np.count_nonzero(rewiredWeights)]

    return [rewiredWeights,ascStats]

def changeConnectivityXReLU(weights, noWeights, initMask, lastTopologyChange, iteration):
    weights = weights * initMask

    weightspos = weights.copy()
    weightspos[weightspos < 0] = 0
    strengthpos = np.sum(weightspos, axis=0)

    weightsneg = weights.copy()
    weightsneg[weightsneg > 0] = 0
    strengthneg = np.sum(weightsneg, axis=0)

    for j in range(strengthpos.shape[0]):
        if (strengthpos[j] + strengthneg[j] < 0):
            difference = strengthpos[j]
            iis = np.nonzero(weightsneg[:, j])[0]
            ww = weightsneg[iis, j]
            iisort = np.argsort(ww)
            for i in iisort:
                if (difference > 0):
                    difference += weightsneg[iis[i], j]
                else:
                    weights[iis[i], j] = 0
    rewiredWeights = weights.copy();
    rewiredWeights[rewiredWeights != 0] = 1;

    nrAdd = 0
    if (lastTopologyChange == False):
        noRewires = noWeights - np.sum(rewiredWeights)
        while (nrAdd < noRewires):
            i = np.random.randint(0, rewiredWeights.shape[0])
            j = np.random.randint(0, rewiredWeights.shape[1])
            if (rewiredWeights[i, j] == 0):
                rewiredWeights[i, j] = 1
                nrAdd += 1

    ascStats = [iteration, nrAdd, noWeights, np.count_nonzero(rewiredWeights)]
    return [rewiredWeights, ascStats]


# def adaptSparseConnectivity(model, lastTopologyChangemodel, total_it, ascTopologyChangePeriod, zeta, theta, device):
#     if not lastTopologyChangemodel:
#         if total_it % ascTopologyChangePeriod == 2:
#             [model.mask1, ascStats1] = zeta.changeConnectivitySET(model.l1.weight.data.cpu().numpy(),
#                                                                       model.noPar1, model.mask1,
#                                                                       zeta, theta,
#                                                                       lastTopologyChangemodel, total_it)
#             model.torchMask1 = torch.from_numpy(model.mask1).float().to(device)
#             [model.mask2, ascStats2] = zeta.changeConnectivitySET(model.l2.weight.data.cpu().numpy(),
#                                                                       model.noPar2, model.mask2,
#                                                                       zeta, theta,
#                                                                       lastTopologyChangemodel, total_it)
#             model.torchMask2 = torch.from_numpy(model.mask2).float().to(device)
#             [model.mask4, ascStats4] = zeta.changeConnectivitySET(model.l4.weight.data.cpu().numpy(),
#                                                                       model.noPar4, model.mask4,
#                                                                       zeta, theta,
#                                                                       lastTopologyChangemodel, total_it)
#             model.torchMask4 = torch.from_numpy(model.mask4).float().to(device)
#             [model.mask5, ascStats5] = zeta.changeConnectivitySET(model.l5.weight.data.cpu().numpy(),
#                                                                       model.noPar5, model.mask5,
#                                                                       zeta, theta,
#                                                                       lastTopologyChangemodel, total_it)
#             model.torchMask5 = torch.from_numpy(model.mask5).float().to(device)
#
#     # Maintain the same sparse connectivity for model
#     model.l1.weight.data.mul_(model.torchMask1)
#     model.l2.weight.data.mul_(model.torchMask2)
#     model.l4.weight.data.mul_(model.torchMask4)
#     model.l5.weight.data.mul_(model.torchMask5)
