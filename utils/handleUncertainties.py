def handleUncertainties(value, err):
    lower, upper = [], []
    for i in range(0, len(err)):
        if (value[i] + err[i] > 1):
            upper.append(1 - value[i])
        else:
            upper.append(err[i])
        if (value[i] - err[i] < 0):
            lower.append(value[i])
        else:
            lower.append(err[i])
    return [lower, upper]
        
