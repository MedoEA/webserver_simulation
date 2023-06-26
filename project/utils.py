def max_(values):
    max_index = 0
    maxv = values[max_index]
    for i in range(len(values)):
        if values[i] > maxv:
            maxv = values[i]
            max_index = i
    return max_index
