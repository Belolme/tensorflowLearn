def getIndexWithOp(input_t, validate_t, validate_v, op):
    """
    get index in set of input_t and validate_t equal validate_v and op is True
    """
    result = -1
    for i, validate_value in enumerate(validate_t):
        if validate_value == validate_v:
            if result == -1:
                result = i
            elif op(input_t[i], input_t[result]):
                result = i

    return result


def getMaxIndex(input_t, validate_t, validate_v):
    return getIndexWithOp(input_t, validate_t, validate_v, lambda x, y: x > y)


def getMinIndex(input_t, validate_t, validate_v):
    return getIndexWithOp(input_t, validate_t, validate_v, lambda x, y: x < y)
