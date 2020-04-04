def binary_search(m, sorted_list):
    """
    Return first found position of the number m in sorted list sorted_list.
    If number is not in the list, returns None.
    """
    print('Looking for {} in {}...'.format(m, sorted_list))

    low = 0
    high = len(sorted_list) - 1

    while low <= high:
        # find the middle
        median = (high - low) // 2 + low
        if m == sorted_list[median]:
            print('found element at position {} \n'.format(median))
            return True
        if m > sorted_list[median]:
            low = median + 1
        else:
            high = median - 1

    print('not found \n')
    return False


if __name__=='__main__':
    # test
    binary_search(5, [])
    binary_search(5, [5])
    binary_search(5, list(range(6)))
    binary_search(5, list(range(10)))
    binary_search(5, [0,1,2,3,6,7])
    binary_search(5, [5,5,5,5,5])
    binary_search(59, list(range(100)))