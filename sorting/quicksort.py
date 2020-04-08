from check_sorted import check_sorted

def quicksort(l):
    """
    Takes as an input a list and returns it sorted with Quicksort algorithm.
    """
    n = len(l)
    if n <= 1:
        return l

    barrier = l[0] # usually random is used though
    left = [i for i in l if i < barrier]
    middle = [i for i in l if i == barrier]
    right = [i for i in l if i > barrier]

    quicksort(left)
    quicksort(right)

    sorted_l = left + middle + right

    for i in range(n):
        l[i] = sorted_l[i]

    return l


if __name__=='__main__':
    # tests
    check_sorted(quicksort([]))
    check_sorted(quicksort([0]))
    check_sorted(quicksort([1,2,3,4,0]))
    check_sorted(quicksort([1,2,3,4,0,0,1,2,3,4]))
    check_sorted(quicksort(list(range(100,-1,-1))))