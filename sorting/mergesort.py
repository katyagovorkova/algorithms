from check_sorted import check_sorted


def merge(A, B):
    """
    Takes as an input two sorted lists and returns merged sorted list.
    """
    i, k = 0, 0 # reserve incidences to loop over A and B
    merged = [] # initialize merged list
    while i < len(A) and k < len(B):
        if A[i] <= B[k]:
            merged.append(A[i])
            i += 1
        else:
            merged.append(B[k])
            k += 1
    # now we might have leftovers in either A or B
    while i < len(A):
        merged.append(A[i])
        i += 1
    while k < len(B):
        merged.append(B[k])
        k += 1

    return merged


def mergesort(l):
    """
    Takes as an input a list and returns it sorted with Mergesort algorithm.
    """
    n = len(l)

    if n <= 1:
        return l

    middle = n//2
    left = l[:middle]
    right = l[middle:]

    mergesort(left)
    mergesort(right)

    sorted_l = merge(left, right)

    for i in range(n):
        l[i] = sorted_l[i]

    return l


if __name__=='__main__':
    # tests
    check_sorted(mergesort([]))
    check_sorted(mergesort([0]))
    check_sorted(mergesort([1,2,3,4,0]))
    check_sorted(mergesort([1,2,3,4,0,0,1,2,3,4]))
    check_sorted(mergesort(list(range(100,-1,-1))))