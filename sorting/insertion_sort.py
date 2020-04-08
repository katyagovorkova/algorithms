from check_sorted import check_sorted

def insertion_sort(l):
    """
    Takes as an input a list and returns it sorted with Insertion sort algorithm.
    """
    n = len(l)

    for top in range(n):
        k = top
        while k>0 and l[k-1] > l[k]:
            l[k-1], l[k] = l[k], l[k-1]
            k -= 1

    return l


if __name__=='__main__':
    # tests
    check_sorted(insertion_sort([]))
    check_sorted(insertion_sort([0]))
    check_sorted(insertion_sort([1,2,3,4,0]))
    check_sorted(insertion_sort([1,2,3,4,0,0,1,2,3,4]))
    check_sorted(insertion_sort(list(range(100,-1,-1))))