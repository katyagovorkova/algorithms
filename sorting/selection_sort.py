from check_sorted import check_sorted

def selection_sort(l):
    """
    Takes as an input a list and returns it sorted with Selection sort algorithm.
    """
    n = len(l)

    for i in range(n-1): # to (n-1) because last element is sorted automatically
        for k in range(i+1, n):
            if l[i]>l[k]:
                l[i], l[k] = l[k], l[i]
    return l


if __name__=='__main__':
    # tests
    check_sorted(selection_sort([]))
    check_sorted(selection_sort([0]))
    check_sorted(selection_sort([1,2,3,4,0]))
    check_sorted(selection_sort([1,2,3,4,0,0,1,2,3,4]))
    check_sorted(selection_sort(list(range(100,-1,-1))))