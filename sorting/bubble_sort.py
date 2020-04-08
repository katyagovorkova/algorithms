from check_sorted import check_sorted

def bubble_sort(l):
    """
    Takes as an input a list and returns it sorted with Bubble sort algorithm.
    """
    n = len(l)

    for top in range(1,n):
        for k in range(0,n-top):
            if l[k] > l[k+1]:
                l[k], l[k+1] = l[k+1], l[k]
    return l


if __name__=='__main__':
    # tests
    check_sorted(bubble_sort([]))
    check_sorted(bubble_sort([0]))
    check_sorted(bubble_sort([1,2,3,4,0]))
    check_sorted(bubble_sort([1,2,3,4,0,0,1,2,3,4]))
    check_sorted(bubble_sort(list(range(100,-1,-1))))