def insertion_sort(l):
    """
    Takes as an input a list and returns it sorted with Insertion sort algorithm.
    """
    print('Sorting list {}...'.format(l))
    n = len(l)

    for top in range(n):
        k = top
        while k>0 and l[k-1] > l[k]:
            l[k-1], l[k] = l[k], l[k-1]
            k -= 1

    print('Sorted list {}'.format(l))
    return l

if __name__=='__main__':
    # tests
    print([]==insertion_sort([]))
    print([0]==insertion_sort([0]))
    print([0,1,2,3,4]==insertion_sort([1,2,3,4,0]))
    print(list(range(101))==insertion_sort(list(range(100,-1,-1))))