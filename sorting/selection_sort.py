def selection_sort(l):
    """
    Takes as an input a list and returns it sorted with Selection sort algorithm.
    """
    print('Sorting list {}...'.format(l))
    n = len(l)

    for i in range(n-1): # to (n-1) because last element is sorted automatically
        for k in range(i+1, n):
            if l[i]>l[k]:
                l[i], l[k] = l[k], l[i]

    print('Sorted list {}'.format(l))
    return l

if __name__=='__main__':
    # tests
    print([]==selection_sort([]))
    print([0]==selection_sort([0]))
    print([0,1,2,3,4]==selection_sort([1,2,3,4,0]))
    print(list(range(101))==selection_sort(list(range(100,-1,-1))))