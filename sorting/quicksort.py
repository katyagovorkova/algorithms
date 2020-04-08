def quicksort(l):
    """
    Takes as an input a list and returns it sorted with Quicksort algorithm.
    """
    print('Sorting list {}...'.format(l))


if __name__=='__main__':
    # tests
    print([]==insertion_sort([]))
    print([0]==insertion_sort([0]))
    print([0,1,2,3,4]==insertion_sort([1,2,3,4,0]))
    print(list(range(101))==insertion_sort(list(range(100,-1,-1))))