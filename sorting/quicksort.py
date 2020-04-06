def quicksort(l):
    """
    Takes as an input a list and returns it sorted with Quicksort method.
    """
    print('Sorting list {}...'.format(l))


if __name__=='__main__':
    # tests
    print([]==insertion_sort([]))
    print([0]==insertion_sort([0]))
    print(list(range(100,-1,1))==insertion_sort(list(range(100))))