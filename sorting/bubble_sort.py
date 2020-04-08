def bubble_sort(l):
    """
    Takes as an input a list and returns it sorted with Bubble sort algorithm.
    """
    print('Sorting list {}...'.format(l))
    n = len(l)

    for top in range(1,n):
        for k in range(0,n-top):
            if l[k] > l[k+1]:
                l[k], l[k+1] = l[k+1], l[k]

    print('Sorted list {}'.format(l))
    return l

if __name__=='__main__':
    # tests
    print([]==bubble_sort([]))
    print([0]==bubble_sort([0]))
    print([0,1,2,3,4]==bubble_sort([1,2,3,4,0]))
    print(list(range(101))==bubble_sort(list(range(100,-1,-1))))