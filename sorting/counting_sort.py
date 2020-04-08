def counting_sort(l):
    """
    Takes as an input a list and returns it sorted with Counting sort algorithm,
    with keys from 0 to 9 including
    """
    print('Sorting list {}...'.format(l))

    counts = [0]*10 # reserve list with counts for each key

    for element in l:
        counts[element] += 1

    sorted_l = [ i for i in range(len(counts)) for _ in range(counts[i]) ]


    print('Sorted list {}'.format(sorted_l))
    return sorted_l

if __name__=='__main__':
    # tests
    print([]==counting_sort([]))
    print([0]==counting_sort([0]))
    print([0,1,2,3,4]==counting_sort([1,2,3,4,0]))
    print([0,1,2,3,4]==counting_sort([1,2,3,4,0]))
    print([0,0,1,1,2,2,3,3,4,4]==counting_sort([1,2,3,4,0,0,1,2,3,4]))