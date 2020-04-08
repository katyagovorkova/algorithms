from check_sorted import check_sorted

def counting_sort(l):
    """
    Takes as an input a list and returns it sorted with Counting sort algorithm,
    with keys from 0 to 9 including
    """
    counts = [0]*10 # reserve list with counts for each key

    for element in l:
        counts[element] += 1

    sorted_l = [ i for i in range(len(counts)) for _ in range(counts[i]) ]

    return sorted_l


if __name__=='__main__':
    # tests
    check_sorted(counting_sort([]))
    check_sorted(counting_sort([0]))
    check_sorted(counting_sort([1,2,3,4,0]))
    check_sorted(counting_sort([1,2,3,4,0,0,1,2,3,4]))