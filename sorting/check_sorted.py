def check_sorted(l, ascending=True):
    """
    Check if given list is sorted with runtime complexity O(n).
    By default checks for ascending order, can do descending as well.
    """
    print('Is {} sorted...'.format(l), end='')
    n = len(l)
    sign = 2*int(ascending)+1

    for i in range(n-1):
        if sign*l[i] > sign*l[i+1]:
            print('nope')
            return

    print('yes!')