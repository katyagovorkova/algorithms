class LinkedList:
    """
    Each element of a linked list contains a link to the next element of the list.
    To access the N-th element of a list, O(N) operations is required.
    However, inserting new element is easier than array since list elements
    can be in a random places in memory.
    """
    def __init__(self):
        self._begin = None

    def insert(self, x):
        # Add new element to the list
        self._begin = [x, self._begin]

    def pop(self):
        # First check that the list is not empty
        assert self._begin is not None, 'The linked list is empty'
        # Remove last element in the list and return it
        x = self._begin[0]
        self._begin = self._begin[1]
        return x


if __name__=='__main__':
    # tests
    ll = LinkedList()
    ll.insert(10)
    ll.insert(0)
    print(ll)
    print(ll.pop())
    print(ll.pop())
    print(ll.pop())