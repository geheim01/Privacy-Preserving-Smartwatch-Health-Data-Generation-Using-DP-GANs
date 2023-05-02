from typing import List


# most frequent element in list
def most_common(lst):
    return max(set(lst), key=lst.count)


# if stress occurress in time interval return 1
def get_max_value_from_list(lst: List[int]) -> int:
    return max(set(lst))
