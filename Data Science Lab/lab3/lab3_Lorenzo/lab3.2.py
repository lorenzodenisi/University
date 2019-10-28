# apriori implementation

import json

toy_ds = [['a', 'b'], ['b', 'c', 'd'], ['a', 'c', 'd', 'e'], ['a', 'd', 'e'], ['a', 'b', 'c'], ['a', 'b', 'c', 'd'],
          ['b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'd'], ['b', 'c', 'e']]


def main():
    '''
    res = apriori(toy_ds, 0.1)

    for r in res:
        for occ in r:
            print(str(occ[0]) + ' --> ' + str(occ[1]))
    '''
    with open('modified_coco.json', 'r') as f:
        datastore = json.load(f)
        dataset = {}
        for key in datastore[0].keys():
            dataset[key] = []

        for row in datastore:
            for key, value in zip(row.keys(), row.values()):
                dataset[key].append(value)

        res = apriori(dataset['annotations'], minsup=0.02)
        for r in res:
            for occ in r:
                print(str(occ[0]) + ' --> ' + str(occ[1]))


def apriori(dataset, minsup):
    minsup *= len(dataset)  # I want the occurrencies, in absolute terms, not the ratio

    # getting max dimension of set
    k_max = 0
    for row in dataset:
        if len(row) > k_max:
            k_max = len(row)

    # getting unique set of the elements
    unique_elem = []
    for row in dataset:
        for item in row:
            if item not in unique_elem:
                unique_elem.append(item)

    print("Max number of elements is" + str(k_max))

    # init L and C
    L = [[]] * k_max
    C = [[]] * k_max

    # Getting starting L
    L[0] = get_occurrencies(unique_elem, dataset)
    print(L[0])

    for k in range(0, k_max):
        # I put in C the combinations already pruned
        C[k + 1] = get_combinations(get_col(L[0], 0), k + 2, L)
        print("Found " + str(len(C[k + 1])) + " combinations of " + str(k + 2) + " elements")
        # C become a list of tuples, where the first element is a set and the second is the number of occurrencies in the dataset
        C[k + 1] = get_occurrencies(C[k + 1], dataset)

        to_pop = []
        for i, c in enumerate(C[k + 1]):
            if c[1] <= int(minsup):  # Delete elements that occur less than minsup times
                to_pop.append(i)

        for i in reversed(to_pop):  # reversed because popping in the right direction causes index out of range
            C[k + 1].pop(i)

        L[k + 1] = C[k + 1]  # saving C to L
        print(L[k + 1])

        if len(get_col(L[k + 1], 1)) == 0:  # stop if the set in empty
            break
    return L


def get_combinations(set, dim, blacklists):
    '''
    :param set: set of unique elements
    :param dim: dimension of final combinations
    :param blacklists: subsets to avoid, useful for pruning
    :return: list of combinations
    '''

    combinations = []
    __get_combinations_recursive(set, dim, [''] * dim, 0, combinations, blacklists)
    return combinations


def __get_combinations_recursive(set, dim, current, index, combinations, blacklists):
    '''
    recursive part of get_combinations
    :param set: same as get_combinations
    :param dim: same as get_combinations
    :param current: current combination
    :param index: current index
    :param combinations: same as get_combinations
    :param blacklists: same as get_combinations
    :return:
    '''
    if index == dim:  # stop condition
        combinations.append(current)
        return

    if index > 0:
        if not is_present([list(current[:index])], get_col(blacklists[index - 1], 0)):  # Pruning on previous sequences
            return

        last = current[
            index - 1]  # setting the start index, avoiding to pick an already picked element of set (or an element not in lexical order)
        start = set.index(last) + 1
    else:
        start = 0

    for item in set[start:]:
        current[index] = item

        __get_combinations_recursive(set, dim, current.copy(), index + 1, combinations, blacklists)


'''     NOT USED
def prune(C, L):
    good = []
    subset_dim = len(C[0]) - 1
    count = 0
    for c in C:
        count += 1
        print('\r' + str(count) + '/' + str(len(C)))
        comb = get_combinations(c, subset_dim, None)

        if not is_present(comb, L):
            continue

        good.append(c)

    return good
'''


def get_occurrencies(set, dataset):
    occ = []

    for item in set:
        occ.append((item, 0))  # building the tuple list

    for row in dataset:  # for each row of dataset
        for item in set:  # for each item in the set
            if is_present(item, row):  # I check if its present, in that case the value in the tuple in incremented
                index = get_index(item, occ)
                occ[index] = (occ[index][0], occ[index][1] + 1)

    return occ


def is_present(A, B):
    # different methods of checking inclusivity
    # bad practise but it works :/

    if type(A) == str:
        return A in B

    if len(A[0]) == 1:
        return all(elem[0] in B for elem in A)

    return all(elem in B for elem in A)


def get_index(key_set, list):  # retrieve index from key_set
    for i, row in enumerate(list):
        if all(e in row[0] for e in key_set):
            return i
    return -1


def get_col(list, col):  # get just a column of tuple list
    return [a[col] for a in list]


if __name__ == '__main__':
    main()
