import random

def performSelectionSort(lst):
    for itr1 in range(0, len(lst)):
        for itr2 in range(itr1+1, len(lst)):
            if lst[itr1] < lst[itr2]:
                lst[itr1], lst[itr2] = lst[itr2], lst[itr1]
    return lst

def bubblesort(lst):
    for idx1 in range(0, len(lst)):
        for idx2 in range(0, idx1):
            if lst[idx1] < lst[idx2]:
                lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
    return lst
N = 10
lstNumbers = list(range(N))
random.shuffle(lstNumbers)

print(lstNumbers)
print(performSelectionSort(lstNumbers))

# lstNumbers2 = [2, 5, 0, 3, 3, 3, 1, 5, 4, 2]

print(lstNumbers)
print(bubblesort(lstNumbers))

#%%

import random

def performSelectionSort(lst):
    for itr1 in range(0, len(lst)):
        for itr2 in range(itr1+1, len(lst)):
            if lst[itr1] < lst[itr2]:
                lst[itr1], lst[itr2] = lst[itr2], lst[itr1]
    return lst



N = 10
lstNumbers = list(range(N))
random.shuffle(lstNumbers)

print('Bubble sort 하기전:',lstNumbers)
print('Bubble sort 한 후',performSelectionSort(lstNumbers))

# %%

def bubblesort_1(lst):
    for idx1 in range(0, len(lst)):
        for idx2 in range(0, idx1):
            if lst[idx1] < lst[idx2]:
                lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
                print(lst)
    return lst

    
def bubbleSort_2(lst):
    for itr1 in range(0, len(lst)):
        for itr2 in range(itr1+1, len(lst)):
            if lst[itr1] < lst[itr2]:
                lst[itr1], lst[itr2] = lst[itr2], lst[itr1]
                print(lst)
    return lst

lstNumbers2 = [3,1,6,5,2,4]

print(lstNumbers2)
print(performSelectionSort(lstNumbers2))


# %%


# %%
