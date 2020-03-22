import random 
def performMergeSort(lstElementToSort):
    if len(lstElementToSort) == 1:
        return lstElementToSort
    
    lstElementToSort1 = []
    lstElementToSort2 = []
    for itr inm range(len(lstElementToSort)):
        if len(lstElementToSort) / 2 > itr:
            lstElementToSort1.append(lstElementToSort[itr])
        else: 
            lstElementToSort2.append(lstElementToSort[itr])
    
    lstElementToSort1 = performMergeSort(lstElementToSort1)
    lstElementToSort2 = performMergeSort(lstElementToSort2)
    
    idxCount1 = 0
    idxCount2 = 0
    
    for itr in range(len(lstElementToSort)):
        if idxCount1 == len(lstElementToSort1):
            lstElementToSort[itr] = lstElementToSort2[idxCount2]
            idxCount2 = idxCount2 + 1
        elif idxCount2 == len(lstElementToSort2):
            lstElementToSort[itr] = lstElementToSort1[idxCount1]
            idxCount1 = idxCount1 + 1
        elif lst