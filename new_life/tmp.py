import numpy as np

arr = [[0.0, 0.0, 1, 2, 0.0],
[0.0, 3.142, 1, 2, 0.0],
[-0.0, 0.0, 1, 2, 0.0],
[3.142, 0.0, 1, 2, 0.0],
[3.142, 3.142, 1, 2, 0.0],
[0.0, 3.142, 1, 2, 0.0],
[4, 0.0, 1, 2, 0.0],
[0.0, 5, 1, 2, 0.0],
[0.0, -0.0, 1, 2, 0.0],
[0.0, 3.142, 1, 2, 0.0],
[0.0, 0.0, 1, 2, 0.0],
[0.0, 3.142, 1, 3, 0.0],
[0.0, 0.0, 1, 3, 0.0]]

def trash_off(arr):
    tmp = []
    tmp_res = []
    res = []
    par = arr[0][2:5]
    for i in arr:
        if par == i[2:5]:
            tmp = [round(np.sin(i[0])),round(np.sin(i[1]))]
            if tmp not in tmp_res:
                tmp_res.append(tmp)
                res.append(i)
        else :
            tmp_res = []
            par = i[2:5]
            tmp = [round(np.sin(i[0])+np.cos(i[0])),round(np.sin(i[1])+np.cos(i[1]))]
            tmp_res.append(tmp)
            res.append(i)
    return res
        
print(trash_off(arr))