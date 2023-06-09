import numpy as np

def razb_config():
    way = '2_garmonic\\config.txt'
    res = []
    tmp = ''
    with open(way,'r') as f:
        for c in f.readline():
            if c==',':
                res.append(int(tmp))
                tmp = ''
                continue
            tmp+=c
    res.append(int(tmp))
    return res

def cm_to_inch(value):
    return value/2.54

def razb_text(way):
    res = []
    elem = []
    value = ''
    f_htrix = False
    f_kvadr = False

    with open(way+'text.txt','r') as f:
        for line in f.readlines():
            for c in line:
                if c == ' ':
                    continue
                if c == '[':
                    f_kvadr = True
                    continue
                if c == ']':
                    f_kvadr = False
                    if value != '':
                        elem.append(float(value))
                    if len(elem) != 0:
                        res.append(elem)
                        f_htrix = False
                    value = ''
                    elem = []
                    continue
                if f_kvadr:
                    if c==',':
                        if value != '':
                            if f_htrix:
                                elem.append(value)
                            else:
                                elem.append(float(value))
                            value = ''
                        continue
                    if c=='\'':
                        f_htrix = True
                        continue
                    value+=c

    return res

def get_ind_text(arr,point):
    alpha,beta = point
    eps = 0.05
    ind = None
    for elem in arr:
        if np.abs(elem[0]-alpha)<eps and np.abs(elem[1]-beta)<eps:
            ind = arr.index(elem)
            break
    return ind

def add_to_good_arr(good_arr,arr):
    tmp= good_arr.copy()
    for line in arr:
        for elem in line:
            if elem !=[]:
                tmp.append(elem)
    return tmp

def add_unic_good_arr(good_arr,arr):
    tmp= good_arr.copy()
    f = True
    for a in arr:
        for g_a in tmp:
            if a[:2] == g_a[:2]:
                f = False
                break
        if f:
            tmp.append(a)
        f = True
    return tmp

def get_good_arr(arr):
    good_arr = []
    for line in arr:
        for elem in line:
            if elem !=[]:
                good_arr.append(elem)
    return good_arr

def mod(val,module):
    tmp = val
    while tmp>module:
        tmp-=module
    return tmp

if __name__ == "__main__":
    # arr1 = [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
    # arr2 = [[1,2,10],[2,3,4],[4,3,5],[4,5,6]]

    # f = True
    # for a2 in arr2:
    #     for a1 in arr1:
    #         if a1[:2] == a2[:2]:
    #             f = False
    #             break
    #     if f:
    #         arr1.append(a2)
    #     f = True
    # print(arr1)

    tmp = razb_config()
    param = [1.609267, 2, 0.0, 2.0944, 1, 1]

    
    way= f"2_garmonic\\res\\n_{tmp[0]}\\tmp\\{param}\\"
    
    arr = razb_text(way)
    ind = get_ind_text(arr,[0.2,2.2])
    if ind==None:
        print("not in array")
    else:
        print(arr[ind])

    # val = 1232123.124214
    # print(mod(val,np.pi))

    













# point_analyse 
# par = self.anti_par_zam(params)
        # par = np.reshape(par, (len(par[0])))

        # start_point=np.zeros(4)
        # start_point[0],start_point[1] = par[0:2] 
        # start_point[0] = start_point[0]+eps
        # start_point[1] = start_point[1]+eps
        # start_point[2] = eps
        # start_point[3] = eps

        # return self.__ord_par_tong__(start_point,show,t)