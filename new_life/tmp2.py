

def razb_str(str):
        all = []
        tmp = ''
        tmp_arr = []
        flag = False
        flag2 = False

        for c in str:
            if len(tmp_arr) == 2:
                break
            if c == '[' or (flag and c == ' '):
                if len(tmp)>1:
                    flag2 = True
                else:
                    flag2 = False
                flag = False
                continue
            if c==',' or c==']' or (c==' ' and flag2):
                tmp_arr.append(float(tmp))
                tmp = ''
                flag = True
                continue
            tmp += c
        return tmp_arr


# string = '[ 2.44630735 10.        ]'
# a = razb_str(string)
# print(a)
way = f"new_life\\res\\n_5\\border_tongue_tmp.txt"
ar = [[0,1],[1,0]]
with open(way,'w',encoding="utf-8") as file:
    for el in ar:
        string = str(el[0]) + ',' + str(el[1]) + '\n'
        file.write(string)