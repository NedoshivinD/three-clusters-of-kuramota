
N = 3
m = 1
eps = 0.7
omega = 1


def razb_str(str):
    all = []
    tmp = ''

    for c in str:
        if c==' ' or c=='[':
            continue
        if c==',' or c==']':
            all.append(float(tmp))
            tmp = ''
            continue
        tmp+=c
    return all