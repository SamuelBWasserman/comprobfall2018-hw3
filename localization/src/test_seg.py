import numpy as np

def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_interset(a1,a2,b1,b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, db)
    return (num / denom.astype(float))*db + b1


if __name__ == "__main__":
    p1 = np.array([0.0, 0.0])
    p2 = np.array([0.0, 1.0])
    p3 = np.array([-1.0, 0.0])
    p4 = np.array([1.0, 1.0])

    # another way
    t, s = np.linalg.solve(np.array([p2-p1, p3-p4]).T, p3-p1)
    print((1-t)*p1 +t*p2)

    print seg_interset(p1, p2, p3, p4)