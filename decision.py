import numpy as np
import cvxpy as cp
import warnings

warnings.filterwarnings('error')

gd_r = 0.99
bc_r = 0.98


def decision(type, r2, r3, s2, s3, x1, x2, x3):  # r2,r3分别是黄金比特币增长率, s2,s3分别是黄金比特币风险, x1,x2,x3是美元黄金比特币初始值
    # 根据类型设置规划问题值
    if type == 0:  # 双买
        c1 = np.array([-r2 * gd_r, -r3 * bc_r])
        c2 = np.array([s2 * gd_r, s3 * bc_r])
        a = np.array([[1, 1]])
        b = np.array([x1])
        x = cp.Variable(2, nonneg=True)
    elif type == 1:  # 双卖
        c1 = np.array([-r2 / gd_r, -r3 / bc_r])
        c2 = np.array([s2 / gd_r, s3 / bc_r])
        a = np.array([[-1, 0], [0, -1]])
        b = np.array([gd_r * x2, bc_r * x3])
        x = cp.Variable(2, nonpos=True)
    elif type == 2:  # gold买 bitcoin卖
        c1 = np.array([-r2 * gd_r, -r3 / bc_r])
        c2 = np.array([s2 * gd_r, s3 / bc_r])
        a = np.array([[-1, 0], [0, 1], [0, -1], [1, 1]])
        b = np.array([0, 0, bc_r * x3, x1])
        x = cp.Variable(2)
    else:  # gold卖 bitcoin买
        c1 = np.array([-r2 / gd_r, -r3 * bc_r])
        c2 = np.array([s2 / gd_r, s3 * bc_r])
        a = np.array([[1, 0], [-1, 0], [0, -1], [1, 1]])
        b = np.array([0, gd_r * x2, 0, x1])
        x = cp.Variable(2)

    # 解多目标线性规划
    con = [a @ x <= b]

    obj1 = cp.Minimize(c1 @ x)
    prob1 = cp.Problem(obj1, con)
    prob1.solve(solver='SCS')
    v1 = prob1.value  # 第一个目标函数的最优值
    obj2 = cp.Minimize(c2 @ x)
    prob2 = cp.Problem(obj2, con)
    prob2.solve(solver='SCS')
    v2 = prob2.value  # 第二个目标函数的最优值
    # print('\n======结果为======\n')
    # print('两个目标函数的最优值分别为：', v1, v2)
    obj3 = cp.Minimize((c1 @ x - v1) ** 2 + (c2 @ x - v2) ** 2)
    prob3 = cp.Problem(obj3, con)
    prob3.solve(solver='SCS')  # GLPK_MI 解不了二次规划，只能用CVXOPT求解器

    # 结果
    res = x.value


    if res is None:
        return -np.inf, 0, 0, 0

    # 根据类型设置残余
    if type == 0:  # 双买
        y2 = res[0] * gd_r + x2  # 黄金当前量
        y3 = res[1] * bc_r + x3  # 比特币当前量
    elif type == 1:  # 双卖
        y2 = res[0] / gd_r + x2  # 黄金当前量
        y3 = res[1] / bc_r + x3  # 比特币当前量
    elif type == 2:  # gold买 bitcoin卖
        y2 = res[0] * gd_r + x2  # 黄金当前量
        y3 = res[1] / bc_r + x3  # 比特币当前量
    else:  # gold卖 bitcoin买
        y2 = res[0] / gd_r + x2  # 黄金当前量
        y3 = res[1] * bc_r + x3  # 比特币当前量

    y1 = x1 - res[0] - res[1]

    pred = y2 * r2 + y3 * r3

    return pred, y1, y2, y3


def choose(r2, r3, s2, s3, x1, x2, x3):
    m = -np.inf
    y1_res = x1
    y2_res = x2
    y3_res = x3  # 如果全部异常,不重新分配

    for type in range(4):  # 找最大值
        try:
            pred, y1, y2, y3 = decision(type, r2, r3, s2, s3, x1, x2, x3)
        except:
            continue

        if pred > m:
            m = pred
            y1_res = y1
            y2_res = y2
            y3_res = y3
    return m, y1_res, y2_res, y3_res


def bitcoin_only(type, r3, s3, x1, x2, x3):
    if type == 0:
        c1 = np.array([-r3 * bc_r])
        c2 = np.array([s3 * bc_r])
        a = np.array([[1], [-1]])
        b = np.array([x1, 0])
        x = cp.Variable(1)
    else:
        c1 = np.array([-r3 / bc_r])
        c2 = np.array([s3 / bc_r])
        a = np.array([[1], [-1]])
        b = np.array([0, bc_r * x3])
        x = cp.Variable(1)

    # 解多目标线性规划
    con = [a @ x <= b]

    obj1 = cp.Minimize(c1 @ x)
    prob1 = cp.Problem(obj1, con)
    prob1.solve(solver='SCS')
    v1 = prob1.value  # 第一个目标函数的最优值
    obj2 = cp.Minimize(c2 @ x)
    prob2 = cp.Problem(obj2, con)
    prob2.solve(solver='SCS')
    v2 = prob2.value  # 第二个目标函数的最优值
    # print('\n======结果为======\n')
    # print('两个目标函数的最优值分别为：', v1, v2)
    obj3 = cp.Minimize((c1 @ x - v1) ** 2 + (c2 @ x - v2) ** 2)
    prob3 = cp.Problem(obj3, con)
    prob3.solve(solver='SCS')  # GLPK_MI 解不了二次规划，只能用CVXOPT求解器

    # 结果
    res = x.value

    if res is None:
        return -np.inf, 0, 0, 0

    # 根据类型设置残余
    if type == 0:  # 买
        y3 = res[0] * bc_r + x3  # 比特币当前量
    else:  # 卖
        y3 = res[0] / bc_r + x3  # 比特币当前量

    y2 = x2
    y1 = x1 - res[0]
    pred = y2 * 1 + y3 * r3

    return pred, y1, y2, y3


def choose2(r3, s3, x1, x2, x3):
    m = -np.inf
    y1_res = x1
    y2_res = x2
    y3_res = x3  # 如果全部异常,不重新分配
    for type in range(2):  # 找最大值
        try:
            pred, y1, y2, y3 = bitcoin_only(type, r3, s3, x1, x2, x3)
        except:
            continue

        if pred > m:
            m = pred
            y1_res = y1
            y2_res = y2
            y3_res = y3
    return m, y1_res, y2_res, y3_res


if __name__ == '__main__':
    print(choose(0.017297237590586764,-0.0007621129556117445,0.009154421103010827,0.00817252970406087,954.38,37.45,0.0))
