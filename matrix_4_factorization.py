# 课程：矩阵分析与应用
# 实现矩阵的4个分解：LU分解，施密特正交化QR分解，Householder实现的QR分解，Givens实现的QR分解
# 此代码中假设待分解的矩阵均为n*n的方阵

import numpy as np


# LU分解
def LU_fac(matrix, matrix_row_num, matrix_col_num):
    # 根据矩阵的shape初始化L矩阵
    L_matrix = np.zeros((matrix_row_num, matrix_col_num))
    for i in range(matrix_row_num):
        L_matrix[i][i] = 1

    # 高斯消去过程
    is_enable_LU_flag = True
    for j in range(0, matrix_col_num):  # 列遍历
        current_pivot = matrix[j][j]
        if (current_pivot == 0):
            is_enable_LU_flag = False
            break
        for i in range(j + 1, matrix_row_num):  # 行遍历
            current_eliminate_coefficient = -(matrix[i][j] / current_pivot)
            L_matrix[i][j] = -current_eliminate_coefficient
            for k in range(0, matrix_col_num):
                matrix[i][k] = matrix[i][k] + matrix[j][k] * current_eliminate_coefficient
    # 获得U矩阵
    U_matrix = matrix
    if is_enable_LU_flag == True:
        return L_matrix, U_matrix
    else:
        return None, None


# 经典的施密特正交化实现QR分解
def Schmidt_QR_fac(matrix, matrix_row_num, matrix_col_num):
    # 初始化Q矩阵和R矩阵
    Q_matrix = np.zeros((matrix_row_num, matrix_col_num))
    R_matrix = np.zeros((matrix_row_num, matrix_col_num))
    for j in range(0, matrix_col_num):  #列遍历
        # 进行行遍历获取当前列的向量
        current_col_vec = np.array([])
        for i in range(0, matrix_row_num):
            current_col_vec = np.append(current_col_vec, matrix[i][j])
        if j == 0:
            # 处理第一列
            # 计算当前列的模并将值赋给r11
            norm = np.linalg.norm(current_col_vec)
            R_matrix[j][j] = norm
            current_col_vec = current_col_vec / norm
            # 将单位化的向量赋值给Q矩阵的第一列
            for i in range(0, matrix_col_num):
                Q_matrix[i][j] = current_col_vec[i]
        else:
            # 处理其他列
            # 第一层循环，循环次数为0到j-1，计算当前列的rij值
            # 计算当前列的rij值：qi的转置乘以aj，即对应元素相乘（qi在之前循环中均已得到）
            # 计算完rij后，将rij存入R矩阵中
            # 当前列减去0到j-1列的（rij乘qi）
            # 计算rjj即取当前列q的模
            # 将当前列q单位化
            # 计算当前列的模并将值赋给rjj
            for k in range(0, j):
                r = np.dot(Q_matrix[:, k], matrix[:, j])
                R_matrix[k][j] = r

            q = np.zeros((matrix_col_num))
            for k in range(0, j):
                q = matrix[:, j] - R_matrix[k][j] * Q_matrix[:, k]
            # 计算rjj，并将q单位化
            rjj = np.linalg.norm(q)
            q = q / rjj
            R_matrix[j][j] = rjj
            # 将q向量赋值给Q矩阵的第j列
            for i in range(0, matrix_col_num):
                    Q_matrix[i][j] = q[i]
    return Q_matrix, R_matrix


# Householder约简方法实现QR分解
def Householder_fac(matrix, matrix_row_num, matrix_col_num):
    # 外层循环产生当前待处理的子矩阵current_matrix，每一个循环处理一列
    # 循环体内，计算当前矩阵的uj，然后计算响应的映射算子Rj，
    # 然后将Rj乘到原矩阵得到更新后的矩阵
    # 直到处理到倒数第二列为止
    Q_matrix = np.zeros((matrix_row_num, matrix_col_num))
    R_matrix = np.zeros((matrix_row_num, matrix_col_num))
    P_matrix = gene_unit_matrix(matrix_col_num)
    for j in range(0, matrix_col_num-1):  # 该循环控制处理列的次数
        # 寻找出当前待处理的子矩阵
        current_matrix = matrix[j:matrix_row_num, j:matrix_col_num]
        # 计算u
        u = current_matrix[:,0] - np.linalg.norm(current_matrix[:,0]) * gene_unit_vec(current_matrix[:,0].size, 0)
        # 计算映射算子R
        unit_matrix = gene_unit_matrix(current_matrix[:,0].size)
        R = gene_reflector(unit_matrix, u)
        RR = gene_unit_matrix(matrix_row_num)
        RR[j:matrix_row_num, j:matrix_col_num] = R[:]
        P_matrix = np.dot(RR, P_matrix)
        # 将R乘到原矩阵上得到新的矩阵
        matrix = np.dot(R, current_matrix)
        R_matrix[j:matrix_row_num, j:matrix_col_num] = matrix
    Q_matrix = np.transpose(P_matrix)  # 矩阵的转置
    return Q_matrix, R_matrix


# Givens约简方法实现QR分解
def Givens_fac(matrix, matrix_row_num, matrix_col_num):
    # 最外层循环，列循环，从第0列到倒数第一列
    # 嵌套一个行循环，从第j+1个元素开始到该列最后一个元素
    # 在内层循环即行循环中：
    # 首先计算c和s，然后构造旋转算子P，将P乘到当前的矩阵上，之后更新一下当前矩阵
    P_matrix = gene_unit_matrix(matrix_col_num)
    for j in range(0, matrix_col_num-1):
        for i in range(j+1, matrix_row_num):
            # 计算c与s
            c = matrix[j][j]/((matrix[j][j]**2 + matrix[i][j]**2)**0.5)
            s = matrix[i][j]/((matrix[j][j]**2 + matrix[i][j]**2)**0.5)
            # 构建旋转算子P
            P = gene_unit_matrix(matrix_col_num)
            P[j][j] = c
            P[j][i] = s
            P[i][j] = -s
            P[i][i] = c
            # 将旋转算子乘到原矩阵并更新原矩阵
            matrix = np.dot(P, matrix)
            # 记录P的累积乘积
            P_matrix = np.dot(P, P_matrix)
    Q_matrix = np.transpose(P_matrix)
    R_matrix = matrix[:]
    return Q_matrix, R_matrix


# 生成规定长度len，下标为i（从0开始）的单位向量
def gene_unit_vec(len, i):
    unit_vec = np.zeros((len))
    unit_vec[i] = 1
    return unit_vec


# 生成规定长度len的单位矩阵
def gene_unit_matrix(len):
    unit_matrix = np.zeros((len, len))
    for i in range(unit_matrix.shape[0]):
        for j in range(unit_matrix.shape[1]):
            if i==j:
                unit_matrix[i][j] = 1
    return unit_matrix


# 生成映射算子Ri（i为列的下标，从0开始）
def gene_reflector(unit_matrix, uj):
    # 计算ui乘ui的转置
    ui_multi_result_matrix = np.zeros((uj.size, uj.size))
    for i in range(0, uj.size): # 行遍历
        for j in range(0, uj.size): # 列遍历
            ui_multi_result_matrix[i][j] = uj[i] * uj[j]
    Rj = unit_matrix - 2 * ((ui_multi_result_matrix) / np.dot(uj, uj))
    return Rj


if __name__ == '__main__':
    print('----请输入数字选择矩阵的分解方式----')
    fac_type = int(input('1：LU分解\n2：施密特正交化QR分解\n3：Householder分解\n4：Givens分解:\n'))

    while (True):
        if (fac_type!=1) & (fac_type!=2) & (fac_type!=3) & (fac_type!=4):
            print('序号输入错误')
            fac_type = int(input('1：LU分解\n2：施密特正交化QR分解\n3：Householder分解\n4：Givens分解:\n'))
        else:
            break

    matrix_row_num = int(input('请输入矩阵的维数：'))
    matrix_col_num = matrix_row_num
    input_content = input('请输入矩阵中的' + str(matrix_row_num * matrix_col_num) + '个元素，以英文逗号分隔：')
    matrix_num_list_str = input_content.split(',')
    matrix_num_list = []  # 存放矩阵所有元素的列表
    for num in matrix_num_list_str:
        matrix_num_list.append(float(num))
    matrix_num_list = np.array(matrix_num_list)
    matrix = matrix_num_list.reshape(matrix_row_num, matrix_col_num)  # 待LU分解的矩阵

    if fac_type == 1:  # LU分解
        L_matrix, U_matrix = LU_fac(matrix, matrix_row_num, matrix_col_num)
        if (not L_matrix is None) & (not U_matrix is None):
            print('矩阵的LU分解结果为：')
            print('L矩阵为：\n', L_matrix)
            print('U矩阵为：\n', U_matrix)
        else:
            print('该矩阵无法进行LU分解')
    elif fac_type == 2:  # 施密特正交化实现的QR分解
        Q_matrix, R_matrix = Schmidt_QR_fac(matrix, matrix_row_num, matrix_col_num)
        print('矩阵的施密特正交化QR分解结果为：')
        print('Q矩阵为：\n', Q_matrix)
        print('R矩阵为：\n', R_matrix)
    elif fac_type == 3: # Householder实现的QR分解
        Q_matrix, R_matrix = Householder_fac(matrix, matrix_row_num, matrix_col_num)
        print('矩阵的Householder的QR分解结果为：')
        print('Q矩阵为：\n', Q_matrix)
        print('R矩阵为：\n', R_matrix)
    else:  # Givens实现的QR分解
        Q_matrix, R_matrix = Givens_fac(matrix, matrix_row_num, matrix_col_num)
        print('矩阵的Givens的QR分解结果为：')
        print('Q矩阵为：\n', Q_matrix)
        print('R矩阵为：\n', R_matrix)
