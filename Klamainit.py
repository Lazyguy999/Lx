from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# 辅助函数：将复数格式化为字符串
def format_complex_number(c):
    return f"{c.real:.4f} + {c.imag:.4f}j"

# 计算导纳矩阵
def calculate_admittance_matrix(line_parameters, num_nodes, harmonic_order, base_frequency):
    """
    根据线路参数和谐波阶数计算电力系统的导纳矩阵。
    """
    omega_base = 2 * np.pi * base_frequency  # 基频的角频率
    omega_harmonic = harmonic_order * omega_base  # 谐波频率的角频率

    # 初始化导纳矩阵为复数零矩阵
    admittance_matrix = np.zeros((num_nodes, num_nodes), dtype=complex)

    for start_node, end_node, resistance, reactance in line_parameters:
        # 计算谐波频率下的阻抗
        impedance = resistance + 1j * (reactance * harmonic_order)
        admittance = 1 / impedance

        # 更新导纳矩阵
        admittance_matrix[start_node, start_node] += admittance
        admittance_matrix[end_node, end_node] += admittance
        admittance_matrix[start_node, end_node] -= admittance
        admittance_matrix[end_node, start_node] -= admittance

    # 在参考节点（节点0）添加并联导纳
    shunt_admittance = 1 / (0.001 * 1j)
    admittance_matrix[0, 0] += shunt_admittance

    return admittance_matrix

# 计算阻抗矩阵
def calculate_impedance_matrix(admittance_matrix):
    """
    通过求逆导纳矩阵来计算阻抗矩阵。
    """
    impedance_matrix = np.linalg.inv(admittance_matrix)
    return impedance_matrix

def build_incidence_matrix(line_parameters, num_nodes):
    """
    构建线路-节点关联矩阵 A。

    参数:
        line_parameters (list of tuples): 每条线路的参数，格式为
            (起始节点, 终止节点, 电阻, 电抗)。
        num_nodes (int): 系统中的节点数量。

    返回:
        np.ndarray: 线路-节点关联矩阵 A，形状为 (线路数量, 节点数量)。
    """
    num_lines = len(line_parameters)
    incidence_matrix = np.zeros((num_lines, num_nodes), dtype=complex)

    for idx, (start_node, end_node, _, _) in enumerate(line_parameters):
        incidence_matrix[idx, start_node] = 1  # 起始节点标记为 +1
        incidence_matrix[idx, end_node] = -1  # 终止节点标记为 -1

    return incidence_matrix

def extract_rows(matrix, row_indices):
    """
    从输入矩阵中根据指定的行索引数组提取行，形成新的矩阵。

    参数:
        matrix (np.ndarray): 输入矩阵。
        row_indices (list or np.ndarray): 指定要提取的行索引。

    返回:
        np.ndarray: 包含指定行的新矩阵。
    """
    # 转换为 NumPy 数组（如果输入为列表）
    row_indices = np.array(row_indices)
    
    # 检查行索引是否超出矩阵行范围
    if np.any(row_indices < 0) or np.any(row_indices >= matrix.shape[0]):
        raise ValueError("行索引超出矩阵范围！")
    
    # 提取指定的行
    new_matrix = matrix[row_indices, :]
    
    return new_matrix

def combine_matrices(A, B, C):
    """
    将矩阵 A 和 B 垂直拼接，并在 B 下面添加 C。

    参数:
        A (np.ndarray): 矩阵 A。
        B (np.ndarray): 矩阵 B。

    返回:
        np.ndarray: 拼接后的新矩阵。
    """
    # 检查 A 和 B 是否有相同的列数
    if A.shape[1] != B.shape[1]:
        raise ValueError("矩阵 A 和 B 的列数必须相同！")
    if C.shape[1] != B.shape[1]:
        raise ValueError("矩阵 C 和 B 的列数必须相同！")
    
    # 按照垂直方向拼接矩阵 A、B 和单位矩阵 C
    combined_matrix = np.vstack((A, B, C))
    
    return combined_matrix
def kalam_init(Hc,voltage_pu, current_pu, num_voltage_meas, num_branch_current_meas,num_states):
    """
    初始化卡尔曼滤波器。
    """
    F = np.eye(num_states, dtype=complex)  # 状态转移矩阵 F，复数类型
    Hc = Hc
    Q = np.eye(num_states, dtype=complex) * 0.001 * current_pu  # 过程噪声协方差矩阵 Q，复数类型

    R = np.eye(Hc.shape[0], dtype=complex)  # 观测噪声协方差矩阵 R，复数类型
    R[:num_voltage_meas, :] *= 0.01 * voltage_pu
    R[num_voltage_meas:num_voltage_meas + num_branch_current_meas, :] *= 0.00002 * current_pu
    R[num_voltage_meas + num_branch_current_meas:, :] *= 0.0001 * current_pu
     
    P = np.eye(num_states, dtype=complex) * 1000 * current_pu  # 初始估计误差协方差矩阵 P，复数类型
    x = np.zeros((num_states, 1), dtype=complex)  # 初始状态估计 x0，复数类型

    return F,Hc,Q,R,P,x





# 定义 POST 接口
@app.route('/calculate-matrices', methods=['POST'])
def calculate_matrices():
    # 从请求中获取 JSON 数据
    data = request.get_json()

    # 提取参数
    line_parameters = data.get("line_parameters", [])
    num_nodes = data.get("num_nodes", 0)
    harmonic_order = data.get("harmonic_order", 1)
    base_frequency = data.get("base_frequency", 50.0)
    nodevoltageposition = data.get("nodevoltageposition", [])
    branchcurrentposition = data.get("branchcurrentposition", [])
    injectcurrentposition = data.get("injectcurrentposition", [])
    voltage_pu = data.get("voltage_pu", 1)
    current_pu = data.get("current_pu", 1)


    
    try:
        # 计算导纳矩阵
        admittance_matrix = calculate_admittance_matrix(
            line_parameters,
            num_nodes,
            harmonic_order,
            base_frequency
        )

        # 计算阻抗矩阵
        impedance_matrix = calculate_impedance_matrix(admittance_matrix)
        incidence_matrix= build_incidence_matrix(line_parameters, num_nodes)


        H1=extract_rows(impedance_matrix, nodevoltageposition)
        H2=extract_rows(incidence_matrix, branchcurrentposition)
        H3=extract_rows(np.eye(impedance_matrix.shape[0]), injectcurrentposition)
        Hc=combine_matrices(H1, H2, H3)
        # 格式化结果为可序列化的形式（列表嵌套）
        formatted_H1 = [[format_complex_number(c) for c in row] for row in H1]
        formatted_H2 = [[format_complex_number(c) for c in row] for row in H2]
        formatted_H3 = [[format_complex_number(c) for c in row] for row in H3]
        formatted_Hc = [[format_complex_number(c) for c in row] for row in Hc]
        num_voltage_meas=len(nodevoltageposition)
        num_branch_current_meas=len(branchcurrentposition)
       # num_states=impedance_matrix.shape[0]
        num_states=3
        F,Hc,Q,R,P,x=kalam_init(Hc,voltage_pu, current_pu, num_voltage_meas, num_branch_current_meas,num_states)
        formatted_F = [[format_complex_number(c) for c in row] for row in F]
        formatted_Q = [[format_complex_number(c) for c in row] for row in Q]
        formatted_R = [[format_complex_number(c) for c in row] for row in R]
        formatted_P = [[format_complex_number(c) for c in row] for row in P]
        formatted_x = [[format_complex_number(c) for c in row] for row in x]
        
        # 返回结果
        return jsonify({
            "Hc": formatted_Hc,
            "F": formatted_F,
            "Q": formatted_Q,
            "R": formatted_R,
            "P": formatted_P,
            "x": formatted_x

        })

    except np.linalg.LinAlgError as e:
        # 处理可能的矩阵求逆问题
        return jsonify({"error": "Failed to calculate impedance matrix. Matrix might be singular.", "details": str(e)}), 400

    except Exception as e:
        # 捕获其他错误
        return jsonify({"error": str(e)}), 400




if __name__ == '__main__':
    # 启动服务
    app.run(debug=True,port=9000)
