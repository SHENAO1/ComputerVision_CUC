import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# ==========================================
# 第一部分：生成数据点集 
# ==========================================
def generate_data():
    print("正在生成数据...")
    
    # 2) 自定义平面参数 (真值)
    # 例如：z = 0.5x - 0.3y + 10
    alpha_true = 0.5
    beta_true = -0.3
    gamma_true = 10.0
    print(f"真值参数: alpha={alpha_true}, beta={beta_true}, gamma={gamma_true}")

    # 3) 生成 900 个内点
    # x, y 在 [0, 30) 且为整数
    x_range = np.arange(0, 30)
    y_range = np.arange(0, 30)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # 拉平以便处理
    x_inliers = X_grid.flatten()
    y_inliers = Y_grid.flatten()
    
    # 计算 z 值并添加高斯噪声
    # 假设噪声标准差为 0.5
    noise = np.random.normal(0, 0.5, size=x_inliers.shape)
    z_inliers = alpha_true * x_inliers + beta_true * y_inliers + gamma_true + noise
    
    # 组合内点 (900, 3)
    inliers = np.column_stack((x_inliers, y_inliers, z_inliers))

    # 4) 生成 100 个外点
    # 确定空间范围以便生成合理的外点
    x_min, x_max = 0, 30
    y_min, y_max = 0, 30
    z_min, z_max = int(np.min(z_inliers)) - 10, int(np.max(z_inliers)) + 10
    
    # 在稍大的空间范围内生成随机整数坐标
    outliers = np.random.randint(
        low=[x_min, y_min, z_min], 
        high=[x_max + 10, y_max + 10, z_max], 
        size=(100, 3)
    )

    # 合并数据 (1000个点)
    all_points = np.vstack((inliers, outliers))
    
    # 5) 保存为 points.npy
    np.save("points.npy", all_points)
    print("数据已保存至 points.npy\n")
    return alpha_true, beta_true, gamma_true

# ==========================================
# 第二部分：基于 RANSAC 的内点集获取
# ==========================================
def ransac_plane_fitting():
    # 1) 加载数据
    points = np.load("points.npy")
    N_total = points.shape[0]
    
    # RANSAC 参数
    threshold = 1.0  # 设定的距离阈值
    p = 100 / 1000   # 外点比例 p = 0.1
    prob_success = 0.99 # 要求至少99%概率成功
    
    # 3) 计算理论需要的采样次数 N
    # N >= log(1 - prob_success) / log(1 - (1-p)^3)
    numerator = np.log(1 - prob_success)
    denominator = np.log(1 - (1 - p)**3)
    N_iter = int(math.ceil(numerator / denominator))
    
    print(f"外点比例 p={p}")
    print(f"理论最少采样次数 N={N_iter}")

    best_model = None
    best_inliers_indices = []
    max_inliers_count = 0

    # 4) 重复步骤 N 次
    for i in range(N_iter):
        # -- 采样: 随机选取 3 个点 --
        sample_indices = np.random.choice(N_total, 3, replace=False)
        sample_points = points[sample_indices]
        
        p1, p2, p3 = sample_points[0], sample_points[1], sample_points[2]
        
        # -- 检查共线 --
        # 通过构建矩阵求行列式或者判断向量叉乘是否为0
        # 这里构建方程组求解参数的矩阵 A
        A_sample = np.column_stack((sample_points[:, 0], sample_points[:, 1], np.ones(3)))
        b_sample = sample_points[:, 2]
        
        # 如果行列式接近0，说明共线或无解 (奇异矩阵)
        if np.abs(np.linalg.det(A_sample)) < 1e-6:
            continue

        # -- 模型计算: 求 alpha, beta, gamma --
        # 解方程 A * [a, b, g].T = z
        try:
            params = np.linalg.solve(A_sample, b_sample)
            a_hat, b_hat, g_hat = params
        except np.linalg.LinAlgError:
            continue # 跳过无法求解的情况

        # -- 内点统计: 计算所有点到平面的距离 --
        # distance = |ax + by + g - z| / sqrt(a^2 + b^2 + 1)
        x_all, y_all, z_all = points[:, 0], points[:, 1], points[:, 2]
        
        numer = np.abs(a_hat * x_all + b_hat * y_all + g_hat - z_all)
        denom = np.sqrt(a_hat**2 + b_hat**2 + 1)
        distances = numer / denom
        
        # 统计小于阈值的点
        current_inliers_mask = distances < threshold
        current_inliers_count = np.sum(current_inliers_mask)
        
        # 更新最佳模型
        if current_inliers_count > max_inliers_count:
            max_inliers_count = current_inliers_count
            best_model = (a_hat, b_hat, g_hat)
            best_inliers_indices = current_inliers_mask

    print(f"RANSAC 完成。找到最大内点数: {max_inliers_count}")
    return points, best_inliers_indices, best_model

# ==========================================
# 第三部分：平面的最小二乘拟合
# ==========================================
def least_squares_refinement(points, inliers_mask, true_params):
    print("\n开始最小二乘优化...")
    
    # 1) 构建非齐次线性方程组 Ax = b
    # 注意：只使用 RANSAC 筛选出的最佳内点集
    best_points = points[inliers_mask]
    
    x_data = best_points[:, 0]
    y_data = best_points[:, 1]
    z_data = best_points[:, 2]
    
    # A 矩阵: [x, y, 1]
    A = np.column_stack((x_data, y_data, np.ones(len(x_data))))
    b = z_data
    
    # 2) 使用最小二乘法求解优化后的参数
    # result[0] 是解向量，result[1] 是残差等
    solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    alpha_fit, beta_fit, gamma_fit = solution
    
    print("优化后的参数:")
    print(f"alpha: {alpha_fit:.4f}")
    print(f"beta : {beta_fit:.4f}")
    print(f"gamma: {gamma_fit:.4f}")
    
    # 3) 计算并打印误差
    alpha_true, beta_true, gamma_true = true_params
    print("\n与真值的绝对误差:")
    print(f"Err_alpha: {abs(alpha_fit - alpha_true):.6f}")
    print(f"Err_beta : {abs(beta_fit - beta_true):.6f}")
    print(f"Err_gamma: {abs(gamma_fit - gamma_true):.6f}")

    return alpha_fit, beta_fit, gamma_fit

# ==========================================
# 可视化
# ==========================================
def plot_results(points, inliers_mask, fit_params):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制外点 (红色)
    outliers = points[~inliers_mask]
    ax.scatter(outliers[:,0], outliers[:,1], outliers[:,2], c='r', marker='x', label='Outliers', alpha=0.5)
    
    # 绘制内点 (绿色)
    inliers = points[inliers_mask]
    ax.scatter(inliers[:,0], inliers[:,1], inliers[:,2], c='g', marker='.', label='Inliers', alpha=0.3)
    
    # 绘制拟合平面
    alpha, beta, gamma = fit_params
    x_range = np.arange(0, 30, 2)
    y_range = np.arange(0, 30, 2)
    X, Y = np.meshgrid(x_range, y_range)
    Z = alpha * X + beta * Y + gamma
    
    ax.plot_surface(X, Y, Z, alpha=0.3, color='b')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.title("RANSAC & Least Squares Plane Fitting")
    plt.show()

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 生成数据
    true_params = generate_data()
    
    # 2. RANSAC
    points, inliers_mask, ransac_model = ransac_plane_fitting()
    
    if np.sum(inliers_mask) > 0:
        # 3. 最小二乘优化
        fit_params = least_squares_refinement(points, inliers_mask, true_params)
        
        # 4. 绘图
        plot_results(points, inliers_mask, fit_params)
    else:
        print("RANSAC 未找到足够的内点。")