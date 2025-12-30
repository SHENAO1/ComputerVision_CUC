import numpy as np
import cv2
import matplotlib.pyplot as plt

# =============================================================
# 1: 加载数据
# =============================================================
def load_data(img1_path, img2_path, k_path):
    """读取图片和内参矩阵"""
    
    # 使用 cv2.imread 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # 检查图片是否读取成功
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"无法读取图片，请检查路径: {img1_path} 或 {img2_path}")

    # 使用 np.loadtxt 读取内参矩阵
    K = np.loadtxt(k_path)
    

    return img1, img2, K

# =========================================================
# 2: 特征提取与匹配
# =========================================================
def extract_and_match_features(img1, img2):
    """
    使用 SIFT 提取特征，使用 KNN 进行匹配，并进行 Lowe's Ratio Test。
    (参考从 11 立体视觉（下）单独拆分的图像矫正 PDF)
    """
    # 1. 初始化 SIFT 检测器
    # 注意：图像矫正 PDF Page 24 使用了默认参数 sift = cv2.SIFT_create() 
    # 但脚本模板中已经给定了特定的参数 (contrastThreshold等)，保留模板的参数
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=5, contrastThreshold=0.001, edgeThreshold=10, sigma=1.4)

    # 2. 检测关键点和计算描述子
    # 参考 图像矫正 PDF Page 23-25 的 “# Find the keypoints and descriptors with SIFT” 部分代码
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. 特征匹配 (使用 FLANN)
    # 参考 图像矫正 PDF Page 29 的 “# Match keypoints in both images” 部分代码
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) #  Page 29 中的 trees=5
    search_params = {} # 图像矫正 PDF Page 29 传入了空字典 {}

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches_all = flann.knnMatch(des1, des2, k=2)

    # 4. Lowe's Ratio Test (比率测试)
    # 参考 图像矫正 PDF Page 30 “# Keep good matches: calculate distinctive image features” 部分代码
    good = []
    pts1 = []
    pts2 = []
    
    for m, n in matches_all:
        if m.distance < 0.7 * n.distance: # 0.7 是 PPT 中的阈值 
            # --- 修改点 ---
            # PDF Page 32 写的是 good.append([m]) 用于 drawMatchesKnn 
            # 但任务要求使用 drawMatches，所以这里必须 append(m)，去掉中括号。
            good.append(m) 
            
            # 提取坐标用于后续计算 (参考 PDF Page 32) 
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            
    # 将点转换为 numpy 数组，参考 PDF Page 35，后续计算基础矩阵需要 float32
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    
    print(f"[Info] Found {len(good)} good matches.")
    return pts1, pts2, good, kp1, kp2

# =========================================================
# 3.1: 数据归一化
# =========================================================
def normalize_points(pts):
    """
    将特征点平移至原点，并缩放至平均距离为 sqrt(2)
    输入: pts (Nx2)
    输出: norm_pts (Nx2), T (3x3 变换矩阵)
    """
    pts = np.array(pts)
    N = pts.shape[0]
    
    # 1. 计算质心 (cx, cy)
    mean = np.mean(pts, axis=0)
    cx, cy = mean[0], mean[1]
    
    # 2. 平移
    shifted_pts = pts - mean
    
    # 3. 计算平均距离 (Mean Distance from origin)
    # dist = sqrt(x^2 + y^2)
    dist = np.sqrt(np.sum(shifted_pts**2, axis=1))
    mean_dist = np.mean(dist)
    
    # 4. 计算缩放因子 s (使得平均距离为 sqrt(2))
    # 避免除以0
    if mean_dist < 1e-8:
        s = 1.0
    else:
        s = np.sqrt(2) / mean_dist

    # 5. 缩放
    norm_pts = shifted_pts * s
    
    # 6. 构造变换矩阵 T
    # T = [s  0  -s*cx]
    #     [0  s  -s*cy]
    #     [0  0     1 ]
    T = np.array([
        [s, 0, -s*cx],
        [0, s, -s*cy],
        [0, 0, 1.0]
    ])
    
    return norm_pts, T

# =========================================================
# 3.2: 归一化8点算法
# =========================================================
def solve_F_8point(pts1, pts2):
    """
    使用 8 点算法计算基础矩阵 F
    输入: pts1, pts2 (Nx2)
    输出: F (3x3)
    参考拆分的8点算法 PDF 中的第五页和第六页
    """
    # --- Step 1: 归一化 (Normalization) ---
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)
    
    N = pts1_norm.shape[0]
    
    # --- Step 2: 构建系数矩阵 A (Nx9) ---
    # 方程: p_r^T F p_l = 0  =>  [u'u, u'v, u', v'u, v'v, v', u, v, 1] * f = 0
    # 参考 PDF Page 5: (xr*xl, xr*yl, xr, yr*xl, yr*yl, yr, xl, yl, 1) 
    # 这里 u, v 对应左图 (xl, yl); u', v' 对应右图 (xr, yr)
    u = pts1_norm[:, 0]
    v = pts1_norm[:, 1]
    u_p = pts2_norm[:, 0]
    v_p = pts2_norm[:, 1]
    
    A = np.column_stack((
        u_p * u, u_p * v, u_p,
        v_p * u, v_p * v, v_p,
        u,       v,       np.ones(N)
    ))
    
    # --- Step 3: SVD 求解 Af=0 ---
    # 对 A 进行 SVD，最小奇异值对应的右奇异向量即为解
    U, S, Vt = np.linalg.svd(A)
    F_vec = Vt[-1] # 取最后一行
    F_norm = F_vec.reshape(3, 3)
    
    # --- Step 4: 强制秩约束 (Enforce Rank-2) ---
    # 基础矩阵的秩必须为 2
    Uf, Sf, Vft = np.linalg.svd(F_norm)
    Sf[2] = 0 # 将最小的奇异值置为 0
    F_norm = Uf @ np.diag(Sf) @ Vft
    
    # --- Step 5: 去归一化 (Denormalization) ---
    # F = T2^T * F_norm * T1
    F = T2.T @ F_norm @ T1
    
    # 归一化 F 使得最后一个元素为 1 (可选，但通常有助于数值稳定性)
    if abs(F[2, 2]) > 1e-8:
        F = F / F[2, 2]
        
    return F
# =========================================================
# 3.3: 误差计算 (用于 RANSAC)
# =========================================================
def compute_epipolar_errors(F, pts1, pts2):
    """
    计算点到极线的距离 (Sampson Distance 的一阶近似)
    Distance = |x'Fx| / sqrt((Fx)_1^2 + (Fx)_2^2)
    即点 x' 到极线 l = Fx 的几何距离
    """
    N = pts1.shape[0]
    pts1_homo = np.hstack((pts1, np.ones((N, 1))))
    pts2_homo = np.hstack((pts2, np.ones((N, 1))))
    
    # 1. 计算极线 lines1 = F^T * x2 (在图1中的极线)
    #    计算极线 lines2 = F * x1   (在图2中的极线)
    lines2 = (F @ pts1_homo.T).T  # (N, 3)
    
    # 2. 计算代数误差 x'Fx (N,)
    # 方法: sum(x2 * lines2, axis=1)
    algebraic_error = np.sum(pts2_homo * lines2, axis=1)
    
    # 3. 计算几何距离 (点到直线的距离公式)
    # line = [a, b, c], pt = [u, v, 1] -> dist = |au + bv + c| / sqrt(a^2 + b^2)
    # lines2 的前两列是 a, b
    denom = np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2)
    
    # 避免除以0
    denom[denom < 1e-8] = 1e-8
    
    errors = np.abs(algebraic_error) / denom
    
    return errors

# =========================================================
# 3.4: RANSAC 循环实现
# =========================================================
def ransac_fundamental_matrix(pts1, pts2, threshold=1.0, max_iters=1000):
    """
    RANSAC 估计基础矩阵
    输入:
        pts1, pts2: 匹配点对 (Nx2)
        threshold: 内点判定的距离阈值 (像素单位)
        max_iters: 最大迭代次数
    输出:
        best_F: 最佳基础矩阵
        best_mask: 内点掩码 (bool array)
    """
    N = pts1.shape[0]
    best_inliers_count = 0
    best_F = None
    best_mask = np.zeros(N, dtype=bool)

    # 如果点数少于8个，无法计算
    if N < 8:
        return None, best_mask
    
    for i in range(max_iters):
        # 1. 随机采样 8 个点
        indices = np.random.choice(N, 8, replace=False)
        p1_sample = pts1[indices]
        p2_sample = pts2[indices]
        
        
        # 2. 使用 8 点法估计模型
        F = solve_F_8point(p1_sample, p2_sample)
        
        if F is None:  # 无法计算出有效的 F
            continue
            
        # 3. 计算所有点的误差
        errors = compute_epipolar_errors(F, pts1, pts2)
        
        # 4. 统计内点
        mask = errors < threshold
        inliers_count = np.sum(mask)
        
        # 5. 更新最佳模型
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_F = F
            best_mask = mask

    # 6. 最终优化
    # 使用所有内点重新计算 F，以获得更稳定的结果
    if best_inliers_count >= 8:
        pts1_inliers = pts1[best_mask]
        pts2_inliers = pts2[best_mask]
        best_F = solve_F_8point(pts1_inliers, pts2_inliers)
            
    print(f"RANSAC Finished. Inliers: {best_inliers_count}/{N}, Iters: {max_iters}")
    
    return best_F, best_mask

# =========================================================
# 3.5: 位姿估计
# =========================================================
def estimate_pose(pts1, pts2, K):
    """
    计算本质矩阵 E, 并恢复 R, t。
    """
    # 1. 计算本质矩阵 (Essential Matrix)
    # 使用 RANSAC 剔除误匹配
    # threshold=1.0 表示点到极线的距离阈值 (像素单位)
    # 首先估计基础矩阵 (Fundamental Matrix)
    F, mask = ransac_fundamental_matrix(pts1, pts2, threshold=1.0, max_iters=1000)
    # 然后计算本质矩阵
    E = K.T @ F @ K
    
    # 获取 RANSAC 的内点 (Inliers)
    pts1_inliers = pts1[mask.ravel() == 1]
    pts2_inliers = pts2[mask.ravel() == 1]
    print(f"[Info] Inliers after RANSAC: {len(pts1_inliers)} / {len(pts1)}")

    # 2. 恢复姿态 (Recover Pose)
    # 从 E 分解出 R 和 t，并自动进行手性校验 (Cheirality Check)
    # 确保返回的解使得大多数点在两个相机前方
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)

    return R, t, pts1_inliers, pts2_inliers

# =========================================================
# 4: 三角测量
# =========================================================
def triangulate(P1, P2, pts1, pts2):
    """
    线性三角测量 (DLT Algorithm)
    输入: 投影矩阵 P1, P2 (3x4), 匹配点 (Nx2)
    输出: 3D 点云 (Nx3)
    """
    N = pts1.shape[0]
    points_3d = []
    
    for i in range(N):
        u1, v1 = pts1[i]
        u2, v2 = pts2[i]
        
        # --- 构建矩阵 A (4x4) ---
        # 对应公式: x x PX = 0  =>  u p3^T X - p1^T X = 0
        # Row 1: u1 * pi1_3T - pi1_1T
        # Row 2: v1 * pi1_3T - pi1_2T
        # Row 3: u2 * pi2_3T - pi2_1T
        # Row 4: v2 * pi2_3T - pi2_2T
        
        A = np.zeros((4, 4))
        
        # P1 的行向量 (P1[0], P1[1], P1[2] 分别对应公式中的 pi1^1T, pi1^2T, pi1^3T)
        A[0] = u1 * P1[2] - P1[0]
        A[1] = v1 * P1[2] - P1[1]
        
        # P2 的行向量
        A[2] = u2 * P2[2] - P2[0]
        A[3] = v2 * P2[2] - P2[1]
        
        # --- SVD 求解 AX = 0 ---
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1] # X is [X, Y, Z, w]
        
        # --- 齐次坐标归一化 ---
        # 避免除以 0
        if abs(X[3]) > 1e-6:
            points_3d.append(X[:3] / X[3])
        else:
            # 如果 w 接近 0，说明点在无穷远或者数值不稳定
            points_3d.append(X[:3]) 
            
    return np.array(points_3d)

# =========================================================
# 5: 可视化与后处理
# =========================================================
def visualize_point_cloud(points_3d):
    """
    使用 Matplotlib 绘制 3D 散点图，并强制保持坐标轴等比例。
    """
    # 1. 简单的过滤：剔除距离过远的点 (可能是噪声)
    # 根据数据集尺度调整阈值，这里假设物体在相机前方 0~50 单位内
    # 如果你的点云显示为空，请检查这里的阈值是否太严格
    # mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 50)
    # points_3d = points_3d[mask]

    if len(points_3d) == 0:
        print("[Warning] No points to visualize after filtering!")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 2. 绘制点云
    # c=points_3d[:, 2] 将深度映射为颜色，增加立体感
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
               c=points_3d[:, 2], cmap='viridis', s=50, alpha=0.5, marker='.')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Z Axis (Depth)')
    ax.set_zlabel('Y Axis')
    ax.set_title('Sparse 3D Reconstruction')
    
    # 3. 设置视点
    # elev=-80, azim=-90 通常适合俯视观察 SfM 重建结果
    ax.view_init(elev=-80, azim=-90)

    # 4. 强制坐标轴比例一致  
    # 获取当前自动生成的坐标轴范围
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    limits = np.array([x_limits, y_limits, z_limits])
    
    # 计算包围盒的中心和最大半径
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    
    # 强制设置所有轴的范围一致，以中心点向外扩展
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    plt.show()

# =============================================================
# 主程序流程
# =============================================================
if __name__ == '__main__':
    # 路径配置 (请确保目录下有 view1.png, view2.png, K.txt)
    img1_path = 'view1.png' 
    img2_path = 'view2.png'
    k_path = 'K.txt'
    
    # 1. 加载数据
    img1, img2, K = load_data(img1_path, img2_path, k_path)
    
    # =========================================================
    # 将图像左右连接并使用 cv2.imshow 并列显示
    # =========================================================
    # 使用 np.hstack 将两个图像数组在水平方向上堆叠 (Left-Right Concatenation)
    # 前提是两张图片的高度必须一致
    if img1.shape[0] == img2.shape[0]:
        combined_img = np.hstack((img1, img2))
        
        cv2.imshow('Loaded Images (Press any key to continue)', combined_img)
        print("图片已显示，按任意键继续执行后续步骤...")
        cv2.waitKey(0)        # 等待按键
        cv2.destroyAllWindows() # 关闭窗口
    else:
        print("[Warning] 图片高度不一致，跳过拼接显示。")
    # =========================================================



    # 2. 特征提取与匹配
    print("Step 1: Feature Matching...")
    pts1, pts2, matches, kp1, kp2 = extract_and_match_features(img1, img2)
    
    # =========================================================
    # 任务要求 2: 使用 cv2.drawMatches 显示匹配结果
    # 注意：PPT Page 127 使用的是 drawMatchesKnn ，但这里要求改用 drawMatches
    # =========================================================
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow('Feature Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 显示匹配结果
    

    # 3. 位姿估计
    print("Step 2: Estimating Pose...")
    R, t, pts1_inliers, pts2_inliers = estimate_pose(pts1, pts2, K)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)

    # 4. 三角测量
    print("Step 3: Triangulation...")

    # 必须先构建 3x4 的投影矩阵 P = K[R|t]
    # 相机 1：位于原点，无旋转
    # 构造 [I | 0]
    I = np.eye(3)
    zeros = np.zeros((3, 1))
    P1 = K @ np.hstack((I, zeros))
    
    # 相机 2：通过 estimate_pose 恢复的 R 和 t
    # 构造 [R | t]
    P2 = K @ np.hstack((R, t))
    
    # 传入正确的 P1, P2
    points_3d = triangulate(P1, P2, pts1_inliers, pts2_inliers)

    # 5. 可视化
    print("Step 4: Visualizing...")
    visualize_point_cloud(points_3d)