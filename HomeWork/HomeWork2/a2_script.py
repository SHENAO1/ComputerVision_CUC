import cv2
import numpy as np
import sys
import os  # 引入 os 模块以处理文件路径

# --- 辅助函数：保存图像 ---
def save_debug_image(filename, image, folder_path):
    """保存图像到指定路径并打印提示信息"""
    full_path = os.path.join(folder_path, filename)
    cv2.imwrite(full_path, image)
    print(f"  [已保存] {filename}")


# --- 步骤 1: 输入 ---

# 获取当前脚本所在的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, 'peppers.png')

print(f"正在尝试读取图像路径: {img_path}")

try:
    # 1.3: 读取图像并转为灰度
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if im is None:
        raise FileNotFoundError(f"错误：找不到图像。请检查文件是否存在于: {img_path}")

    cv2.imshow('1. Grayscale Image (灰度图)', im)
    # 保存灰度图
    save_debug_image('1_grayscale.png', im, script_dir)
    cv2.waitKey(1)

except FileNotFoundError as e:
    print(e)
    sys.exit()
except Exception as e:
    print(f"发生未知错误: {e}")
    sys.exit()

print("正在应用高斯模糊以减少噪声...")
im = cv2.GaussianBlur(im, (5, 5), 1.4)

# 保存模糊后的图像
save_debug_image('1_blurred.png', im, script_dir)

# --- 步骤 2: 计算图像梯度 ---

# 2.1: 使用 sobel 滤波器
im_dx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
im_dy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)

# 2.2: 计算梯度的幅度和方向
grad_mag = cv2.magnitude(im_dx, im_dy)
grad_dir = cv2.phase(im_dx, im_dy, angleInDegrees=True)  # 函数会将计算出的角度转换为度的形式返回。angleInDegrees=True 代表返回角度值

# 2.3: 梯度方向量化
quantized_dir = np.zeros_like(grad_dir, dtype=np.int32)
angle = grad_dir % 180 

quantized_dir[ (angle >= 0)     & (angle < 22.5)  ] = 0
quantized_dir[ (angle >= 157.5) & (angle <= 180)  ] = 0
quantized_dir[ (angle >= 22.5)  & (angle < 67.5)  ] = 45
quantized_dir[ (angle >= 67.5)  & (angle < 112.5) ] = 90
quantized_dir[ (angle >= 112.5) & (angle < 157.5) ] = 135

# 2.4: 显示并保存梯度幅度和方向
grad_mag_display = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imshow('2a. Gradient Magnitude (梯度幅度)', grad_mag_display)
# 保存梯度幅度图
save_debug_image('2a_gradient_magnitud.png', grad_mag_display, script_dir)
cv2.waitKey(1)

grad_dir_display = (quantized_dir / 45 * 64).astype(np.uint8)
cv2.imshow('2b. Quantized Gradient Direction (量化后的梯度方向)', grad_dir_display)
# 保存方向图
save_debug_image('2b_gradient_direction.png', grad_dir_display, script_dir)
cv2.waitKey(1)


# --- 步骤 3: 执行非极大值抑制 ---
print("正在执行非极大值抑制 (可能需要一点时间)...")
M, N = grad_mag.shape
nms_result = np.zeros((M, N), dtype=np.float32)

for i in range(1, M - 1):
    for j in range(1, N - 1):
        direction = quantized_dir[i, j]

        if direction == 0:  # 梯度方向是水平，梯度垂直于边缘，边缘本身垂直；沿着梯度方向（法线）进行比较
            neighbor1, neighbor2 = grad_mag[i, j - 1], grad_mag[i, j + 1]
        elif direction == 90: # 梯度方向是垂直，边缘本身水平，沿着梯度方向（法线）进行比较
            neighbor1, neighbor2 = grad_mag[i - 1, j], grad_mag[i + 1, j]
        elif direction == 45: #  对角
            neighbor1, neighbor2 = grad_mag[i - 1, j + 1], grad_mag[i + 1, j - 1]
        elif direction == 135: # 反对角
            neighbor1, neighbor2 = grad_mag[i - 1, j - 1], grad_mag[i + 1, j + 1]
        else:
            neighbor1, neighbor2 = 0, 0

        if grad_mag[i, j] >= neighbor1 and grad_mag[i, j] >= neighbor2:
            nms_result[i, j] = grad_mag[i, j]
        else:
            nms_result[i, j] = 0

# 显示并保存非极大值抑制结果
nms_result_display = cv2.normalize(nms_result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imshow('3. Non-Maximum Suppression (非极大值抑制)', nms_result_display)
# 保存 NMS 结果
save_debug_image('3_nms_result.png', nms_result_display, script_dir)
cv2.waitKey(1)


# --- 步骤 4: 阈值化并连接 (滞后阈值法) ---
print("正在执行双阈值连接...")

# 4.1: 定义两个阈值
thresh_high = 100 # 高于此值肯定是边缘
thresh_low = 50   # 低于此值肯定不是边缘，介于两者之间是“待定”

# 4.2: 初步分类
im_thresh_high = (nms_result >= thresh_high).astype(np.float32)
im_thresh_low = (nms_result >= thresh_low).astype(np.float32) 

# 4.3: 标记
# BW = 1.0  -> 强边缘 (Strong Edge)
# BW = 0.5  -> 弱边缘 (Weak Edge)
# BW = 0.0  -> 非边缘
BW = (im_thresh_high + im_thresh_low) / 2.0

strong_edges_r, strong_edges_c = np.where(BW == 1.0)
queue = list(zip(strong_edges_r, strong_edges_c))

while len(queue) > 0:
    r, c = queue.pop(0)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0: continue
            nr, nc = r + i, c + j
            if 0 <= nr < M and 0 <= nc < N and BW[nr, nc] == 0.5:
                BW[nr, nc] = 1.0
                queue.append((nr, nc))

BW[BW == 0.5] = 0.0

final_edges = (BW * 255).astype(np.uint8)
cv2.imshow('4. Final Canny Edges (最终边缘图像)', final_edges)
# 保存最终结果
save_debug_image('4_final_canny_edges.png', final_edges, script_dir)

print("-" * 30)
print(f"所有图片已保存至: {script_dir}")
print("请查看弹出的窗口，按键盘任意键关闭所有图像窗口。")
cv2.waitKey(0)
cv2.destroyAllWindows()