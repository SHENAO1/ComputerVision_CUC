import cv2
import numpy as np
import os # 用于路径操作

# 202520081000092 - 申奥
# python==3.11.13
# numpy==2.2.6
# opencv-python==4.12.0.88


def my_similarity(im, M, output_shape):
    """
    实现图像的仿射变换，采用逆卷绕（inverse warping）方式
    
    参数：
        im: 输入图像 (numpy array)
        M: 2x3 的仿射变换矩阵。矩阵已包含了所有平移、旋转、缩放，
           
        output_shape: tuple (h, w)，指定输出图像的尺寸。
    返回：
        变换后的完整图像
    """
    out_h, out_w = output_shape

    # 为实现“逆卷绕”，使用变换矩阵的逆矩阵
    # 这个逆矩阵能将新画布上的坐标(x', y')映射回原图的坐标(x, y)
    M_inv = cv2.invertAffineTransform(M)

    # 使用 remap 实现逆映射，根据逆矩阵计算每个输出像素对应的输入坐标
    # 生成新画布所有像素的坐标网格
    x_coords, y_coords = np.meshgrid(np.arange(out_w), np.arange(out_h))
    
    # 添加一个常数维度，用于齐次坐标
    ones = np.ones_like(x_coords) # 修正: 'ones' 变量未定义，需要先创建


    # 将坐标网格展平为 [3, N] 的形式，其中N是总像素数
    coords = np.stack((x_coords, y_coords, ones), axis=-1).reshape(-1, 3).T

    # 逆矩阵作用到新画布的坐标上，得到这些点在原图对应的位置
    src_coords = np.dot(M_inv, coords)

    # 分离 x, y 坐标，并重塑为新画布的尺寸
    map_x = src_coords[0].reshape(out_h, out_w).astype(np.float32)
    map_y = src_coords[1].reshape(out_h, out_w).astype(np.float32)

    # 使用 cv2.remap 根据计算出的映射关系进行插值，生成最终图像
    # BORDER_CONSTANT 表示超出原图边界的区域用黑色填充
    warped = cv2.remap(im, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


if __name__ == "__main__":
    
    # 定义输出目录
    output_dir = "Homework1"
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读图并显示
    # 这里的路径是相对于运行脚本的目录 (HOMEWORK/)
    im = cv2.imread(os.path.join(output_dir, "baboon.png")) 
    if im is None:
        print(f"错误：无法读取图像，请检查文件路径 '{os.path.join(output_dir, 'baboon.png')}' 是否正确。")
        exit()
        
    cv2.imshow("Original", im)
    # 保存原始图像到 Homework1 目录
    cv2.imwrite(os.path.join(output_dir, "original_image.png"), im) 

    # 转灰度图像
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    # 保存灰度图像到 Homework1 目录
    cv2.imwrite(os.path.join(output_dir, "gray_image.png"), gray) 


    # 1. 定义变换参数
    dx, dy, theta, s = 50, 30, 64, 0.92
    h, w = gray.shape[:2]
    center = (w / 2, h / 2)


    # 2. 计算原始的（未经调整的）仿射变换矩阵
    M_orig = cv2.getRotationMatrix2D(center, theta, s)
    M_orig[0, 2] += dx
    M_orig[1, 2] += dy


    # 3. 计算原图四个角点变换后的新坐标
    # 原图角点 (齐次坐标)
    corners = np.array([
        [0, 0, 1],
        [w - 1, 0, 1],
        [w - 1, h - 1, 1],
        [0, h - 1, 1]
    ]).T  # 转置为 3x4 矩阵



    # 应用变换
    transformed_corners = M_orig @ corners


    # 4. 找到新坐标的边界，以确定新画布的大小
    min_x, max_x = np.min(transformed_corners[0]), np.max(transformed_corners[0])
    min_y, max_y = np.min(transformed_corners[1]), np.max(transformed_corners[1])


    # 新画布的宽度和高度，向上取整以确保完整
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))


    # 5. 调整变换矩阵，加入一个额外的平移，将图像移到新画布的(0,0)位置
    # 这个平移量是 (-min_x, -min_y)
    M_adjusted = M_orig.copy()
    M_adjusted[0, 2] -= min_x  # M[0,2] 是 x 方向的平移
    M_adjusted[1, 2] -= min_y  # M[1,2] 是 y 方向的平移

    
    # 将灰度图、调整后的最终矩阵、新画布尺寸传入函数
    transformed = my_similarity(gray, M_adjusted, (new_h, new_w))

    cv2.imshow("Transformed (Complete)", transformed)
    # 保存变换后的完整图像到 Homework1 目录
    cv2.imwrite(os.path.join(output_dir, "transformed_complete.png"), transformed) 
    

    # 为了对比，看一下OpenCV自带函数的效果，结果应该是一样的
    # cv2.warpAffine 需要 (width, height) 格式的尺寸
    opencv_warped = cv2.warpAffine(gray, M_adjusted, (new_w, new_h))
    cv2.imshow("OpenCV warpAffine for comparison", opencv_warped)
    # 保存OpenCV自带函数的结果到 Homework1 目录
    cv2.imwrite(os.path.join(output_dir, "opencv_warpAffine_comparison.png"), opencv_warped) 

    cv2.waitKey(0)
    cv2.destroyAllWindows()

