import cv2
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class ImagePyramidInfo:
    def __init__(self, image, grad_size=25, num_layers=6):
        self.mean_focus = self._make_mean_focus(image, grad_size)
        self.gaussian_pyramid = self._make_gaussian_pyramid(image.astype(np.float32) / 255.0, num_layers)
        self.laplacian_pyramid = self._make_laplacian_pyramid(self.gaussian_pyramid)

    def _make_mean_focus(self, image, grad_size):
        if image.dtype != np.uint8:
            raise ValueError("Input image must be of type uint8.")

        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        grad_rows = (gray.shape[0] + grad_size - 1) // grad_size
        grad_cols = (gray.shape[1] + grad_size - 1) // grad_size
        mean_focus = np.zeros((grad_rows, grad_cols), dtype=np.float32)

        for r in range(grad_rows):
            for c in range(grad_cols):
                y_start = r * grad_size
                y_end = min((r + 1) * grad_size, gray.shape[0])
                x_start = c * grad_size
                x_end = min((c + 1) * grad_size, gray.shape[1])

                roi = gray[y_start:y_end, x_start:x_end]
                blurred = cv2.GaussianBlur(roi, (3, 3), 0)
                sobel = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
                mean_focus[r, c] = np.mean(sobel)

        return mean_focus

    def _make_gaussian_pyramid(self, image, num_layers):
        gaussian_pyramid = [image.copy()]
        for _ in range(1, num_layers):
            image = cv2.pyrDown(image)
            gaussian_pyramid.append(image)
        return gaussian_pyramid

    def _make_laplacian_pyramid(self, gaussian_pyramid):
        laplacian_pyramid = []
        num_layers = len(gaussian_pyramid)
        for i in range(num_layers - 1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            gaussian_up = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyramid[i], gaussian_up)
            laplacian_pyramid.append(laplacian)
        laplacian_pyramid.append(gaussian_pyramid[-1].copy())  # 最后一层与高斯金字塔相同
        return laplacian_pyramid

    def make_mask_pyramid(self, mask, num_layers):
        mask_pyramid = [mask.copy()]
        for _ in range(1, num_layers):
            mask = cv2.pyrDown(mask)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)  # 应用高斯模糊平滑掩码
            mask_pyramid.append(mask)
        self.mask_pyramid = mask_pyramid

    def get_mean_focus(self):
        return self.mean_focus

    def get_gaussian_pyramid(self):
        return self.gaussian_pyramid

    def get_laplacian_pyramid(self):
        return self.laplacian_pyramid

    def get_mask_pyramid(self):
        return self.mask_pyramid if hasattr(self, 'mask_pyramid') else None

def compute_pyramid_info(image, grad_size, num_layers):
    return ImagePyramidInfo(image, grad_size, num_layers)

def multi_focus_band_blender_multithreaded(images, grad_size=25, num_layers=6, num_threads=4):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(compute_pyramid_info, img, grad_size, num_layers) for img in images]
        pyramid_infos = []
        for future in as_completed(futures):
            try:
                pyramid_info = future.result()
                pyramid_infos.append(pyramid_info)
            except Exception as e:
                print(f"处理图像时发生错误: {e}")

    # 计算每幅图像的平均聚焦矩阵
    mean_focus_list = [p.mean_focus for p in pyramid_infos]
    mean_focus_stack = np.stack(mean_focus_list, axis=0)  # Shape: (num_images, rows, cols)

    # 确定每个网格的最佳聚焦图像
    best_indices = np.argmax(mean_focus_stack, axis=0)  # Shape: (rows, cols)

    # 生成掩码（浮点数，0.0 或 1.0）
    masks = [ (best_indices == i).astype(np.float32) for i in range(len(images)) ]

    # 构建掩码金字塔
    for i, p in enumerate(pyramid_infos):
        p.make_mask_pyramid(masks[i], num_layers)

    # 构建拉普拉斯金字塔融合
    blended_pyramid = []
    for layer in range(num_layers):
        blended_layer = np.zeros_like(pyramid_infos[0].laplacian_pyramid[layer])
        for i in range(len(images)):
            mask = pyramid_infos[i].mask_pyramid[layer]
            # 将掩码调整为与拉普拉斯金字塔层相同的大小
            mask_resized = cv2.resize(mask,
                                      (pyramid_infos[i].laplacian_pyramid[layer].shape[1],
                                       pyramid_infos[i].laplacian_pyramid[layer].shape[0]),
                                      interpolation=cv2.INTER_LINEAR)  # 使用线性插值
            if len(mask_resized.shape) == 2:
                mask_resized = mask_resized[:, :, np.newaxis]  # 添加通道维度
            elif len(mask_resized.shape) == 3 and mask_resized.shape[2] == 1:
                pass  # 已经是单通道
            else:
                raise ValueError(f"Unexpected mask shape: {mask_resized.shape}")
            blended_layer += pyramid_infos[i].laplacian_pyramid[layer] * mask_resized
        blended_pyramid.append(blended_layer)
        # 可视化每层融合结果（可选）
        # cv2.imwrite(f"blended_pyramid_layer_{layer}.bmp", (blended_layer * 255).astype(np.uint8))

    # 重建融合图像
    result = blended_pyramid[-1]
    for layer in range(num_layers - 2, -1, -1):
        result = cv2.pyrUp(result, dstsize=blended_pyramid[layer].shape[:2][::-1])
        result += blended_pyramid[layer]
        # 可视化重建过程（可选）
        # cv2.imwrite(f"reconstructed_layer_{layer}.bmp", cv2.convertScaleAbs(result * 255))

    # 转换为uint8
    result = cv2.convertScaleAbs(result * 255)

    return result

def sharpen_image(image, strength=2.0, iterations=1):
    """
    应用Unsharp Masking进行图像锐化，多次迭代。
    """
    sharpened = image.copy()
    for _ in range(iterations):
        blurred = cv2.GaussianBlur(sharpened, (0, 0), sigmaX=1)
        sharpened = cv2.addWeighted(sharpened, 1 + strength, blurred, -strength, 0)
    return sharpened

def gamma_correction(image, gamma=1.0):
    """
    应用伽马校正以调整图像亮度。
    """
    if gamma <= 0:
        gamma = 1.0
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def denoise_image(image, kernel_size=(3, 3), sigmaX=0.3):
    """
    使用高斯滤波进行去噪，参数可调。
    """
    denoised = cv2.GaussianBlur(image, kernel_size, sigmaX=sigmaX)
    return denoised

def apply_clahe_to_luminance(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    应用自适应直方图均衡（CLAHE）仅增强亮度通道的对比度，保持色彩不变。

    参数:
        image (numpy.ndarray): 输入的BGR图像。
        clip_limit (float): CLAHE的对比度限制。
        tile_grid_size (tuple): CLAHE的网格大小。

    返回:
        numpy.ndarray: 对比度增强后的BGR图像。
    """
    # 检查图像是否为彩色图像
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是彩色图像（BGR）。")

    # 将BGR图像转换为LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 应用CLAHE到亮度通道
    l_clahe = clahe.apply(l)

    # 合并CLAHE增强后的亮度通道与原始色彩通道
    lab_clahe = cv2.merge((l_clahe, a, b))

    # 将LAB图像转换回BGR颜色空间
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return enhanced_image


def enhance_saturation(image, saturation_scale=1.3):
    """
    增强图像的颜色饱和度，使颜色更加鲜艳。

    参数:
        image (numpy.ndarray): 输入的BGR图像。
        saturation_scale (float): 饱和度增强比例（>1 增加饱和度，<1 减少饱和度）。

    返回:
        numpy.ndarray: 颜色饱和度增强后的BGR图像。
    """
    # 将BGR图像转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 增强饱和度
    hsv[:, :, 1] *= saturation_scale

    # 限制饱和度范围在 [0, 255]
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    # 转换回BGR颜色空间
    enhanced_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return enhanced_image


def apply_gamma_correction(image, gamma=1.2):
    """
    对图像应用伽马矫正，用于调整图像亮度。

    参数:
        image (numpy.ndarray): 输入的BGR图像。
        gamma (float): 伽马值，影响亮度的调整。

    返回:
        numpy.ndarray: 伽马矫正后的BGR图像。
    """
    # 将图像的像素值归一化到[0, 1]范围
    image_normalized = image / 255.0

    # 应用伽马矫正
    image_corrected = np.power(image_normalized, gamma)

    # 将图像的像素值恢复到[0, 255]范围，并转换为整数
    image_corrected = np.uint8(image_corrected * 255)

    return image_corrected


def get_average_sampled_images(folder_path, num_samples=50):
    """
    从指定文件夹中平均抽取指定数量的图片文件，并返回完整的文件路径。

    参数:
        folder_path (str): 图片文件夹的路径。

    返回:
        list: 抽取的图片完整路径列表。
    """
    # 获取文件夹中的所有图片文件，支持大小写的扩展名
    img_files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # 对文件进行排序（可选，根据需要排序）
    img_files.sort()

    total_images = len(img_files)

    if total_images == 0:
        print("文件夹中没有找到任何图片文件。")
        return []

    if total_images <= num_samples:
        print(f"文件夹中的图片数量 ({total_images}) 少于或等于 {num_samples}，将返回所有图片。")
        # 返回完整路径
        return [os.path.join(folder_path, f) for f in img_files]
    else:
        # 计算抽取的索引，确保均匀分布
        indices = np.linspace(0, total_images - 1, num=num_samples, dtype=int)
        selected_files = [os.path.join(folder_path, img_files[i]) for i in indices]
        print(f"从 {total_images} 张图片中平均抽取了 {num_samples} 张。")
        return selected_files


def main():
    # 开始计时
    start_time = time.time()

    # 图片文件夹路径
    folder_path = "F:\\depth of field fusion\\images3"  # 替换为你存放图片的文件夹路径

    sampled_image_paths= get_average_sampled_images(folder_path, num_samples=30)

    print(sampled_image_paths)


    # 验证是否成功加载图像
    frames = []
    for img_path in sampled_image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            frames.append(img)
            # print(f"成功加载图像: {img_path}")
        else:
            print(f"无法加载图像: {img_path}")



    if not frames:
        print("错误：没有加载有效的图像。")
        return

    # 确保所有帧尺寸一致
    reference_size = frames[0].shape[:2]  # (height, width)
    frames = [cv2.resize(frame, (reference_size[1], reference_size[0])) for frame in frames]

    # 多聚焦多频段图像融合（多线程）
    fused_image = multi_focus_band_blender_multithreaded(
        frames,
        grad_size=10,     # 减小梯度网格大小
        num_layers=5,      # 增加金字塔层数
        num_threads=4      # 线程数改为4
    )


    # 结束计时
    end_time = (time.time() - start_time) * 1000
    print(f"处理时间：{int(end_time)} 毫秒")

    # # 保存最终融合后的图像
    # cv2.imwrite("mergeImageColor_from_video_optimized.bmp", fused_image)
    # print("融合后的图像已保存为 'mergeImageColor_from_video_optimized.bmp'。")


    # # 检查融合后的图像是否存在
    # if not os.path.exists(fused_image_path):
    #     print(f"错误：融合后的图像文件 {fused_image_path} 不存在。")
    #     return
    #
    # # 加载融合后的图像
    # fused_image = cv2.imread(fused_image_path)
    # if fused_image is None:
    #     print(f"错误：无法加载图像 {fused_image_path}。")
    #     return

    # print("成功加载融合后的图像。")

    # 2. 对比度增强（CLAHE）
    try:
        enhanced_contrast = apply_clahe_to_luminance(fused_image, clip_limit=2.0, tile_grid_size=(4,4))
        # cv2.imwrite("contrast_enhanced_image.bmp", enhanced_contrast)
        # print("对比度增强完成，并保存为 'contrast_enhanced_image.bmp'。")
    except Exception as e:
        print(f"对比度增强过程中发生错误: {e}")

    # 3. 颜色饱和度增强
    try:
        enhanced_saturation = enhance_saturation(enhanced_contrast, saturation_scale=1.1)
        # cv2.imwrite("saturation_enhanced_image.bmp", enhanced_saturation)
        # print("颜色饱和度增强完成，并保存为 'saturation_enhanced_image.bmp'。")
    except Exception as e:
        print(f"颜色饱和度增强过程中发生错误: {e}")

    # 4. 伽马矫正
    try:
        gamma_corrected_image = apply_gamma_correction(enhanced_saturation, gamma=1.27)
        cv2.imwrite("corrected_image.bmp", gamma_corrected_image)
        # print("伽马矫正完成，并保存为 'gamma_corrected_image.bmp'。")
    except Exception as e:
        print(f"伽马矫正过程中发生错误: {e}")

    print("所有后处理步骤完成。")




if __name__ == "__main__":
    main()