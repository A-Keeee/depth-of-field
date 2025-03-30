import cv2
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from numba import njit, prange
from queue import Queue, Empty

@njit(parallel=True)
def accumulate_and_compute(sum_gi_srci, sum_gi, weighted_src, gradient_img, current_img):
    height, width, channels = sum_gi_srci.shape
    for i in prange(height):
        for j in range(width):
            for k in prange(channels):
                sum_gi_srci[i, j, k] += weighted_src[i, j, k]
            sum_gi[i, j] += gradient_img[i, j]
            for k in prange(channels):
                current_img[i, j, k] = sum_gi_srci[i, j, k] / (sum_gi[i, j] + 1e-6)
    return

@njit(parallel=True)
def clip_image(current_img):
    height, width, channels = current_img.shape
    for i in prange(height):
        for j in range(width):
            for k in prange(channels):
                if current_img[i, j, k] < 0:
                    current_img[i, j, k] = 0
                elif current_img[i, j, k] > 255:
                    current_img[i, j, k] = 255
    return current_img

def get_pre_deal_img(src_img):
    if len(src_img.shape) == 3 and src_img.shape[2] == 3:
        gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = src_img.copy()

    x_img = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
    y_img = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_REPLICATE)
    gradient_magnitude = np.sqrt(np.square(x_img.astype(np.float32)) + np.square(y_img.astype(np.float32)))
    blurred_img = cv2.GaussianBlur(gradient_magnitude, (41, 41), 0)

    return blurred_img

def enhance_image(image, saturation_factor=1.5, sharpness_factor=1.5, contrast_factor=1.3):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation_factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    saturated_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    blurred = cv2.GaussianBlur(saturated_image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(saturated_image, 1 + sharpness_factor, blurred, -sharpness_factor, 0)
    contrasted = cv2.convertScaleAbs(sharpened, alpha=contrast_factor, beta=-30)

    return contrasted

class VideoProcessor:
    def __init__(self, frame, gradient_img):
        self.sum_gi_srci = frame.astype(np.float32) * gradient_img[:, :, np.newaxis]
        self.sum_gi = gradient_img.copy()
        self.current_img = self.sum_gi_srci / (self.sum_gi[:, :, np.newaxis] + 1e-6)
        clip_image(self.current_img)
        self.current_img = np.clip(self.current_img, 0, 255).astype(np.float32)
        self.enhanced_img = enhance_image(self.current_img.astype(np.uint8), saturation_factor=1.5, sharpness_factor=1.6, contrast_factor=1.5)
        print(f"初始化 VideoProcessor 完成，形状: {self.enhanced_img.shape}, 类型: {self.enhanced_img.dtype}")

def process_frame(frame, video_processor):
    gradient_img = get_pre_deal_img(frame).astype(np.float32) + 1e-6
    src_img_float = frame.astype(np.float32)
    weighted_src = src_img_float * gradient_img[:, :, np.newaxis]

    accumulate_and_compute(video_processor.sum_gi_srci, video_processor.sum_gi, weighted_src, gradient_img, video_processor.current_img)
    clip_image(video_processor.current_img)

    current_img_uint8 = video_processor.current_img.astype(np.uint8)
    enhanced_img = enhance_image(current_img_uint8, saturation_factor=1.5, sharpness_factor=1.6, contrast_factor=1.5)

    return enhanced_img

def producer(cap, frame_queue, stop_event):
    frame_count = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            try:
                frame_queue.put_nowait(None)
            except:
                pass
            break
        frame_count += 1
        cv2.imshow("Original Video", frame)
        if frame_count % 20 == 0:
            try:
                frame_queue.put_nowait(frame)
            except:
                pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            try:
                frame_queue.put_nowait(None)
            except:
                pass
            break

def consumer(frame_queue, video_processor, stop_event):
    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            try:
                frame = frame_queue.get(timeout=0.1)
                if frame is None:
                    break
                future = executor.submit(process_frame, frame, video_processor)
                try:
                    enhanced_img = future.result()
                    cv2.imshow("Merged Image", enhanced_img)
                    video_processor.enhanced_img = enhanced_img
                except Exception as e:
                    print(f"处理帧时发生错误: {e}")
                frame_queue.task_done()
            except Empty:
                if stop_event.is_set():
                    break
                continue

def main():
    video_path = "F:\\depth of field fusion\\video\\9.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件 {video_path}。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频帧率: {fps} FPS")

    window_size_half = (768, 512)
    cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Merged Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original Video", window_size_half[0], window_size_half[1])
    cv2.resizeWindow("Merged Image", window_size_half[0], window_size_half[1])

    frame_count = 0
    video_processor = None
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            return
        frame_count += 1
        cv2.imshow("Original Video", frame)
        if frame_count % 20 == 0:
            gradient_img = get_pre_deal_img(frame).astype(np.float32) + 1e-6
            video_processor = VideoProcessor(frame, gradient_img)
            cv2.imshow("Merged Image", video_processor.enhanced_img)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    frame_queue = Queue(maxsize=10)
    stop_event = threading.Event()

    producer_thread = threading.Thread(target=producer, args=(cap, frame_queue, stop_event), daemon=True)
    producer_thread.start()

    consumer_thread = threading.Thread(target=consumer, args=(frame_queue, video_processor, stop_event), daemon=True)
    consumer_thread.start()

    try:
        while producer_thread.is_alive() or consumer_thread.is_alive():
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                try:
                    frame_queue.put_nowait(None)
                except:
                    pass
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
        try:
            frame_queue.put_nowait(None)
        except:
            pass

    producer_thread.join()
    consumer_thread.join()

    cap.release()
    cv2.destroyAllWindows()

    if video_processor and video_processor.enhanced_img is not None:
        cv2.imwrite("final_img.bmp", video_processor.enhanced_img.astype(np.uint8))
        print("最终图像已保存为 final_img.bmp")

if __name__ == "__main__":
    main()