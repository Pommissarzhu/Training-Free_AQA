import os

frame_path = os.path.join("output", "fisv_001")

def sliding_window(frame_num, size=10, step=5):
    window_list = []
    # 遍历起始位置，步长为step
    start = 0
    while start + size <= frame_num:
        end = start + size
        # 生成当前窗口的所有帧索引（start到end-1）
        window_indices = range(start, end)
        # 生成对应的帧路径（使用四位数字编号）
        window_paths = [os.path.join(frame_path, f"{idx:03d}0.jpg") for idx in window_indices]
        window_list.append(window_paths)
        # 移动到下一个窗口起始位置
        start += step
    # 生成当前窗口的所有帧索引（start到end-1）
    window_indices = range(frame_num-size, frame_num)
    # 生成对应的帧路径（使用四位数字编号）
    window_paths = [os.path.join(frame_path, f"{idx:03d}0.jpg") for idx in window_indices]
    window_list.append(window_paths)
    return window_list

if __name__ == "__main__":
    # 检查目录是否存在
    if not os.path.exists(frame_path):
        print(f"错误: 目录 {frame_path} 不存在")
        exit(1)

    # 获取目录下所有JPG文件
    jpg_files = [f for f in os.listdir(frame_path) if f.lower().endswith(".jpg")]

    # 统计数量
    frame_count = len(jpg_files)

    print(f"目录 {frame_path} 下共有 {frame_count} 个JPG视频帧")

    f_list = sliding_window(frame_num=frame_count)
    print(f_list)
