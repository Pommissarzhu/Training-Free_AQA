video_path = "videos/fisv_001.mp4"
output_path = "output/fisv_001/"
frame_skip_video_path = "frame-skipped_video/"

import cv2
import os
import sys

# 创建输出目录
os.makedirs(output_path, exist_ok=True)
os.makedirs(frame_skip_video_path, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print(f"错误: 无法打开视频文件 {video_path}", file=sys.stderr)
    sys.exit(1)

# 获取视频属性
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"视频信息:")
print(f"- 帧率: {fps:.2f} FPS")
print(f"- 总帧数: {frame_count}")
print(f"- 分辨率: {width}x{height}")

# 初始化抽帧视频写入器
output_video_path = os.path.join(frame_skip_video_path, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
if not out.isOpened():
    print(f"错误: 无法创建输出视频文件 {output_video_path}", file=sys.stderr)
    sys.exit(1)

# 读取并保存每一帧
success, frame = cap.read()
frame_number = 0



while success:
    # 构造输出文件名 (四位数字编号，确保排序正确)
    frame_filename = os.path.join(output_path, f"frame_{frame_number:04d}.jpg")
    
    # 保存帧图像
    # cv2.imwrite(frame_filename, frame)

    # 每10帧抽取1帧写入新视频
    if frame_number % 10 == 0:
        out.write(frame)
        # 保存抽帧图像
        cv2.imwrite(frame_filename, frame)
    
    # 显示进度
    if frame_number % 100 == 0:
        print(f"已保存 {frame_number}/{frame_count} 帧")
    
    # 读取下一帧
    frame_number += 1
    success, frame = cap.read()

# 释放资源
cap.release()
out.release()

print(f"处理完成! 共保存 {frame_number} 帧到 {output_path}")
print(f"抽帧视频已保存到 {output_video_path}")