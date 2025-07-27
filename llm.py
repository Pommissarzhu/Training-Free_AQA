import torch
#from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from get_frames import sliding_window
import os
import json
from datetime import datetime

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct-AWQ", torch_dtype=torch.float16, device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct-AWQ")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

frame_path = os.path.join("output", "fisv_001")
frame_num = 364
img_list = sliding_window(frame_path, frame_num)

for sublist_idx, sublist in enumerate(img_list):
    # 初始化历史上下文（保留最近两轮）
    if 'history' not in locals():
        history = []
    
    # 构造当前轮次消息
    current_message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_url} for img_url in sublist
            ] + [{"type": "text", "text": "现在你是具有多年丰富经验的花样滑冰裁判，多次参加奥运会等顶级赛事的裁判工作。每次我会给你10张图片，请你判断这些图片对应的动作类型是什么（从下面四种类型中选择：1.跳跃Jump、2.旋转Spins、3.步法与移动、4.过渡动作）并简要的评价一下动作完成的情况如何"}]
        }
    ]
    
    # 合并历史上下文（最多两轮）
    messages = history + current_message
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=32768)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])

    # 构造保存的JSON数据（包含时间戳和输出内容）
    save_data = {
        "sublist_idx": sublist_idx,
        "timestamp": datetime.now().isoformat(),
        "output": output_text[0]
    }
    
    # 追加写入JSON文件（每行一个独立JSON对象）
    with open("llm_outputs.json", "a", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False)
        f.write("\n")

    # 构造LLM的回答消息
    assistant_message = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": output_text[0]}]  # 假设output_text是列表形式
        }
    ]
    
    # 更新历史为最近两轮（每轮包含user+assistant消息）
    history = (history + current_message + assistant_message)[-2*2:]  # 每轮2条消息，保留两轮共4条


# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to(model.device)

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)