import os
from gtts import gTTS

def convert_txt_to_tts(txt_path, save_root="output_audio"):
    os.makedirs(save_root, exist_ok=True)

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 2:
            continue  # 跳过无效行
        
        audio_path, korean_text = parts[0], parts[1]
        save_path = os.path.join(save_root, audio_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            tts = gTTS(text=korean_text, lang='ko')
            tts.save(save_path)
            print(f"保存成功: {save_path}")
        except Exception as e:
            print(f"转换失败: {audio_path}，错误：{e}")

# 使用示例
convert_txt_to_tts(r"data\transcript.v.1.4.txt", save_root="tts_outputs")
