import os
from tqdm import tqdm
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.api import TTS

# 修改torch.load函数
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, weights_only=False, **kwargs)

# 添加白名单
torch.serialization.add_safe_globals({
    "TTS.tts.configs.xtts_config.XttsConfig": XttsConfig,
    "TTS.tts.models.xtts.XttsAudioConfig": XttsAudioConfig,
})

# 初始化 TTS 模型
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

input_txt_path = r"data\transcript.v.1.4.txt"
output_dir = "output_audio"
os.makedirs(output_dir, exist_ok=True)

# 读入txt文件
# 读取文本
with open(input_txt_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 遍历生成语音
for line in tqdm(lines, desc="正在生成语音", unit="句", ncols=100):
    line = line.strip()
    if not line:
        continue
    try:
        fields = line.split('|')
        if len(fields) < 6:
            print(f"⚠️ 格式错误，跳过：{line}")
            continue

        speaker_wav = r"data\kss/"+fields[0]
        # 拼接韩语句子
        text = f"{fields[1]} {fields[2]} {fields[3]}"
        out_path = os.path.join(output_dir, os.path.basename(speaker_wav))

        if not os.path.exists(speaker_wav):
            print(f"❌ 找不到 speaker_wav 文件：{speaker_wav}，跳过。")
            continue

        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="ko",
            file_path=out_path
        )

    except Exception as e:
        print(f"❌ 出错跳过：{line}\n错误信息：{e}")