import torch
from model import VoiceEncoder
from wav2mel import wav_to_mel
import os
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VoiceEncoder()
model.load_state_dict(torch.load("voice_encoder_best.pth", map_location=device))
model.eval().to(device)

# 构建声纹库
standard_dir = r'data\kss\1'
voicebank = []
for file in os.listdir(standard_dir):
    mel = torch.tensor(wav_to_mel(os.path.join(standard_dir, file))).unsqueeze(0).to(device)
    emb = model(mel).detach().cpu()
    voicebank.append(emb)

def score(audio_path):
    mel = torch.tensor(wav_to_mel(audio_path)).unsqueeze(0).to(device)
    emb = model(mel).detach().cpu()
    sims = [F.cosine_similarity(emb, std_emb).item() for std_emb in voicebank]
    return max(sims) * 100  # 返回最高相似度（0~100）

# 示例：
print(score(r'111(1).m4a'))  # 输出发音标准度评分
