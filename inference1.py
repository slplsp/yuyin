import torch
import torch.nn.functional as F
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
import matplotlib.pyplot as plt
from model import VoiceEncoder
from wav2mel import wav_to_mel
from openai import OpenAI  # 使用新版SDK
from tenacity import retry, stop_after_attempt, wait_exponential
import json

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = VoiceEncoder()
model.load_state_dict(torch.load("voice_encoder_best.pth", map_location=device))
model.eval().to(device)

# 大模型配置（建议从环境变量获取API密钥）
client = OpenAI(api_key="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")  # 更安全的密钥管理方式
GPT_MODEL = "gpt-4"

# 构建声纹库
standard_dir = r'data\kss\1'
voicebank = []
melbank = []
metadata = []
mel_norms = []

print("Building voice bank...")
for file in os.listdir(standard_dir):
    filepath = os.path.join(standard_dir, file)
    try:
        mel = torch.tensor(wav_to_mel(filepath)).unsqueeze(0).to(device)
        mel_norm = torch.norm(mel).item()
        
        with torch.no_grad():
            emb = model(mel).detach().cpu()
        
        voicebank.append(emb)
        melbank.append(mel.squeeze(0).cpu().numpy())
        metadata.append(filepath)
        mel_norms.append(mel_norm)
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        continue

# 计算动态距离缩放因子
scale_factor = max(1, np.median(mel_norms) / 10)
print(f"Calculated scale factor: {scale_factor:.2f}")

# 将声纹库转换为tensor
voicebank_tensor = torch.cat(voicebank, dim=0)

def plot_dtw_alignment(test_mel, ref_mel, distance):
    """可视化DTW对齐路径"""
    distance, path = fastdtw(test_mel.T, ref_mel.T, dist=euclidean)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(test_mel.T, aspect='auto', origin='lower', cmap='viridis')
    plt.title("Your Pronunciation")
    
    plt.subplot(2, 1, 2)
    plt.imshow(ref_mel.T, aspect='auto', origin='lower', cmap='viridis')
    plt.title("Reference Pronunciation")
    
    plt.figure(figsize=(8, 6))
    plt.plot(path[0], path[1], 'r-', alpha=0.5)
    plt.title(f"DTW Alignment Path (Distance: {distance:.2f})")
    plt.xlabel("Test Frame")
    plt.ylabel("Reference Frame")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_score(cosine_sim, dtw_distance, scale):
    """改进的评分函数"""
    norm_cosine = (cosine_sim + 1) / 2
    penalty = np.exp(-dtw_distance / scale)
    score = (0.6 * norm_cosine + 0.4 * penalty) * 100
    return max(10, min(100, score))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_ai_feedback(analysis_data):
    """调用大模型API获取专业建议（使用新版API）"""
    prompt = f"""
    作为专业语音教练，请根据以下发音分析提供改进建议：
    {json.dumps(analysis_data, indent=2)}
    
    请用中文列出3-5条具体改进建议，包括：
    1. 发音问题定位（如元音/辅音不准、语调问题等）
    2. 针对性练习方法
    3. 常见错误纠正技巧
    """
    
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "你是一位经验丰富的语音训练专家"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content  # 注意新版API的响应结构变化

def score_with_alignment(audio_path, top_k=3, plot_alignment=True):
    """完整评分流程"""
    try:
        # 1. 特征提取
        test_mel = torch.tensor(wav_to_mel(audio_path)).unsqueeze(0).to(device)
        test_mel_np = test_mel.squeeze(0).cpu().numpy()
        
        # 2. 粗筛
        with torch.no_grad():
            test_emb = model(test_mel).detach().cpu()
        
        sims = F.cosine_similarity(test_emb, voicebank_tensor)
        topk_values, topk_indices = torch.topk(sims, k=min(top_k, len(voicebank_tensor)))
        
        # 3. 精对齐
        best_score = 0
        best_result = None
        for ref_idx in topk_indices:
            ref_mel = melbank[ref_idx]
            distance, _ = fastdtw(test_mel_np.T, ref_mel.T, dist=euclidean)
            normalized_dist = distance / test_mel_np.shape[1]
            
            current_score = calculate_score(
                sims[ref_idx].item(),
                normalized_dist,
                scale_factor
            )
            
            if current_score > best_score:
                best_score = current_score
                best_result = {
                    'score': current_score,
                    'cosine_sim': sims[ref_idx].item(),
                    'dtw_distance': distance,
                    'norm_distance': normalized_dist,
                    'ref_audio': metadata[ref_idx]
                }
        
        # 4. 可视化
        if plot_alignment and best_result:
            plot_dtw_alignment(test_mel_np, melbank[topk_indices[0]], best_result['dtw_distance'])
        
        # 5. 获取AI建议
        if best_result:
            analysis_data = {
                "score": best_result['score'],
                "similarity_analysis": {
                    "cosine_similarity": best_result['cosine_sim'],
                    "interpretation": ">0.8:优秀, 0.6-0.8:良好, <0.6:需改进"
                },
                "alignment_analysis": {
                    "dtw_distance": best_result['dtw_distance'],
                    "normalized_distance": best_result['norm_distance'],
                    "ref_audio": best_result['ref_audio']
                }
            }
            ai_feedback = get_ai_feedback(analysis_data)
            best_result['ai_feedback'] = ai_feedback
        
        return best_result
    
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return None

# 示例使用
if __name__ == "__main__":
    test_audio = r'output.mp3'
    
    print("Starting evaluation...")
    result = score_with_alignment(test_audio)
    
    if result:
        print("\n=== 发音分析结果 ===")
        print(f"综合评分: {result['score']:.1f}/100")
        print(f"声学相似度: {result['cosine_sim']:.2f}")
        print(f"对齐距离: {result['dtw_distance']:.2f}")
        
        print("\n=== AI改进建议 ===")
        print(result['ai_feedback'])