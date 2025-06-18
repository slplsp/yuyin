import torch
import torch.nn.functional as F
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
import matplotlib.pyplot as plt
from model import VoiceEncoder
from wav2mel import wav_to_mel

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = VoiceEncoder()
model.load_state_dict(torch.load("voice_encoder_best.pth", map_location=device))
model.eval().to(device)

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

def plot_dtw_alignment(test_mel, ref_mel, distance, path):
    """可视化DTW对齐路径"""
    plt.figure(figsize=(12, 8))
    
    # 绘制测试语音和参考语音的梅尔频谱
    plt.subplot(2, 2, 1)
    plt.imshow(test_mel.T, aspect='auto', origin='lower')
    plt.title("Test Mel Spectrogram")
    plt.xlabel("Frame")
    plt.ylabel("Mel Band")
    
    plt.subplot(2, 2, 2)
    plt.imshow(ref_mel.T, aspect='auto', origin='lower')
    plt.title("Reference Mel Spectrogram")
    plt.xlabel("Frame")
    plt.ylabel("Mel Band")
    
    # 绘制对齐路径
    plt.subplot(2, 1, 2)
    plt.plot(path[0], path[1], 'r-', linewidth=0.5)
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

def score_with_alignment(audio_path, top_k=3, plot_alignment=True):
    """完整评分函数，包含可视化选项"""
    # 加载测试音频
    try:
        test_mel = torch.tensor(wav_to_mel(audio_path)).unsqueeze(0).to(device)
        test_mel_np = test_mel.squeeze(0).cpu().numpy()
    except Exception as e:
        print(f"Error loading test audio: {str(e)}")
        return None

    # 粗筛阶段
    with torch.no_grad():
        test_emb = model(test_mel).detach().cpu()
    
    sims = F.cosine_similarity(test_emb, voicebank_tensor)
    sims = torch.clamp(sims, min=-1.0, max=1.0)
    
    topk_values, topk_indices = torch.topk(sims, k=min(top_k, len(voicebank_tensor)))
    print("\nTop Candidates:")
    for i, idx in enumerate(topk_indices):
        print(f"{i+1}. {metadata[idx]} (cosine={topk_values[i]:.4f})")

    # 精细对齐
    best_score = 0
    best_result = None
    
    for ref_idx in topk_indices:
        ref_mel = melbank[ref_idx]
        
        try:
            distance, path = fastdtw(
                test_mel_np.T,
                ref_mel.T,
                dist=euclidean,
                radius=50
            )
            
            normalized_dist = distance / (len(path) + 1e-6)
            current_score = calculate_score(
                sims[ref_idx].item(),
                normalized_dist,
                scale_factor
            )
            
            if current_score > best_score or best_result is None:
                best_score = current_score
                best_result = {
                    'ref_idx': ref_idx.item(),
                    'score': current_score,
                    'cosine': sims[ref_idx].item(),
                    'distance': distance,
                    'normalized_dist': normalized_dist,
                    'path': path,
                    'test_mel': test_mel_np,
                    'ref_mel': ref_mel
                }
                
        except Exception as e:
            print(f"DTW error for {metadata[ref_idx]}: {str(e)}")
            continue

    if best_result is None:
        print("All alignments failed")
        return None

    # 显示结果
    print("\n===== Final Result =====")
    print(f"Pronunciation Score: {best_result['score']:.2f}/100")
    print(f"Best Match: {metadata[best_result['ref_idx']]}")
    print(f"Cosine Similarity: {best_result['cosine']:.4f}")
    print(f"Raw DTW Distance: {best_result['distance']:.2f}")
    print(f"Normalized Distance: {best_result['normalized_dist']:.4f}")

    # 可视化对齐
    if plot_alignment:
        plot_dtw_alignment(
            best_result['test_mel'],
            best_result['ref_mel'],
            best_result['distance'],
            best_result['path']
        )
    
    return best_result

# 示例使用
if __name__ == "__main__":
    test_audio = r'data\kss\2\2_0000.wav'  # 替换为你的测试音频
    
    print("\nStarting evaluation...")
    result = score_with_alignment(test_audio)
    
    if result is not None:
        print("\nEvaluation completed successfully!")
        print(f"Final pronunciation score: {result['score']:.2f}/100")