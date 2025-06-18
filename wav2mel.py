import librosa
import numpy as np

def wav_to_mel(filepath, sample_rate=16000, n_mels=40):
    # 加载音频文件
    y, sr = librosa.load(filepath, sr=sample_rate)  # 自动重采样为 sample_rate

    # 计算 mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sample_rate, 
        n_mels=n_mels, 
        hop_length=512, 
        n_fft=2048
    )

    # 转换为 dB（对数尺度）
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db  # [n_mels, time]


