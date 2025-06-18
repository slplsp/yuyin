from flask import Flask, request, jsonify, send_from_directory, send_file
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import uuid
import torch
import torch.nn.functional as F
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from model import VoiceEncoder
from wav2mel import wav_to_mel
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import soundfile as sf
import io
import logging
import wave

app = Flask(__name__, static_folder='static')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'audio_history'
CACHE_FOLDER = 'cache'
HISTORY_FILE = 'history.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER
app.config['CACHE_FOLDER'] = CACHE_FOLDER

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载语音编码模型
model = VoiceEncoder()
model.load_state_dict(torch.load("voice_encoder_best.pth", map_location=device))
model.eval().to(device)

# 初始化OpenAI客户端
client = OpenAI(api_key="xxx") #openai api
GPT_MODEL = "gpt-4o-mini-2024-07-18"

# 构建声纹库
standard_dir = r'output_audio\1'
voicebank = []
melbank = []
metadata = []
mel_norms = []

logger.info("正在构建声纹库...")
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
        logger.error(f"处理文件 {filepath} 时出错: {str(e)}")
        continue

# 计算动态距离缩放因子
scale_factor = max(1, np.median(mel_norms) / 10)
logger.info(f"计算得到的缩放因子: {scale_factor:.2f}")

# 将声纹库转换为tensor
voicebank_tensor = torch.cat(voicebank, dim=0)

def calculate_score(cosine_sim, dtw_distance, scale):
    """改进的评分函数"""
    norm_cosine = (cosine_sim + 1) / 2
    penalty = np.exp(-dtw_distance / scale)
    score = (0.6 * norm_cosine + 0.4 * penalty) * 100
    return max(10, min(100, score))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_ai_feedback(analysis_data):
    """调用大模型API获取专业建议"""
    prompt = f"""
    전문 발음 코치로서, 아래 한국어 발음 분석을 기반으로 개선 방안을 제시해 주세요：
    {json.dumps(analysis_data, indent=2)}
    
    3~5가지 구체적인 개선 방안을 한국어로 작성해 주세요. 내용에는 다음이 포함되어야 합니다：
    1. 발음 문제의 위치 (예: 특정 모음/자음의 부정확함, 억양 문제 등)
    2. 목표에 맞는 연습 방법
    3. 자주 발생하는 오류를 교정하는 팁
    
    다음 형식으로 개선 방안을 작성해 주세요：
    1. [문제 설명]: [구체적인 제안]
    2. [문제 설명]: [구체적인 제안]
    3. [문제 설명]: [구체적인 제안]
    """
    
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "당신은 발음 문제를 분석하고 구체적인 개선 방안을 제시하는 데 능숙한 경험 많은 발음 훈련 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"AI反馈生成失败: {str(e)}")
        return "未能生成改进建议，请稍后再试"

def evaluate_audio(filepath):
    """语音评分函数"""
    try:
        # 1. 检查文件有效性
        if not os.path.exists(filepath):
            raise Exception("音频文件不存在")
            
        if os.path.getsize(filepath) == 0:
            raise Exception("音频文件为空")
        
        # 2. 特征提取
        try:
            logger.info(f"正在处理文件: {filepath}")
            mel = wav_to_mel(filepath)
            if mel is None or len(mel) == 0:
                raise Exception("梅尔频谱提取失败")
                
            test_mel = torch.tensor(mel).unsqueeze(0).to(device)
            test_mel_np = test_mel.squeeze(0).cpu().numpy()
            
        except Exception as feature_error:
            raise Exception(f"特征提取失败: {str(feature_error)}")
        
        # 3. 粗筛
        try:
            with torch.no_grad():
                test_emb = model(test_mel).detach().cpu()
                
            sims = F.cosine_similarity(test_emb, voicebank_tensor)
            if len(sims) == 0:
                raise Exception("声纹比对失败")
                
            topk_values, topk_indices = torch.topk(sims, k=min(3, len(voicebank_tensor)))
            
        except Exception as comparison_error:
            raise Exception(f"声纹比对失败: {str(comparison_error)}")
        
        # 4. 精对齐
        best_score = 0
        best_result = None
        
        try:
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
                    
            if not best_result:
                raise Exception("未能计算出有效评分")
                
        except Exception as alignment_error:
            raise Exception(f"对齐计算失败: {str(alignment_error)}")
        
        # 5. 获取AI建议
        try:
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
            
            best_result['ai_feedback'] = get_ai_feedback(analysis_data)
            
        except Exception as ai_error:
            logger.error(f"AI反馈生成失败: {str(ai_error)}")
            best_result['ai_feedback'] = "未能生成改进建议: " + str(ai_error)
        
        return best_result
        
    except Exception as e:
        logger.error(f"语音评分失败: {str(e)}", exc_info=True)
        return None

# 历史记录管理函数
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"加载历史记录失败: {str(e)}")
        return []

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存历史记录失败: {str(e)}")

# 录音缓存管理
@app.route('/cache_recording', methods=['POST'])
def cache_recording():
    """将录音数据保存到缓存文件"""
    try:
        if 'audio_data' not in request.json:
            return jsonify({'success': False, 'error': '缺少音频数据'}), 400
        
        # 创建缓存目录
        os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)
        
        # 保存为WAV文件
        cache_path = os.path.join(app.config['CACHE_FOLDER'], 'test.wav')
        
        # 将音频数据转换为字节
        audio_data = bytes(request.json['audio_data'])
        
        # 写入WAV文件
        with wave.open(cache_path, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16位
            wf.setframerate(16000)  # 16kHz采样率
            wf.writeframes(audio_data)
        
        return jsonify({
            'success': True,
            'cache_path': cache_path,
            'message': '录音缓存成功'
        })
    except Exception as e:
        logger.error(f"保存录音缓存失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '保存录音缓存失败'
        }), 500

@app.route('/evaluate_cached', methods=['POST'])
def evaluate_cached():
    """评测缓存中的录音"""
    try:
        cache_path = os.path.join(app.config['CACHE_FOLDER'], 'test.wav')
        
        # 检查缓存文件是否存在
        if not os.path.exists(cache_path):
            return jsonify({
                'success': False,
                'error': '缓存文件不存在',
                'message': '请先录制音频'
            }), 400
        
        # 执行语音评分
        result = evaluate_audio(cache_path)
        if not result:
            raise Exception("语音评分失败，无法生成有效结果")
            
        return jsonify({
            'success': True,
            'result': result,
            'message': '评测成功完成'
        })
        
    except Exception as e:
        logger.error(f"评测缓存录音失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '评测过程中发生错误: ' + str(e)
        }), 500

# 其他路由
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': '未提供音频数据',
            'message': '请选择要评测的音频文件'
        }), 400
    
    temp_audio_path = None
    try:
        # 处理上传文件
        audio_file = request.files['file']
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': '未选择文件',
                'message': '请选择要评测的音频文件'
            }), 400
        
        # 先尝试直接读取文件
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_file.read()))
            audio_file.seek(0)  # 重置文件指针
        except Exception as read_error:
            logger.error(f"上传的音频文件解码失败: {str(read_error)}")
            raise Exception("无法读取上传的音频文件，请检查文件格式")
            
        # 保存为标准的WAV文件
        temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_upload_{uuid.uuid4()}.wav")
        sf.write(temp_audio_path, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    
        # 检查文件是否有效
        if not os.path.exists(temp_audio_path):
            raise Exception("音频文件保存失败")
            
        if os.path.getsize(temp_audio_path) == 0:
            raise Exception("音频文件为空")
            
        # 验证文件格式
        try:
            with sf.SoundFile(temp_audio_path) as f:
                if f.format != 'WAV' or f.subtype != 'PCM_16':
                    raise Exception("音频格式不符合要求")
        except Exception as format_error:
            raise Exception(f"音频格式验证失败: {str(format_error)}")
        
        # 执行语音评分
        result = evaluate_audio(temp_audio_path)
        if not result:
            raise Exception("语音评分失败，无法生成有效结果")
            
        return jsonify({
            'success': True,
            'result': result,
            'message': '评测成功完成'
        })
        
    except Exception as e:
        logger.error(f"评测过程中发生错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '评测过程中发生错误: ' + str(e),
            'debug': {
                'temp_file': temp_audio_path,
                'file_size': os.path.getsize(temp_audio_path) if temp_audio_path and os.path.exists(temp_audio_path) else 0
            }
        }), 500
        
    finally:
        # 确保清理临时文件
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as clean_error:
                logger.error(f"清理临时文件失败: {str(clean_error)}")

@app.route('/save_result', methods=['POST'])
def save_result():
    if 'file' not in request.files or 'result' not in request.form:
        return jsonify({'error': '缺少必要数据'}), 400
    
    try:
        # 生成唯一ID
        result_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存音频文件
        audio_file = request.files['file']
        audio_filename = f"{result_id}.wav"
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
        audio_file.save(audio_path)
        
        # 确保文件保存成功
        if not os.path.exists(audio_path):
            raise Exception("保存音频文件失败")
        
        # 解析结果数据
        result = json.loads(request.form['result'])
        
        # 创建历史记录条目
        history_entry = {
            'id': result_id,
            'timestamp': timestamp,
            'result': result,
            'audio_file': audio_filename
        }
        
        # 更新历史记录
        history = load_history()
        history.insert(0, history_entry)
        save_history(history)
        
        return jsonify({'success': True, 'id': result_id})
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    try:
        history = load_history()
        return jsonify(history)
    except Exception as e:
        logger.error(f"获取历史记录失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    # 安全检查，防止目录遍历攻击
    if '..' in filename or filename.startswith('/'):
        return jsonify({'error': '无效的文件名'}), 400
        
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
    
    if not os.path.exists(audio_path):
        return jsonify({'error': '未找到音频文件'}), 404
    
    return send_file(
        audio_path,
        mimetype='audio/wav',
        as_attachment=False,
        conditional=True
    )

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        # 删除所有音频文件
        for filename in os.listdir(AUDIO_FOLDER):
            file_path = os.path.join(AUDIO_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"删除文件 {file_path} 时出错: {e}")
        
        # 清空历史记录
        save_history([])
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"清空历史记录失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)