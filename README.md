# 发音评估系统

## 项目简介

本项目是一个基于深度学习的韩语发音评估系统，支持音频上传、发音评分、可视化对齐以及 AI 改进建议。该系统集成了声纹比对、DTW 动态时间规整、TTS 语音合成和 Flask Web 服务。

---

## 文件结构

```
.
├── app.py              # Web 服务主程序（Flask）
├── train.py            # 声纹模型训练脚本
├── inference.py        # 发音评分与对齐可视化
├── inference1.py       # 发音评分 + AI 改进建议（新版API）
├── test.py             # 文本批量转 TTS
├── test1.py            # OpenAI GPT API 测试脚本
├── test2.py            # 音频格式转换（m4a -> wav）
├── TTS1.py             # 使用 XTTS 生成韩语音频
├── dataset.py          # 三元组音频数据集定义
├── model.py            # 声纹编码模型定义
├── wav2mel.py          # 梅尔频谱提取工具
├── static/             # 前端静态文件（HTML、JS、CSS）
├── uploads/            # 上传音频临时存放目录
├── output_audio/       # TTS 音频输出目录
├── audio_history/      # 评测历史音频保存目录
├── cache/              # 临时缓存录音文件
└── history.json        # 评测历史记录文件
```

---

## 安装依赖

```bash
pip install torch librosa gTTS moviepy flask tenacity fastdtw matplotlib scikit-learn TTS openai tqdm soundfile
```

---

## 使用说明

### 启动 Web 服务

```bash
python app.py
```

访问浏览器：[http://127.0.0.1:5001/](http://127.0.0.1:5001/)

### 训练模型

```bash
python train.py
```

输出：

* `voice_encoder_best.pth`（最优模型）
* `Cosine Similarity.png`、`t-SNE.png`（可视化）

### 命令行发音评估

```bash
python inference.py
```

或

```bash
python inference1.py
```

### 文本转语音

```bash
python TTS1.py
python test.py
```

### 音频格式转换

```bash
python test2.py
```

---

## 注意事项

* 推荐音频文件为 16kHz 单声道 WAV 格式。
* 配置 OpenAI API KEY 以获取改进建议。
* 请准备 `data/kss/1/` 参考音频文件和相应测试音频。

---

## 联系方式

如有问题请联系项目负责人。
