import torch
from torch.utils.data import DataLoader, random_split
from dataset import TripletAudioDataset
from model import VoiceEncoder
from torch import nn, optim
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# 参数
standard_dir = 'data/kss/1'
non_standard_dir = 'tts_outputs/1'
batch_size = 16
lr = 1e-3
epochs = 2
val_ratio = 0.2
max_frames = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集
full_dataset = TripletAudioDataset(standard_dir, non_standard_dir, max_frames=max_frames)
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 模型与优化器
model = VoiceEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
triplet_loss = nn.TripletMarginLoss(margin=1.0)

best_acc = 0.0

# 记录嵌入
all_embeddings = []
all_labels = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    
    for a, p, n in train_bar:
        a, p, n = a.to(device), p.to(device), n.to(device)
        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)

        loss = triplet_loss(emb_a, emb_p, emb_n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # ========================= 验证 =========================
    model.eval()
    val_loss = 0.0
    cos_ap_total, cos_an_total = 0.0, 0.0
    correct, total = 0, 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
    with torch.no_grad():
        for a, p, n in val_bar:
            a, p, n = a.to(device), p.to(device), n.to(device)
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = triplet_loss(emb_a, emb_p, emb_n)
            val_loss += loss.item()

            cos_ap = F.cosine_similarity(emb_a, emb_p)
            cos_an = F.cosine_similarity(emb_a, emb_n)

            cos_ap_total += cos_ap.sum().item()
            cos_an_total += cos_an.sum().item()
            correct += (cos_ap > cos_an).sum().item()
            total += a.size(0)

            # 记录嵌入和标签
            all_embeddings.append(emb_a.cpu().numpy())
            all_labels.append(torch.ones(a.size(0)).cpu().numpy())  # 标准样本标签

            all_embeddings.append(emb_p.cpu().numpy())
            all_labels.append(torch.ones(a.size(0)).cpu().numpy())  # 正样本标签

            all_embeddings.append(emb_n.cpu().numpy())
            all_labels.append(torch.zeros(a.size(0)).cpu().numpy())  # 负样本标签

    avg_val_loss = val_loss / len(val_loader)
    avg_cos_ap = cos_ap_total / total
    avg_cos_an = cos_an_total / total
    acc = correct / total

    print(f"\n📊 Epoch {epoch+1}:")
    print(f"   Train Loss: {avg_train_loss:.4f}")
    print(f"   Val Loss  : {avg_val_loss:.4f}")
    print(f"   Cos(A,P)  : {avg_cos_ap:.4f} | Cos(A,N): {avg_cos_an:.4f}")
    print(f"   Accuracy  : {acc*100:.2f}%\n")

    # 保存最好的模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "voice_encoder_best.pth")
        print(f"✅ Saved best model with accuracy: {best_acc*100:.2f}%")

# 最终保存
torch.save(model.state_dict(), "voice_encoder_final.pth")
print("🎉 Training complete.")

# 画余弦相似度分布图
cosine_similarities = []
for a, p, n in zip(all_embeddings[::3], all_embeddings[1::3], all_embeddings[2::3]):
    cos_sim = F.cosine_similarity(torch.tensor(a), torch.tensor(p))
    cosine_similarities.extend(cos_sim.cpu().numpy().tolist())

plt.hist(cosine_similarities, bins=50, alpha=0.75, color='blue')
plt.title("Cosine Similarity Distribution between Anchor and Positive")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
# plt.show()
plt.savefig("Cosine Similarity.png")
# t-SNE 可视化嵌入空间
all_embeddings = np.concatenate(all_embeddings)
all_labels = np.concatenate(all_labels)

# 使用 t-SNE 降维至2维
tsne = TSNE(n_components=2, random_state=42)
embedded_2d = tsne.fit_transform(all_embeddings)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(embedded_2d[all_labels == 1][:, 0], embedded_2d[all_labels == 1][:, 1], c='b', label='Standard')
plt.scatter(embedded_2d[all_labels == 0][:, 0], embedded_2d[all_labels == 0][:, 1], c='r', label='Non-Standard')
plt.legend()
plt.title("t-SNE Visualization of Audio Embeddings")
# plt.show()
plt.savefig("t-SNE.png")
