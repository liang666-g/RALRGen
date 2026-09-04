import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# ================= 1. 超参数与配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10           # 训练轮数
BATCH_SIZE = 64
EMB_DIM = 256         # 词向量维度
HID_DIM = 512         # RNN 隐藏层维度
DROPOUT = 0.2
LEARNING_RATE = 0.001
MAX_LEN = 50          # 句子最大长度

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# ================= 2. 数据处理与词表构建 =================
def read_data(file_path):
    reviews, replies = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            terms = line.strip().split("-[split]-")
            if len(terms) >= 6:
                reviews.append(terms[4].strip().lower().split())
                replies.append(terms[5].strip().lower().split())
    return reviews, replies

class Vocab:
    def __init__(self, min_freq=2):
        self.itos = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.stoi = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.min_freq = min_freq

    def build_vocab(self, sentence_list):
        counter = Counter([word for sentence in sentence_list for word in sentence])
        idx = 4
        for word, count in counter.items():
            if count >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in text]

class NMTDataset(Dataset):
    def __init__(self, src_data, trg_data, src_vocab, trg_vocab):
        self.src_data = src_data
        self.trg_data = trg_data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = [self.src_vocab.stoi[SOS_TOKEN]] + self.src_vocab.numericalize(self.src_data[idx])[:MAX_LEN-2] + [self.src_vocab.stoi[EOS_TOKEN]]
        trg = [self.trg_vocab.stoi[SOS_TOKEN]] + self.trg_vocab.numericalize(self.trg_data[idx])[:MAX_LEN-2] + [self.trg_vocab.stoi[EOS_TOKEN]]
        return torch.tensor(src), torch.tensor(trg)

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_item, trg_item in batch:
        src_batch.append(src_item)
        trg_batch.append(trg_item)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=0)
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, padding_value=0)
    return src_batch, trg_batch

# ================= 3. NMT 模型定义 (RNN + Attention) =================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = Attention(hid_dim)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(hid_dim * 2 + emb_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# ================= 4. 推理生成函数 =================
def translate_sentence(sentence_tokens, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()
    tokens = [SOS_TOKEN] + sentence_tokens + [EOS_TOKEN]
    src_indexes = [src_vocab.stoi.get(token, src_vocab.stoi[UNK_TOKEN]) for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        
    trg_indexes = [trg_vocab.stoi[SOS_TOKEN]]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab.stoi[EOS_TOKEN]:
            break
            
    trg_tokens = [trg_vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:-1] # 移除 SOS 和 EOS

# ================= 5. 主流程 =================
def run_nmt_experiment():
    print("🚀 [1/5] 加载数据与构建词表...")
    train_reviews, train_replies = read_data("./data/train.txt")
    test_reviews, test_replies = read_data("./data/test.txt")
    
    # 获取原始测试集的评论文本，用于写入文件
    raw_test_reviews = []
    with open("./data/test.txt", "r", encoding="utf-8") as f:
        for line in f:
            terms = line.strip().split("-[split]-")
            if len(terms) >= 6:
                raw_test_reviews.append(terms[4].strip())

    vocab = Vocab(min_freq=2)
    vocab.build_vocab(train_reviews + train_replies) # 共享词表

    train_dataset = NMTDataset(train_reviews, train_replies, vocab, vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    print(f"✅ 词表大小: {len(vocab.stoi)}")
    
    print("🚀 [2/5] 初始化 NMT 模型...")
    INPUT_DIM = len(vocab.stoi)
    OUTPUT_DIM = len(vocab.stoi)
    
    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 忽略 PAD
    
    print(f"🤖 模型将运行在: {DEVICE}")
    print("🚀 [3/5] 开始训练 NMT 模型 (请耐心等待)...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for i, (src, trg) in enumerate(train_loader):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"   Epoch: {epoch+1:02} | Train Loss: {epoch_loss/len(train_loader):.3f}")

    print("🚀 [4/5] 训练完成！开始在测试集上生成回复...")
    out_dir = "./contrast_model_response/NMT"
    os.makedirs(out_dir, exist_ok=True)
    out_pred_path = os.path.join(out_dir, "pred.txt")
    out_tar_path = os.path.join(out_dir, "tar.txt")

    with open(out_pred_path, "w", encoding="utf-8") as f_pred, \
         open(out_tar_path, "w", encoding="utf-8") as f_tar:
         
        for i in range(len(test_reviews)):
            raw_rev = raw_test_reviews[i]
            target_tokens = test_replies[i]
            pred_tokens = translate_sentence(test_reviews[i], vocab, vocab, model, DEVICE)
            
            pred_str = " ".join(pred_tokens)
            tar_str = " ".join(target_tokens)
            
            f_pred.write(f"{raw_rev}-[split]-{pred_str}\n")
            f_tar.write(f"{raw_rev}-[split]-{tar_str}\n")
            
            if (i+1) % 500 == 0:
                print(f"   已生成 {i+1}/{len(test_reviews)} 条测试数据")

    print("\n==================================================")
    print("✅ [5/5] NMT 基线实验数据生成完毕！")
    print(f"🎯 预测结果已保存至: {out_pred_path}")
    print(f"🎯 真实结果已保存至: {out_tar_path}")
    print("\n🚀 接下来，你可以直接用这行命令去计算 NMT 基线的各项指标了：")
    print(f"python model.py --task evaluate --pred-file {out_pred_path} --no-test-simi")
    print("==================================================")

if __name__ == "__main__":
    run_nmt_experiment()