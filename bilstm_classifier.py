import torch
import torch.nn as nn
from config import Config
from utils import build_dataloader

conf = Config()

class BiLSTMClassifier(nn.Module):
    """
    BiLSTM + 全连接层的分类模型，作为学生模型。
    """
    def __init__(self, embed_size=128, hidden_size=256, num_layers=2, num_classes=conf.num_classes, dropout=0.3):
        """
        初始化BiLSTM模型。
        参数：
            embed_size: 嵌入维度。
            hidden_size: LSTM隐藏状态维度。
            num_layers: LSTM层数。
            num_classes: 分类类别数。
            dropout: Dropout比例。
        """
        super(BiLSTMClassifier, self).__init__()
        vocab_size = conf.tokenizer.vocab_size  # 从BERT分词器动态获取词汇表大小
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.hidden_projection = nn.Linear(hidden_size * 2, conf.hidden_size)  # 映射到BERT隐藏状态维度
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, return_hidden=False):
        """
        前向传播，仅在嵌入层使用 attention_mask 进行掩码处理。
        参数：
            input_ids: 输入的token ID，形状为 [batch_size, seq_len]。
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]，1 表示有效 token，0 表示填充 token。
            return_hidden: 是否返回隐藏状态。
        返回：
            logits: 分类logits，形状为 [batch_size, num_classes]。
            hidden: 最后一时间步的隐藏状态（若 return_hidden=True），形状为 [batch_size, hidden_size*2]。
        """
        # 嵌入层
        embed = self.embedding(input_ids)  # [batch_size, seq_len, embed_size]

        # 使用 attention_mask 掩码填充 token 的嵌入（核心处理）
        attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        embed = embed * attention_mask  # 将填充 token 的嵌入置为 0

        # LSTM 层
        lstm_out, (hidden, _) = self.lstm(embed)  # lstm_out: [batch_size, seq_len, hidden_size*2]

        # 取最后一时间步的隐藏状态（填充 token 已置 0，无需再次处理）
        hidden = lstm_out[:, -1, :]  # [batch_size, hidden_size*2]

        # Dropout 和全连接层
        hidden = self.dropout(hidden)  # [batch_size, hidden_size*2]
        logits = self.fc(hidden)  # [batch_size, num_classes]

        if return_hidden:
            projected_hidden = self.hidden_projection(hidden)  # 映射到768维
            return logits, projected_hidden
        return logits
