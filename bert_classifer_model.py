import torch
import torch.nn as nn
from transformers import BertModel
from config import Config
from utils import build_dataloader
conf = Config()
class BertClassifier(nn.Module):
    """
    BERT + 全连接层的分类模型。
    """
    def __init__(self):
        """
        初始化模型，包括BERT和全连接层。
        """
        super(BertClassifier, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(conf.bert_path)
        # 全连接层：将BERT的隐藏状态映射到类别数
        self.fc = nn.Linear(conf.hidden_size, conf.num_classes)

    def forward(self, input_ids, attention_mask):
        # x: 模型输入，包含句子、句子长度和填充掩码。
        # _是占位符，接收模型的所有输出，而 pooled 是池化的结果,将整个句子的信息压缩成一个固定长度的向量
        _, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        # print(pooled.shape) #batch_size,hidden_size
        # 模型输出，用于文本分类
        out = self.fc(pooled)
        return out


if __name__ == '__main__':
    model = BertClassifier()
    train_dataloader,test_dataloader,dev_dataloader=build_dataloader()
    for  batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        logits = model(input_ids, attention_mask)
        print(logits.shape)
        print(torch.argmax(logits, dim=1))
        print(labels)

