import torch
import os
import datetime
from transformers.models import BertModel,BertTokenizer,BertConfig
current_date=datetime.datetime.now().date().strftime("%Y%m%d")

class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        self.model_name = "bert" # 模型名称
        self.data_path = r"data"  #数据集的根路径
        self.train_path = self.data_path + "\\train.txt"  # 训练集
        self.dev_path = self.data_path + "\\dev.txt"  # 少量验证集，快速验证
        self.test_path = self.data_path + "\\test.txt"  # 测试集

        self.class_path=self.data_path + "\\class.txt" #类别文件

        self.class_list = [line.strip() for line in open(self.class_path, encoding="utf-8")]  # 类别名单

        self.model_save_path = r"save_models\test_bertclassifer_model.pt"  #模型训练结果保存路径

        # 模型训练+预测的时候
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练设备，如果GPU可用，则为cuda，否则为cpu

        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 2  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = r"bert-base-chinese"  # 预训练BERT模型的路径
        self.bert_model=BertModel.from_pretrained(self.bert_path)# pytorch_model.bin
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) # BERT模型的分词器 vocab.txt
        self.bert_config = BertConfig.from_pretrained(self.bert_path) # BERT模型的配置config.json
        self.hidden_size = 768 # BERT模型的隐藏层大小
        self.quantized_model_save_path = r"save_models\quantized_model_bertclassifer_model.pt"

if __name__ == '__main__':
    conf = Config()
    print(conf.bert_config)
    input_size=conf.tokenizer.convert_tokens_to_ids(["你","好","中国","人"])
    print(input_size)
    print(conf.class_list)
