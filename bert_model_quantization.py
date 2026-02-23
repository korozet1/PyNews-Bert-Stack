# bert_model_quantization.py
from bert_classifer_model import BertClassifier
from config import Config
import numpy as np
import torch
from utils import build_dataloader
from train import model2dev

# 初始化配置
conf = Config()

if __name__ == '__main__':
    # 1、创建数据迭代器
    print('加载数据...')
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()

    # 2、加载模型
    print("加载模型...")
    device = conf.device
    model = BertClassifier()
    model_path = conf.model_save_path
    model.load_state_dict(torch.load(model_path, map_location='cpu')) #map_location指定映射到哪个设备上，cpu或gpu
    model.eval()

    print("查看量化前的模型结构=========================")
    print(model)

    # 3、torch.quantization.quantize_dynamic量化BERT模型 dtype=torch.qint8
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    # 检查量化模型中各层的参数数据类型
    print("量化后的模型=========================")
    print(quantized_model)

    #4.model2dev 测试量化后的模型 (quantized_model, test_dataloader, device)
    # report, f1score, accuracy, precision = model2dev(quantized_model, test_dataloader, device)
    # print("Test Classification Report:", report)
    # print("Test F1:", f1score)
    # print("Test Accuracy:", accuracy)
    # print("Test Precision:", precision)
    #5、计算 计算8-bit量化后模型的内存占用（单位：MB）
    # sum(p.numel() * p.element_size() for p in quantized_model.parameters()): 遍历模型参数，计算每个参数张量的元素总数（numel）乘以每个元素字节大小（element_size），累加得到总字节数
    # / 1024 ** 2: 将字节数转换为兆字节（MB）
    # :.2f: 保留两位小数
    print(f"8-bit 量化后的模型内存: {sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024 ** 2:.2f} MB")


    #6、 保存整个量化模型
    torch.save(quantized_model.state_dict(), conf.quantized_model_save_path)
    print("保存量化模型成功！地址为：", conf.quantized_model_save_path)

    """
    注意：模型保存时候动态量化是 int8！！所以模型会变�
    """