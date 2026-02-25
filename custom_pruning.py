"""
BERT 全局非结构化剪枝：对所有 encoder 层注意力权重剪枝 30%，L1 范数。
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import BertModel
from bert_classifer_model import BertClassifier
from utils import build_dataloader
from train import model2dev
from tqdm import tqdm
from itertools import islice
from config import Config
conf = Config()  # 加载配置文件
def compute_sparsity(model):
    """计算所有 encoder 层 query 权重的稀疏度"""
    total_params = 0
    zero_params = 0
    for i in range(12):
        weight = model.bert.encoder.layer[i].attention.self.query.weight
        total_params += weight.numel()
        zero_params += (weight == 0).sum().item()
    return zero_params / total_params if total_params > 0 else 0


def print_weights(weight, name, rows=5, cols=5):
    """打印权重矩阵的前 rows x cols 部分"""
    print(f"\n{name}（前 {rows}x{cols}）：")
    print(weight[:rows, :cols])

def main():
    device = conf.device
    train_dataloader, test_dataloader, dev_dataloader = build_dataloader()

    # 加载模型
    model = BertClassifier()
    # state_dict = torch.load("XXXXX/04-bert/save_models/bert20250521_.pt")
    # model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(torch.load(conf.model_save_path),strict=False)
    model.to(device)

    # 剪枝前
    print("剪枝前模型：")
    print(model.bert.encoder.layer[0].attention.self)
    print_weights(model.bert.encoder.layer[0].attention.self.query.weight, "layer[0].attention.self.query.weight 剪枝前")
    report, f1score, accuracy, precision = model2dev(model, dev_dataloader, device)
    print(f"\n剪枝前准确率: {accuracy:.4f}, F1: {f1score:.4f}")

    # 全局非结构化剪枝：所有 encoder 层 query 权重 30%
    parameters_to_prune = [
        (model.bert.encoder.layer[i].attention.self.query, 'weight') for i in range(12)
    ]
    # 只对 fc 层的权重进行剪枝
    # parameters_to_prune = [
    #     (model.fc, 'weight')  # 只剪 fc 层的 weight
    #     # 如果想同时剪 bias，可以再加一条 (model.fc, 'bias')
    # ]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.3,
    )
    for module, param in parameters_to_prune:
        prune.remove(module, param)

    # 剪枝后
    print("\n剪枝后模型：")
    print(model.bert.encoder.layer[0].attention.self)
    print_weights(model.bert.encoder.layer[0].attention.self.query.weight, "layer[0].attention.self.query.weight 剪枝后")
    report, f1score, accuracy, precision = model2dev(model, dev_dataloader, device)
    sparsity = compute_sparsity(model)
    print(f"\n剪枝后准确率: {accuracy:.4f}, F1: {f1score:.4f}\n稀疏度: {sparsity:.4f}")
    save_path = conf.save_model_path3 + "/pruned_model.pth"
    torch.save(model.state_dict(), save_path)
    print("剪枝后模型已保存为 pruned_model.pth")
if __name__ == '__main__':
    main()