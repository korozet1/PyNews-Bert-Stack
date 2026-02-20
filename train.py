import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score
from tqdm import tqdm
import os
from config import Config
from utils import build_dataloader, get_time_diff
from bert_classifer_model import BertClassifier
import time
# 加载配置对象，包含模型参数、路径等
conf = Config()
# 忽略的警告信息
import warnings
warnings.filterwarnings("ignore")


def model2train():
    """
    训练 BERT 分类模型并在验证集上评估，保存最佳模型。

    参数：
        无显式参数，所有配置通过全局 conf 对象获取。

    返回：
        无返回值，训练过程中保存最佳模型到指定路径。
    """
    # 1. 加载训练、测试和验证数据集的 DataLoader
    train_loader, test_loader, dev_loader = build_dataloader()

    # 2. 定义训练参数，从配置对象中获取
    device = conf.device  # 设备（"cuda" 或 "cpu"）
    num_epochs = conf.num_epochs  # 训练轮数
    learning_rate = conf.learning_rate  # 学习率
    model_save_path = conf.model_save_path  # 模型保存路径

    # 3. 初始化 BERT 分类模型
    model = BertClassifier()

    # 4. 将模型移动到指定设备（GPU 或 CPU）
    model.to(device)

    # 5. 定义优化器（AdamW，适合 Transformer 模型）和损失函数（交叉熵）
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 6. 初始化最佳验证 F1 分数，用于保存性能最好的模型
    best_dev_f1 = 0.0

    # 7. 遍历每个训练轮次（epoch）
    for epoch in range(num_epochs):
        # 设置模型为训练模式（启用 dropout 和 batch norm）

        total_loss = 0  # 累计训练损失
        train_preds, train_labels = [], []  # 存储训练集预测和真实标签

        # 8. 遍历训练 DataLoader 进行模型训练
        for i,batch in enumerate(tqdm(train_loader, desc=f"Bert Classifier Training Epoch {epoch + 1}/{num_epochs}....")):
            model.train()
            # 8.1 提取批次数据并移动到设备
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 8.2 清空优化器的梯度
            optimizer.zero_grad()

            # 8.3 前向传播：模型预测
            logits = model(input_ids, attention_mask)

            # 8.4 计算损失
            loss = criterion(logits, labels)

            # 8.5 反向传播：计算梯度
            loss.backward()

            # 8.6 参数更新：根据梯度更新模型参数
            optimizer.step()

            # 8.7 累计损失
            total_loss += loss.item()

            # 8.8 获取预测结果（最大 logits 对应的类别）
            preds = torch.argmax(logits, dim=1)

            # 8.9 存储预测和真实标签，用于计算训练集指标
            train_preds.extend(preds.tolist())
            train_labels.extend(labels.tolist())

            # 8.10 每 10 个批次或非空批次时，打印训练信息并评估验证集
            # if len(batch) % 10 == 0 or len(batch) != 0:
            # 每100步才验证一次
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"Train Loss: {total_loss / len(train_loader):.4f}")
                # 在验证集上评估模型
                report, f1score, accuracy, precision = model2dev(model, dev_loader, device)
                print(f"Dev F1: {f1score:.4f}")
                print(f"Dev Accuracy: {accuracy:.4f}")

                # 8.11 如果验证 F1 分数优于历史最佳，保存模型
                if f1score > best_dev_f1:
                    best_dev_f1 = f1score
                    torch.save(model.state_dict(), conf.model_save_path)
                    print("模型保存！！")

        # 8.12 计算并打印训练集的分类报告
        train_report = classification_report(
            train_labels, train_preds, target_names=conf.class_list, output_dict=True
        )
        print(train_report)


def model2dev(model, data_loader, device):
    """
    在验证或测试集上评估 BERT 分类模型的性能。

    参数：
        model (nn.Module): BERT 分类模型。
        data_loader (DataLoader): 数据加载器（验证或测试集）。
        device (str): 设备（"cuda" 或 "cpu"）。

    返回：
        tuple: (分类报告, F1 分数, 准确度, 精确度)
            - report: 分类报告（包含每个类别的精确度、召回率、F1 分数等）。
            - f1score: 微平均 F1 分数。
            - accuracy: 准确度。
            - precision: 微平均精确度。
    """
    # 1. 设置模型为评估模式（禁用 dropout 和 batch norm）
    model.eval()

    # 2. 初始化列表，存储预测结果和真实标签
    preds, true_labels = [], []

    # 3. 禁用梯度计算以提高效率并减少内存占用
    with torch.no_grad():
        # 4. 遍历数据加载器，逐批次进行预测
        for batch in tqdm(data_loader, desc="Bert Classifier Evaluating ......"):
            # 4.1 提取批次数据并移动到设备
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 4.2 前向传播：模型预测
            logits = model(input_ids, attention_mask)

            # 4.3 获取预测结果（最大 logits 对应的类别）
            batch_preds = torch.argmax(logits, dim=1)

            # 4.4 存储预测和真实标签
            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 5. 计算分类报告、F1 分数、准确度和精确度
    report = classification_report(true_labels, preds)
    f1score = f1_score(true_labels, preds, average='micro')  # 使用微平均计算 F1 分数
    accuracy = accuracy_score(true_labels, preds)  # 计算准确度
    precision = precision_score(true_labels, preds, average='micro')  # 使用微平均计算精确度

    # 6. 返回评估结果
    return report, f1score, accuracy, precision
if __name__ == '__main__':
    # 【开启训练模式】
    # 这行代码会调用 model2train 函数，从头开始训练你的 BERT
    model2train()

# if __name__ == '__main__':
#     # 主程序入口
#     # model2train()
#     # 1. 加载测试集数据
#     train_dataloader, test_dataloader, dev_dataloader = build_dataloader()
#     # 2. 初始化 BERT 分类模型
#     model = BertClassifier()
#     # 3. 加载预训练模型权重
#     model.load_state_dict(torch.load(r"C:\Lucky_dt\2_bj\22AI_BJStudents\TMFCode\04-bert\save_models\bert20250521_.pt"))
#     # 4. 将模型移动到指定设备
#     model.to(conf.device)
#     # 5. 在测试集上评估模型
#     test_report, f1score, accuracy, precision = model2dev(model, test_dataloader, conf.device)
#     # 6. 打印测试集评估结果
#     print("Test Set Evaluation:")
#     print(f"Test F1: {f1score:.4f}")
#     print("Test Classification Report:")
#     print(test_report)