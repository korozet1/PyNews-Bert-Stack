import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score
from tqdm import tqdm
from config import Config
from utils import build_dataloader, get_time_diff
from bert_classifer_model import BertClassifier
from bilstm_classifier import BiLSTMClassifier
import time
import warnings
from train import model2dev

warnings.filterwarnings("ignore")  # 忽略警告信息

conf = Config()  # 加载配置文件
def model2train(teacher_model, student_model, train_loader, dev_loader, num_epochs, learning_rate, device, save_path):
    """
    训练学生模型（BiLSTM）使用硬标签蒸馏，学习教师模型（BERT）的预测类别。

    参数：
        teacher_model: 教师模型（BERT），提供硬标签。
        student_model: 学生模型（BiLSTM），需要学习教师模型的预测。
        train_loader: 训练数据加载器，提供训练数据批次。
        dev_loader: 验证数据加载器，提供验证数据批次。
        num_epochs: 训练的总轮数（epoch 数量）。
        learning_rate: 学习率，控制优化器更新步长。
        device: 训练设备（"cuda" 或 "cpu"）。
        save_path: 模型保存路径，保存最佳模型权重。
    """
    # 将模型移动到指定设备（GPU 或 CPU）
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # 初始化优化器和损失函数
    optimizer = AdamW(student_model.parameters(), lr=learning_rate)  # 使用 AdamW 优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失，用于硬标签损失
    best_dev_f1 = 0.0  # 记录最佳验证 F1 分数
    step = 0  # 训练步数计数器
    patience = 3  # 早停耐心值
    epochs_no_improve = 0  # 记录未提升的 epoch 数

    # 打印训练参数
    print("训练参数 Training Parameters:")
    print(f"num_epochs: {num_epochs}, learning_rate: {learning_rate}, device: {device}, batch_size: {train_loader.batch_size}")

    # 遍历每个 epoch
    for epoch in range(num_epochs):
        student_model.train()  # 设置学生模型为训练模式
        teacher_model.eval()  # 设置教师模型为评估模式（不更新权重）
        total_loss = 0  # 记录当前 epoch 的总损失
        train_preds, train_labels = [], []  # 记录训练预测和真实标签
        epoch_start_time = time.time()  # 记录 epoch 开始时间

        print(f"\n硬标签蒸馏训练 Hard Label Distillation Epoch {epoch + 1}/{num_epochs}...")
        # 遍历训练数据批次
        for batch in tqdm(train_loader, desc=f"Hard Label Distillation Epoch {epoch + 1}/{num_epochs}"):
            step_start_time = time.time()  # 记录当前 step 开始时间
            input_ids, attention_mask, labels = batch  # 获取输入数据
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()  # 清空优化器梯度
            # 获取教师模型的预测（硬标签）
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask)
                teacher_preds = torch.argmax(teacher_logits, dim=1)
            # 获取学生模型的输出 logits
            student_logits = student_model(input_ids, attention_mask)

            # 计算硬标签损失（交叉熵，使用教师模型的预测）
            loss = criterion(student_logits, teacher_preds)

            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()  # 累加损失

            # 记录预测结果
            preds = torch.argmax(student_logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            step += 1  # 步数加 1
            step_duration = time.time() - step_start_time  # 计算 step 耗时

            # 每 10 个 step 验证一次
            if step % 1000 == 0:
                student_model.eval()  # 切换到评估模式
                avg_loss = total_loss / (len(train_preds) / train_loader.batch_size)  # 计算平均损失
                report, f1score, accuracy, precision = model2dev(student_model, dev_loader, device)  # 验证
                print(f"Step {step}, Epoch {epoch + 1}/{num_epochs}")
                print(f"Step Duration: {step_duration:.2f} seconds")
                print(f"Train Loss: {avg_loss:.4f}")
                print(f"Dev F1: {f1score:.4f}, Dev Accuracy: {accuracy:.4f}")
                print(f"Dev Precision: {precision:.4f}")
                print("Dev Classification Report:")
                print(report)
                student_model.train()  # 切换回训练模式

        # 计算训练集指标
        train_report = classification_report(train_labels, train_preds, target_names=conf.class_list, output_dict=True)
        train_f1 = train_report["weighted avg"]["f1-score"]

        # 验证（每个 epoch 结束时）
        student_model.eval()
        report, f1score, accuracy, precision = model2dev(student_model, dev_loader, device)

        # 计算 epoch 耗时
        epoch_duration = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Epoch Duration: {epoch_duration:.2f} seconds")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}, Train F1: {train_f1:.4f}")
        print(f"Dev F1: {f1score:.4f}, Dev Accuracy: {accuracy:.4f}")
        print(f"Dev Precision: {precision:.4f}")
        print("Dev Classification Report:")
        print(report)

        # 保存最佳模型并检查早停
        if f1score > best_dev_f1:
            best_dev_f1 = f1score
            torch.save(student_model.state_dict(), save_path)
            print("模型保存！！")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Dev F1 未提升，当前未提升 epoch 数: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"早停触发！Dev F1 在 {patience} 个 epoch 内未提升，停止训练。")
                break

        student_model.train()
if __name__ == '__main__':
    conf = Config()

    # 1. 准备数据加载器
    train_loader, test_loader, dev_loader = build_dataloader()  # 假设utils中定义了该函数

    # 2. 初始化教师模型（BERT）并加载预训练权重（假设已有训练好的教师模型）
    teacher_model = BertClassifier()
    # 如果有训练好的教师模型权重，加载它
    teacher_model.load_state_dict(torch.load(conf.model_save_path))
    teacher_model.to(conf.device)
    teacher_model.eval()  # 教师模型固定，不参与训练

    # 3. 初始化学生模型（BiLSTM）
    student_model = BiLSTMClassifier()

    # 4. 设置训练参数
    num_epochs = conf.num_epochs  # 从config读取，例如10
    learning_rate = conf.learning_rate  # 5e-5
    device = conf.device
    save_path = conf.save_model_path3 + "/bilstm_hard_distill.pth"  # 自行定义保存路径

    # 5. 开始训练
    model2train(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_path=save_path
    )