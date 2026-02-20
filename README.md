# BERT-News-Classifier | 中文新闻智能分类系统

<div align="center">
  <h1>🌌 BERT-News-Classifier</h1>
  <p>
    <b>High-Performance Chinese News Classification System based on BERT & PyTorch</b>
  </p>
  
  <img src="[https://img.shields.io/badge/Framework-PyTorch-orange?style=flat-square&logo=pytorch](https://img.shields.io/badge/Framework-PyTorch-orange?style=flat-square&logo=pytorch)" alt="PyTorch">
  <img src="[https://img.shields.io/badge/Model-BERT_Base-yellow?style=flat-square&logo=huggingface](https://img.shields.io/badge/Model-BERT_Base-yellow?style=flat-square&logo=huggingface)" alt="BERT">
  <img src="[https://img.shields.io/badge/Microservice-Flask-green?style=flat-square&logo=flask](https://img.shields.io/badge/Microservice-Flask-green?style=flat-square&logo=flask)" alt="Flask">
  <img src="[https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)" alt="Streamlit">
  <img src="[https://img.shields.io/badge/License-MIT-blue?style=flat-square](https://img.shields.io/badge/License-MIT-blue?style=flat-square)" alt="License">
</div>

---

## 📖 项目背景 (Background)

在自然语言处理（NLP）领域，文本分类是核心的基础任务之一。本项目基于 **Google BERT (Bidirectional Encoder Representations from Transformers)** 预训练模型，构建了一个**高精度、生产级**的中文新闻文本分类系统。

通过在海量中文语料上预训练的 `bert-base-chinese` 模型进行微调（Fine-tuning），本项目能够精准捕捉文本的双向上下文特征，在金融、体育、科技等新闻类别上实现了 **SOTA** 级别的分类效果。项目集成了 **Flask 微服务后端** 与 **Streamlit 可视化前端**，提供了一站式的“训练-推理-展示”解决方案。

---

## 📂 项目目录结构 (Project Structure)

基于 `test-04` 实际环境：

```text
test-04/
├── bert-base-chinese/          # [核心] 本地预训练模型目录
│   ├── config.json             # 模型配置文件
│   ├── pytorch_model.bin       # 模型权重文件 (需手动下载)
│   ├── vocab.txt               # 词表文件
│   ├── tokenizer.json          # 分词器配置
│   └── tokenizer_config.json
├── data/                       # 数据集存放目录
│   ├── class.txt               # 类别标签定义
│   ├── dev.txt                 # 验证集
│   ├── test.txt                # 测试集
│   ├── train.txt               # 训练集
│   └── stopwords.txt           # 停用词表
├── save_models/                # 训练产出目录
│   └── test_bertclassifer_model.pt  # 训练好的最佳模型权重
├── bert_classifer_model.py     # 模型架构定义 (BERT + FC Layer)
├── config.py                   # 全局配置文件 (路径、超参数)
├── utils.py                    # 数据加载与处理工具 (Dataset/DataLoader)
├── train.py                    # 模型训练主脚本
├── predict_fun.py              # 单次推理函数封装
├── api.py                      # Flask 后端接口服务
├── api_test.py                 # 接口测试脚本
├── app.py                      # Streamlit 前端可视化页面
└── README.md                   # 项目说明文档
```

---

## ⚡ 快速启动指南 (Quick Start)

### ⚠️ 第一步：下载模型文件 (Crucial Step)
由于 Git 限制大文件，你需要手动下载预训练模型并放入 `bert-base-chinese` 文件夹。

1.  **下载地址**: [Hugging Face - bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese/tree/main)
2.  **确保目录下包含以下核心文件**:
    * `config.json`
    * `pytorch_model.bin` (约 400MB)
    * `vocab.txt`

### 第二步：环境配置 (Environment)

```bash
pip install torch transformers flask streamlit scikit-learn tqdm requests
```

### 第三步：训练模型 (Training)
运行训练脚本，模型将开始微调。训练完成后，最佳模型会自动保存为 `save_models/test_bertclassifer_model.pt`。

```bash
python train.py
```

### 第四步：启动服务 (Deployment)

**方式 A：启动 Flask 后端 API**
服务将运行在 `http://0.0.0.0:8004`，提供高性能预测接口。

```bash
python api.py
```

**方式 B：启动可视化界面 (Web UI)**
请先启动 `api.py`，然后在一个新的终端窗口运行：

```bash
streamlit run app.py
```
浏览器会自动打开，你可以在网页上直接输入新闻标题进行测试。

---

## 📡 API 接口规范 (Interface Specification)

**服务地址**: `http://127.0.0.1:8004`

### 新闻分类预测接口

* **URL**: `/predict`
* **Method**: `POST`
* **Content-Type**: `application/json`

**请求参数:**

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `text` | string | 是 | 需要分类的新闻文本内容 |

**请求示例 (Python):**

```python
import requests
data = {"text": "SpaceX 星舰今日成功发射，开启火星移民新篇章"}
response = requests.post("http://127.0.0.1:8004/predict", json=data)
print(response.json())
```

**响应示例:**

```json
{
    "text": "SpaceX 星舰今日成功发射...",
    "pred_class": "科技"
}
```

---

## ⚙️ 核心配置 (Configuration)

所有参数均在 `config.py` 中管理，可根据机器性能进行调整：

* `self.device`: 自动检测 `cuda` 或 `cpu`。
* `self.batch_size`: 默认 `128` (显存较小时建议调至 32 或 64)。
* `self.learning_rate`: 默认 `5e-5` (微调标准学习率)。
* `self.pad_size`: 默认 `32` (根据新闻标题长度设定的截断值)。

---

## ❤️ 致谢 (Acknowledgments)

* **Hugging Face**: 提供强大的 Transformers 库。
* **PyTorch**: 深度学习框架支持。

---

## 📄 版权说明 (License)

本项目采用 **MIT License** 开源协议。
> 2026 © Developed by BERT-Team
