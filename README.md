# BERT-News-Classifier | 中文新闻智能分类系统

<div align="center">
  <h1>🌌 BERT-News-Classifier</h1>
  <p>
    <b>High-Performance Chinese News Classification System based on BERT & PyTorch</b>
  </p>
  
  <p>
    <a href="[https://pytorch.org/](https://pytorch.org/)">
      <img src="[https://img.shields.io/badge/Framework-PyTorch-orange.svg](https://img.shields.io/badge/Framework-PyTorch-orange.svg)" alt="PyTorch">
    </a>
    <a href="[https://huggingface.co/](https://huggingface.co/)">
      <img src="[https://img.shields.io/badge/Model-BERT--Base-yellow.svg](https://img.shields.io/badge/Model-BERT--Base-yellow.svg)" alt="BERT">
    </a>
    <a href="[https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)">
      <img src="[https://img.shields.io/badge/Microservice-Flask-green.svg](https://img.shields.io/badge/Microservice-Flask-green.svg)" alt="Flask">
    </a>
    <a href="[https://streamlit.io/](https://streamlit.io/)">
      <img src="[https://img.shields.io/badge/UI-Streamlit-red.svg](https://img.shields.io/badge/UI-Streamlit-red.svg)" alt="Streamlit">
    </a>
    <a href="LICENSE">
      <img src="[https://img.shields.io/badge/License-MIT-lightgrey.svg](https://img.shields.io/badge/License-MIT-lightgrey.svg)" alt="License">
    </a>
  </p>
</div>

---

## 📖 项目背景 (Background)

在自然语言处理（NLP）领域，文本分类是核心的基础任务之一。虽然 FastText 等传统模型速度极快，但在处理复杂语义、长文本理解以及语境依赖性较强的中文新闻时，往往难以达到极致的准确率。

本项目基于 **Google BERT (Bidirectional Encoder Representations from Transformers)** 预训练模型，构建了一个**高精度、生产级**的中文新闻文本分类系统。通过在海量中文语料上预训练的 `bert-base-chinese` 模型进行微调（Fine-tuning），本项目能够精准捕捉文本的双向上下文特征，在 10 个新闻类别（金融、体育、科技等）上实现了 **SOTA (State-of-the-Art)** 级别的分类效果（准确率 > 94%）。

此外，项目集成了 **Flask 微服务后端** 与 **Streamlit 可视化前端**，提供了一站式的“训练-推理-展示”解决方案。

---

## 🏗️ 系统架构与核心特性 (Architecture)

### 1. 核心技术栈
* **Core Engine**: `Hugging Face Transformers` (加载 BERT 预训练权重)
* **Deep Learning Framework**: `PyTorch` (动态图构建与自动微分)
* **Model Architecture**: `BERT-Base` + `Fully Connected Layer` (Linear Classifier)
* **Microservice**: `Flask` (RESTful API 模型服务化)
* **Visualization**: `Streamlit` (交互式 Web 演示界面)

### 2. 工作流 (Workflow)
系统设计遵循现代 AI 工程化标准：
* **⚡ 数据加载 (ETL)**: 自定义 `Dataset` 与 `DataLoader`，实现动态 Padding 与 Mask 生成。
* **🧠 模型微调 (Fine-tuning)**: 冻结/解冻 BERT 参数，利用 `AdamW` 优化器进行下游任务适配。
* **🚀 推理服务 (Inference)**: 模型热加载机制，提供毫秒级 HTTP 接口响应。
* **🎨 用户界面 (UI)**: 极简 Web 界面，支持实时文本输入与分类结果展示。

---

## 📂 项目模块深度解析 (Module Manifest)

本项目采用模块化设计，职责分离，易于维护与扩展。

| 文件名 (Filename) | 模块类型 | 深度功能解析 (Description) |
| :--- | :--- | :--- |
| **`bert_classifer_model.py`** | 🧠 **模型定义** | **神经网络架构核心**。<br>定义了 `BertClassifier` 类，继承自 `nn.Module`。它加载预训练的 BERT Backbone，并附加一个 768维 -> 10维 的全连接层（Classification Head）用于最终分类。 |
| **`train.py`** | 🧪 **训练引擎** | **模型微调脚本**。<br>包含完整的训练循环（Training Loop）与验证逻辑。实现了 `AdamW` 优化器配置、Loss 计算、反向传播、梯度更新以及最佳模型自动保存（Checkpointing）。 |
| **`config.py`** | 🔧 **配置中心** | **全局超参数管理**。<br>统一管理所有路径（Dataset Path, Model Path）、超参数（Batch Size, Learning Rate, Epochs）及设备配置（CPU/GPU）。实现“代码与配置分离”。 |
| **`utils.py`** | 🛠️ **工具库** | **数据处理基础设施**。<br>包含 `TextDataset` 类与核心的 `collate_fn` 函数。负责将原始文本转换为 BERT 所需的 `input_ids`, `attention_mask` 和 `token_type_ids`。 |
| **`api.py`** | 🚀 **服务网关** | **后端 API 主程序**。<br>基于 Flask 构建。在服务启动时预加载 BERT 模型至内存/显存，提供 `/predict` 接口，支持高并发文本预测请求。监听端口: `8004`。 |
| **`app.py`** | 🎨 **前端界面** | **可视化演示系统**。<br>基于 Streamlit 构建。作为客户端调用 `api.py` 的接口，提供友好的 Web 交互界面，展示预测结果与耗时。 |
| **`predict_fun.py`** | ⚙️ **推理封装** | **单次预测逻辑**。<br>封装了分词、Tensor 转换、模型推理及 Label 解码的全过程，供 API 和测试脚本调用。 |

---

## ⚡ 快速启动指南 (Quick Start)

### ⚠️ 关键步骤：下载预训练模型 (Crucial Step)
由于 BERT 模型体积较大（约 400MB），你需要手动下载并将以下文件放入项目根目录下的 `bert-base-chinese/` 文件夹中。

1.  **下载地址**: [Hugging Face - bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese/tree/main)
2.  **必需文件**:
    * `config.json`
    * `pytorch_model.bin`
    * `vocab.txt`

### Phase 1: 环境准备 (Prerequisites)
推荐使用 Python 3.8+ 及 CUDA 环境（如果有 NVIDIA 显卡）。

~~~bash
pip install torch transformers flask streamlit scikit-learn tqdm requests
~~~

### Phase 2: 模型微调 (Fine-tuning)
启动训练脚本，模型将基于你的数据进行学习。

~~~bash
python train.py
~~~
*Log: 训练完成后，最佳模型将自动保存至 `save_models/` 目录。*

### Phase 3: 启动微服务 (Backend)
启动 Flask 后端 API，加载训练好的模型。

~~~bash
python api.py
~~~
*Status: 服务将运行在 `http://0.0.0.0:8004`，等待调用。*

### Phase 4: 启动可视化界面 (Frontend)
新开一个终端，启动 Streamlit 前端。

~~~bash
streamlit run app.py
~~~
*Action: 浏览器将自动打开，你可以在网页上直接测试模型效果。*

---

## 📡 API 接口规范 (Interface Specification)

**Base URL**: `http://127.0.0.1:8004`

### 1. 新闻分类预测 (Predict)

* **Endpoint**: `/predict`
* **Method**: `POST`
* **Content-Type**: `application/json`

**请求参数 (Request):**

| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `text` | string | 是 | 需要分类的新闻文本内容 |

**请求示例 (cURL):**

~~~bash
curl -X POST [http://127.0.0.1:8004/predict](http://127.0.0.1:8004/predict) \
     -H "Content-Type: application/json" \
     -d '{"text": "中华女子学院：本科层次仅1专业招男生"}'
~~~

**响应示例 (Response):**

~~~json
{
    "text": "中华女子学院：本科层次仅1专业招男生",
    "pred_class": "教育"
}
~~~

---

## 📚 参考文献 (References)

本项目的核心算法基于 Google 的 BERT 论文：

1.  **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
    * *Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova*, 2018
    * [Paper Link](https://arxiv.org/abs/1810.04805)
2.  **Hugging Face Transformers Documentation**
    * [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)

---

## ❤️ 致谢 (Acknowledgments)

* **Hugging Face**: 感谢其提供的开源 Transformers 库，极大地降低了 NLP 开发门槛。
* **PyTorch**: 提供了灵活且强大的深度学习框架。
* **Open Source Community**: 感谢每一位贡献者。

---

## 📄 版权说明 (License)

本项目采用 **MIT License** 开源协议。
> **BERT-News-Classifier**
>
> 2026 © Developed by BERT-Team
