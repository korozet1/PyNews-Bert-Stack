import torch
from transformers import BertTokenizer
from bert_classifer_model import BertClassifier
from config import Config
# 初始化配置
conf = Config()
device = 'cpu'
tokenizer = conf.tokenizer
#实例化一个模型结构
model = BertClassifier().to(device)
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
model.load_state_dict(torch.load(conf.quantized_model_save_path))
model.eval()

#预测函数
def predict(data):
    #  处理输入数据data["text"]
    text = data["text"]
    if not text.strip():
        return {"text": text, "pred_class": None}

    # 分词并编码 tokenizer.encode_plus,支持返回pt
    encoded = tokenizer.encode_plus(text,return_tensors="pt")
    #获取input_ids与attention_mask
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    # 开启模型推理with torch.no_grad():
    with torch.no_grad():
        #模型预测
        logits = model(input_ids, attention_mask)
        #  torch.argmax获取最大logits的索引pred_idx
        pred_idx = torch.argmax(logits, dim=1).item()
        # 获取预测的类别conf.class_list[pred_idx]
        pred_class = conf.class_list[pred_idx]

    return {"text": text, "pred_class": pred_class}

if __name__ == "__main__":
    # 测试输入

    sample_data = {"text": "中华女子学院：本科层次仅1专业招男生"}
    result = predict(sample_data)
    print("预测结果：")
    print(result)


