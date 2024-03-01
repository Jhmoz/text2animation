from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModel
import torch
import os
from tqdm import tqdm

checkpoint_dir = "../output/model-v3"
model_name_or_path = "/home/zyy/models/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint_dir, trust_remote_code=True).to("cuda")


def eval_test():
    test = [
        "爸爸带小明去森林里散步",
        "爸爸带小明去森林里散步去了。",
        "爸爸妈妈带小明去医院里找斑马医生看病去了",
        "奶奶带小强去学校里上学",
        "兔子医生给了小强一瓶魔法药水和一颗草莓",
        "爸爸妈妈带小明去公园去了",
        "小狐狸来找小熊玩耍",
        "小明变绿了",
        "小明把小老虎打了一顿",
        "小明带着小兔子爬上了山坡",
        "小美跳了起来，看到了远处的山上有一个小鸟",
        "小明走丢了",
        "小刘小郑小王他们几个去操场上踢足球",
        "丹顶鹤回家里面画画去了",
        "小牛被大灰狼狠狠地吃掉了",
        "小羊被大灰狼吃掉了",
        "大灰狼吃掉了小绵羊",
        "小明的身体健康了",
        "小明痊愈了",
        "小恐龙身体健康了",
        "小明吹了一个大气球",
        "小恐龙和长颈鹿在操场上打篮球",
        "小兔子远远地看见了一只大灰狼"
    ]
    with open("_eval_test_v3.txt","w",encoding="utf-8",errors="ignore") as f:
        for s in tqdm(test):
            prompt = "假设你是一个动画制作项目的脚本提取员，需要你将给定的中文句子中所有人物的动作都提取出来，形成可渲染的动画脚本以供角色动画制作。“在”、“带”作为专门的动作需要被表现出来。动画脚本一共包括五个要素：场景，动作，动作角色，动作接受者以及动画关联者。\n所给中文句子："
            test_sents = f"{s}\n解析结果："
            ts = prompt + test_sents
            f.write(f"{s}\n")
            f.write(model.chat(tokenizer, ts, history=[],temperature=0.01)[0])
            f.write("\n\n")


def eval_train():
    train = []
    annotation_dir = "./data/DataSet_TenExample"
    annotations = os.listdir(annotation_dir)
    for i in range(len(annotations)):
        annotation = str(i)+".txt"
        with open(os.path.join(annotation_dir, annotation), "r",encoding="utf-8",errors="ignore") as f:
            sent = f.readline().strip()
            label = eval(f.readline().strip())
        train.append(
            {
                "sent":sent,
                "label":label
            }
        )

    with open("_eval_train_v3.txt","w",encoding="utf-8",errors="ignore") as f:
        for s in tqdm(train):
            prompt = "假设你是一个动画制作项目的脚本提取员，需要你将给定的中文句子中所有人物的动作都提取出来，形成可渲染的动画脚本以供角色动画制作。“在”、“带”等词也作为专门的动作需要被表现出来。动画脚本一共包括五个要素：场景，动作，动作角色，动作接受者以及动画关联者。\n所给中文句子："
            test_sents = f"{s['sent']}\n解析结果："
            ts = prompt + test_sents
            f.write(f"{s['sent']}\n{s['label']}\n")
            f.write(model.chat(tokenizer, ts, history=[],temperature=0.01)[0])
            f.write("\n\n")

if __name__ == "__main__":
    eval_test()
    #eval_train()
