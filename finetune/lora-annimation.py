from transformers import AutoTokenizer, AutoModel, DataCollatorForSeq2Seq, TrainingArguments, Trainer, set_seed
from datasets import Dataset
import json
import torch
from peft import LoraConfig, TaskType, get_peft_model

output_dir = "../output"
instruction = "假设你是一个动画制作项目的脚本提取员，需要你将给定的中文句子中所有人物的动作都提取出来，形成可渲染的动画脚本以供角色动画制作。“在”、“带”作为专门的动作需要被表现出来。动画脚本一共包括五个要素：场景，动作，动作角色，动作接受者以及动画关联者。\n所给中文句子："
model_name_or_path = "/home/zyy/models/chatglm3-6b"


def read_corpus():
    return json.load(open("./data/corpurs_for_train_v2.json", "r", encoding="utf-8"))


def prepare_dataset(data: dict):
    res = {
        "input": [],
        "output": []
    }
    for example in data:
        res["output"].append(str(example["output"]))
        res["input"].append("{}\n解析结果：".format(example["sentence"]))
    return res


corpus = prepare_dataset(read_corpus())
corpus = Dataset.from_dict(corpus)
print(len(corpus))
print(corpus[-1])

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer.decode(tokenizer.build_chat_input(instruction)["input_ids"][0])


def process_func(example):
    MAX_LENGTH = 512
    prompt_ids = tokenizer.build_chat_input(instruction + example["input"])
    ans_ids = tokenizer("\n" + example["output"], add_special_tokens=False)

    context_length = len(prompt_ids["input_ids"][0])
    input_ids = prompt_ids["input_ids"][0].numpy().tolist() + ans_ids["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = prompt_ids["attention_mask"][0].numpy().tolist() + ans_ids["attention_mask"] + [1]
    labels = [-100] * context_length + ans_ids["input_ids"] + [tokenizer.eos_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH - 1] + [input_ids[-1]]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH - 1] + [labels[-1]]
    else:
        padding_len = MAX_LENGTH - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_len
        attention_mask += [0] * padding_len
        labels += [-100] * padding_len
    assert len(input_ids) == len(
        labels), f"The lengths of input_ids ({len(input_ids)}) and labels ({len(labels)}) do not match"
    # print(len(input_ids))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# test_process_function
tokenized_examples = process_func(corpus[0])
print(tokenizer.decode(tokenized_examples["input_ids"]))
print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_examples["labels"]))))

tokenized_corpus = corpus.map(process_func, remove_columns=corpus.column_names)

model = AutoModel.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.half,
                                  trust_remote_code=True,
                                  device_map="auto")

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value"],
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
print(config)

model = get_peft_model(model, config)
model.print_trainable_parameters()
model = model.half()
model.cuda()
model.enable_input_require_grads()
for name, param in model.named_parameters():
    print(name, param.dtype)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=None,
    padding=False
)

training_args = TrainingArguments(
    output_dir="../output",
    per_device_train_batch_size=3,
    gradient_accumulation_steps=8,
    logging_steps=5,
    num_train_epochs=10,
    learning_rate=1e-4,
    adam_epsilon=5e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_corpus,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("../output/lora-weight-v3")

merge_model = model.merge_and_unload()
test="假设你是一个动画制作项目的脚本提取员，需要你将给定的中文句子中所有人物的动作都提取出来，形成可渲染的动画脚本以供角色动画制作。“在”、“带”作为专门的动作需要被表现出来。动画脚本一共包括五个要素：场景，动作，动作角色，动作接受者以及动画关联者。\n所给中文句子：爸爸妈妈带着小强一起去了医院里面找兔子医生看病\n解析结果："
print(merge_model.chat(tokenizer,test)[0])

merge_model = model.merge_and_unload()
merge_model.save_pretrained("../output/model-v3")

