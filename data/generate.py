import os

data_dir="十个案例修改版"
output_dir="annotion"
files=os.listdir(data_dir)
all_sents=[]
for file in files:
    with open(os.path.join(data_dir,file),"r",encoding="utf-8") as f:
        sents=f.readlines()
    all_sents.extend(sents)

template=[
    {'场景':'','动作角色':[],'动作':'','动作接收者':[],'动作关联者':[]}
]
os.mkdir(output_dir)
for sent_ind in range(len(all_sents)):
    sent = all_sents[sent_ind]
    with open(os.path.join(output_dir,f"{sent_ind}.txt"),"w",encoding="utf-8") as f:
        f.write(sent.strip())
        f.write("\n")
        f.write(str(template))


rephrase_dir="rephrase"
for sent_ind in range(len(all_sents)):
    sent = all_sents[sent_ind]
    with open(os.path.join(rephrase_dir,f"{sent_ind}.txt"),"w",encoding="utf-8") as f:
        f.write(sent.strip())
        f.write("\n")




