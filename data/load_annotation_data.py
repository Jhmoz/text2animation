import os
import json
def get_all_rephrases(rephrase_dir):
    files=os.listdir(rephrase_dir)
    rephrase_sents = {}
    for file in files:
        sents_id = file[:-4]
        file_path = os.path.join(rephrase_dir, file)
        with open(file_path, "r",encoding="utf-8",errors="ignore") as f:
            cur_sents=f.readlines()
        cur_sents=[sent.strip() for sent in cur_sents if sent not in ["\n",""]]
        rephrase_sents[sents_id]=cur_sents
    return rephrase_sents

def get_all_annotation(annotion_dir):
    files = os.listdir(annotion_dir)
    annatation = {}
    for file in files:
        sents_id = file[:-4]
        file_path = os.path.join(annotion_dir, file)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            sents=f.readline()
            label=f.readline()
        try:
            label=eval(label.strip())
        except:
            print(sents_id)
            print(label)
            label=input()
        annatation[sents_id]={
            "sentence":sents.strip(),
            "label":label
        }
    return annatation

if __name__=="__main__":
    rephrase_sents=get_all_rephrases("./rephrase")
    annotation=get_all_annotation("./DataSet_TenExample")
    corpurs_for_train = []
    for sent_id in annotation:
        if sent_id in rephrase_sents:
            assert annotation[sent_id]["sentence"] in rephrase_sents[sent_id],print(f"{sent_id}\n{annotation[sent_id]['sentence']}\n{rephrase_sents[sent_id]}")
            for sents in rephrase_sents[sent_id]:
                corpurs_for_train.append({
                    "sentence":sents,
                    "output":annotation[sent_id]["label"]
                })
        else:
            corpurs_for_train.append({
                "sentence": annotation[sent_id]["sentence"],
                "output": annotation[sent_id]["label"]
            })

    with open("./corpurs_for_train.json", "w", encoding="utf-8",errors="ignore") as f:
        json.dump(corpurs_for_train,f)
