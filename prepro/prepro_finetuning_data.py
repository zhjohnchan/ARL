import json
import os
import re
import random
import spacy
import scispacy
import pandas as pd

from tqdm import tqdm
from collections import Counter
from scispacy.linking import EntityLinker
from make_arrow import make_arrow, make_arrow_vqa, make_arrow_melinda

spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_scibert")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")


def read_entity_vocab():
    entity2id, relation2id = {}, {}
    for line_idx, line in enumerate(open("data/knowledge/entity2id.txt")):
        if line_idx == 0:
            continue
        line = line.strip().split("\t")
        entity2id[line[0]] = int(line[1])
    for line_idx, line in enumerate(open("data/knowledge/relation2id.txt")):
        if line_idx == 0:
            continue
        line = line.strip().split("\t")
        relation2id[line[0]] = int(line[1])
    return entity2id, relation2id


def parse_a_text(text):
    entities = []
    doc = nlp(text)
    for ent in doc.ents:
        # Noise Filtering
        if len(ent.text) == 1:
            continue
        # Link to UMLS
        if len(ent._.kb_ents) == 0:
            continue
        start_id = ent.start_char
        end_id = ent.end_char
        cuis = ent._.kb_ents
        cuis = [cui[0] for cui in cuis if cui[1] >= 0.95]
        if len(cuis) == 0:
            continue
        entities.append((start_id, end_id, ent.text, cuis[0]))
    return entities


def extract_umls(data, text_key="texts"):
    entity2id, relation2id = read_entity_vocab()
    for split in ["train", "val", "test"]:
        split_data = data[split]
        for sample_idx, sample in tqdm(enumerate(split_data)):
            image_entites = []
            text_entities = []
            for text in sample[text_key]:
                entities = parse_a_text(text)
                text_entities.append(entities)
                image_entites.extend(entities)
            if sample_idx < 5:
                print(text_entities)
            sample["image_entities"] = image_entites
            sample["text_entities"] = text_entities

            sample["image_entities"] = [entity2id[ent[-1]] for ent in sample["image_entities"] if ent[-1] in entity2id]
            sample["image_entities"] = sorted(set(sample["image_entities"]))
            sample["text_entities"] = [[[ent[0], ent[1], entity2id[ent[3]]] for ent in ents if ent[3] in entity2id]
                                       for ents in sample["text_entities"]]
    return data


def extract_umls_vqa(data, text_key="texts"):
    entity2id, relation2id = read_entity_vocab()
    for split in ["train", "val", "test"]:
        split_data = data[split]
        for sample_idx, sample in tqdm(enumerate(split_data)):
            image_entites = []
            text_entities = []
            text = sample[text_key]
            entities = parse_a_text(text)
            text_entities.append(entities)
            image_entites.extend(entities)
            if sample_idx < 5:
                print(text_entities)
            sample["image_entities"] = image_entites
            sample["text_entities"] = text_entities

            sample["image_entities"] = [entity2id[ent[-1]] for ent in sample["image_entities"] if ent[-1] in entity2id]
            sample["image_entities"] = sorted(set(sample["image_entities"]))
            sample["text_entities"] = [[[ent[0], ent[1], entity2id[ent[3]]] for ent in ents if ent[3] in entity2id]
                                       for ents in sample["text_entities"]]
            assert len(sample["text_entities"]) == 1
            sample["text_entities"] = sample["text_entities"][0]
    return data


def prepro_vqa_vqa_rad():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/finetune_data/vqa_rad/"
    image_root = f"{data_root}/images"

    for split in ["train", "val", "test"]:
        with open(f"{data_root}/{split}set.json", "r") as fp:
            samples = json.load(fp)
            for sample in samples:
                img_path = os.path.join(image_root, sample["image_name"])
                qid = sample["qid"]
                question = sample["question"]
                answer = sample["answer"]
                answer_type = sample["answer_type"]
                data[split].append({
                    "img_path": img_path,
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "answer_type": answer_type
                })
    extract_umls_vqa(data, text_key="question")
    make_arrow_vqa(data, "vqa_vqa_rad", "data/finetune_arrows_umls/")


def prepro_vqa_slack():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/finetune_data/slack/"
    image_root = f"{data_root}/imgs"

    for split, file in zip(["train", "val", "test"], ["train.json", "validate.json", "test.json"]):
        with open(f"{data_root}/{file}", "r") as fp:
            samples = json.load(fp)
            for sample in samples:
                if sample["q_lang"] != "en":
                    continue
                img_path = os.path.join(image_root, sample["img_name"])
                qid = sample["qid"]
                question = sample["question"]
                answer = sample["answer"]
                answer_type = sample["answer_type"]
                data[split].append({
                    "img_path": img_path,
                    "qid": qid,
                    "question": question,
                    "answer": answer,
                    "answer_type": answer_type
                })
    extract_umls_vqa(data, text_key="question")
    make_arrow_vqa(data, "vqa_slack", "data/finetune_arrows_umls/")


def prepro_vqa_medvqa2019():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/finetune_data/medvqa_2019/"
    image_root = "data/finetune_data/medvqa_2019/{}/images"

    offset = 0
    for split in ["train", "val", "test"]:
        samples = open(f"{data_root}/{split}/QA/Modality.csv").read().strip().split("\n") + \
                  open(f"{data_root}/{split}/QA/Organ.csv").read().strip().split("\n") + \
                  open(f"{data_root}/{split}/QA/Plane.csv").read().strip().split("\n")
        samples = [[idx + offset] + question.split("|") for idx, question in enumerate(samples)]
        offset += len(samples)
        for sample in samples:
            img_path = os.path.join(image_root.format(split), sample[1] + ".jpg")
            qid = sample[0]
            question = sample[2]
            answer = sample[3]
            answer_type = "OPEN"
            data[split].append({
                "img_path": img_path,
                "qid": qid,
                "question": question,
                "answer": answer,
                "answer_type": answer_type
            })
    extract_umls_vqa(data, text_key="question")
    make_arrow_vqa(data, "vqa_medvqa_2019", "data/finetune_arrows_umls/")


def prepro_vqa_medvqa2021():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/finetune_data/medvqa_2021/"
    image_root = "data/finetune_data/medvqa_2021/{}/images"

    all_images = []
    all_answers = []
    offset = 0
    for split in ["train", "val", "test"]:
        samples = open(f"{data_root}/{split}/label.txt").read().strip().split("\n")
        samples = [[idx + offset] + question.split("|") for idx, question in enumerate(samples)]
        offset += len(samples)
        for sample in samples:
            all_images.append(sample[1])
            all_answers.append(sample[3])

            img_path = os.path.join(image_root.format(split), sample[1] + ".jpg")
            qid = sample[0]
            question = sample[2]
            answer = sample[3]
            answer_type = "OPEN"
            data[split].append({
                "img_path": img_path,
                "qid": qid,
                "question": question,
                "answer": answer,
                "answer_type": answer_type
            })

    data_2019_root = "data/finetune_data/medvqa_2019/"
    image_2019_root = "data/finetune_data/medvqa_2019/{}/images"

    for split in ["train", "val", "test"]:
        samples = open(f"{data_2019_root}/{split}/QA/Abnormality.csv").read().strip().split("\n")
        samples = [[idx + offset] + question.split("|") for idx, question in enumerate(samples)]
        for sample in samples:
            if sample[3] not in all_answers:
                continue
            if sample[1] not in all_images:
                all_images.append(sample[1])
            else:
                continue
            offset += 1
            img_path = os.path.join(image_2019_root.format(split), sample[1] + ".jpg")
            qid = sample[0]
            question = sample[2]
            answer = sample[3]
            answer_type = "OPEN"
            data["train"].append({
                "img_path": img_path,
                "qid": qid,
                "question": question,
                "answer": answer,
                "answer_type": answer_type
            })

    extract_umls_vqa(data, text_key="question")
    make_arrow_vqa(data, "vqa_medvqa_2021", "data/finetune_arrows_umls/")


def prepro_cls_melinda():
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/finetune_data/melinda"
    image_root = f"{data_root}/melinda_images"

    for split, file in zip(["train", "val", "test"], ["train.csv", "dev.csv", "test.csv"]):
        samples = pd.read_csv(f"{data_root}/{file}")
        for sample_idx, sample in samples.iterrows():

            img_path = os.path.join(image_root, sample["figure_file"])
            texts = [sample["caption"]]
            i_meth = sample["i_meth"]
            p_meth = sample["p_meth"]
            i_meth_label = sample["i_meth_label"]
            p_meth_label = sample["p_meth_label"]

            if len(texts) > 0:
                data[split].append({
                    "img_path": img_path,
                    "texts": texts,
                    "i_meth": i_meth,
                    "p_meth": p_meth,
                    "i_meth_label": i_meth_label,
                    "p_meth_label": p_meth_label
                })
    extract_umls(data)
    make_arrow_melinda(data, "cls_melinda", "data/finetune_arrows_umls/")


def prepro_irtr_roco(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }
    roco_data_root = "data/pretrain_data/roco"
    roco_image_root = "data/pretrain_data/roco/{}/radiology/images/"

    for split in ["train", "val", "test"]:
        with open(f"{roco_data_root}/{split}/radiology/captions.txt", "r") as fp:
            lines = fp.read().strip().split("\n")
            random.shuffle(lines)
            for line_idx, line in enumerate(lines):
                str_splits = line.strip().split('\t')
                if len(str_splits) == 2:
                    img_path = os.path.join(roco_image_root.format(split), str_splits[0] + ".jpg")
                    texts = [str_splits[1]]
                    texts = [re.sub(r"\s+", " ", text) for text in texts]
                    texts = [text for text in texts if len(text.split()) >= min_length]
                    if len(texts) > 0:
                        data[split].append({
                            "img_path": img_path,
                            "texts": texts
                        })
                        if split == "val" and len(data[split]) == 2000:
                            break
                        if split == "test" and len(data[split]) == 2000:
                            break
    extract_umls(data)
    make_arrow(data, "irtr_roco", "data/finetune_arrows_umls/")


if __name__ == '__main__':
    prepro_vqa_vqa_rad()
    prepro_vqa_slack()
    prepro_vqa_medvqa2019()
    prepro_vqa_medvqa2021()
    prepro_cls_melinda()
    prepro_irtr_roco()
