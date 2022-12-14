import os
import re
import json
import random
import pandas as pd
import spacy
import pickle
import scispacy
from scispacy.linking import EntityLinker
from tqdm import tqdm
from collections import Counter
from OpenKE.train_transe import train_transe
from make_arrow import make_arrow, make_arrow_mimic_cxr

spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_scibert")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")


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


def extract_umls(data):
    for split in ["train", "val", "test"]:
        split_data = data[split]
        for sample in tqdm(split_data):
            image_entites = []
            text_entities = []
            for text in sample["texts"]:
                entities = parse_a_text(text)
                text_entities.append(entities)
                image_entites.extend(entities)
            sample["image_entities"] = image_entites
            sample["text_entities"] = text_entities
    return data


def prepro_medicat(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/pretrain_data/medicat"
    image_root = f"{data_root}/release/figures/"
    medicat_ann_path = f"{data_root}/release/s2_full_figures_oa_nonroco_combined_medical_top4_public.jsonl"

    medicat_samples = [json.loads(sample) for sample in open(medicat_ann_path).read().strip().split("\n")]
    medicat_samples = [sample for sample in medicat_samples if sample["radiology"]]
    indices = list(range(len(medicat_samples)))
    random.shuffle(indices)
    splits = {
        "train": indices[:-2000],
        "val": indices[-2000:-1000],
        "test": indices[-1000:],
    }
    for split, split_indices in splits.items():
        for sample_idx in split_indices:
            sample = medicat_samples[sample_idx]
            img_path = os.path.join(image_root, sample["pdf_hash"] + "_" + sample["fig_uri"])
            texts = []
            if "s2_caption" in sample and len(sample["s2_caption"]) > 0:
                texts.append(sample["s2_caption"])
            if "s2orc_references" in sample and sample["s2orc_references"] is not None and len(
                    sample["s2orc_references"]) > 0:
                texts.extend(sample["s2orc_references"])
            texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
            texts = [text for text in texts if len(text.split()) >= min_length]
            if len(texts) > 0:
                data[split].append({
                    "img_path": img_path,
                    "texts": texts
                })

    data = extract_umls(data)
    return data


def prepro_roco(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }
    roco_data_root = "data/pretrain_data/roco"
    roco_image_root = "data/pretrain_data/roco/{}/radiology/images/"
    medicat_roco_data_root = "data/pretrain_data/medicat"
    medicat_roco_paths = {
        "train": f"{medicat_roco_data_root}/net/nfs2.corp/allennlp/sanjays/roco_files/roco_train_references.jsonl",
        "val": f"{medicat_roco_data_root}/net/nfs2.corp/allennlp/sanjays/roco_files/roco_val_references.jsonl",
        "test": f"{medicat_roco_data_root}/net/nfs2.corp/allennlp/sanjays/roco_files/roco_test_references.jsonl"
    }

    medicat2roco = {}
    for split in ["train", "val", "test"]:
        with open(f"{roco_data_root}/{split}/radiology/dlinks.txt", "r") as fp:
            for line in fp:
                str_splits = line.strip().split('\t')
                medicat2roco[str_splits[1].split(' ')[2].split('/')[-1].split('.')[0] + "_" + str_splits[-1]] = \
                    str_splits[0]

    for split, path in medicat_roco_paths.items():
        samples = [json.loads(sample) for sample in open(path).read().strip().split("\n")]
        for sample in samples:
            img_path = os.path.join(roco_image_root.format(split), medicat2roco[sample["roco_image_id"]] + ".jpg")
            texts = []
            if "gorc_references" in sample and sample["gorc_references"] is not None and len(
                    sample["gorc_references"]) > 0:
                texts.extend(sample["gorc_references"])
            texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
            texts = [text for text in texts if len(text.split()) >= min_length]
            if len(texts) > 0:
                data[split].append({
                    "img_path": img_path,
                    "texts": texts
                })

    for split in ["train", "val", "test"]:
        with open(f"{roco_data_root}/{split}/radiology/captions.txt", "r") as fp:
            for line in fp:
                str_splits = line.strip().split('\t')
                if len(str_splits) == 2:
                    img_path = os.path.join(roco_image_root.format(split), str_splits[0] + ".jpg")
                    texts = [str_splits[1]]
                    texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
                    texts = [text for text in texts if len(text.split()) >= min_length]
                    if len(texts) > 0:
                        data[split].append({
                            "img_path": img_path,
                            "texts": texts
                        })

    data = extract_umls(data)
    return data


def prepro_mimic_cxr(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }
    data_root = "data/pretrain_data/mimic_cxr/"
    image_root = f"{data_root}/files"
    sectioned_path = f"{data_root}/mimic_cxr_sectioned.csv"
    metadata_path = f"{data_root}/mimic-cxr-2.0.0-metadata.csv"
    chexpert_path = f"{data_root}/mimic-cxr-2.0.0-chexpert.csv"
    split_path = f"{data_root}/mimic-cxr-2.0.0-split.csv"

    sectioned_data = pd.read_csv(sectioned_path)
    sectioned_data = sectioned_data.set_index("study")
    metadata = pd.read_csv(metadata_path)
    chexpert_data = pd.read_csv(chexpert_path)
    chexpert_data["subject_id_study_id"] = chexpert_data["subject_id"].map(str) + "_" + chexpert_data["study_id"].map(
        str)
    chexpert_data = chexpert_data.set_index("subject_id_study_id")
    chexpert_data = chexpert_data.fillna(0)
    chexpert_data[chexpert_data == -1] = 0
    split_data = pd.read_csv(split_path)
    split_data = split_data.set_index("dicom_id")

    for sample_idx, sample in metadata.iterrows():
        subject_id = str(sample["subject_id"])
        study_id = str(sample["study_id"])
        dicom_id = str(sample["dicom_id"])
        img_path = os.path.join(image_root,
                                "p" + subject_id[:2],
                                "p" + subject_id,
                                "s" + study_id,
                                dicom_id + ".png")
        split = split_data.loc[dicom_id].split
        if split == "validate":
            split = "val"
        if sample.ViewPosition not in ["PA", "AP"]:
            continue
        if subject_id + "_" + study_id not in chexpert_data.index:
            print("Missing {}".format(subject_id + "_" + study_id))
            continue
        if "s" + study_id not in sectioned_data.index:
            print("Missing {}".format("s" + study_id))
            continue

        chexpert = chexpert_data.loc[subject_id + "_" + study_id].iloc[2:].astype(int).tolist()

        texts = []
        if not pd.isna(sectioned_data.loc["s" + study_id]["impression"]):
            texts.append(sectioned_data.loc["s" + study_id]["impression"])
        if not pd.isna(sectioned_data.loc["s" + study_id]["findings"]):
            texts.append(sectioned_data.loc["s" + study_id]["findings"])
        texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
        texts = [text for text in texts if len(text.split()) >= min_length]

        if len(texts) > 0:
            data[split].append({
                "img_path": img_path,
                "texts": texts,
                "chexpert": chexpert
            })

    data = extract_umls(data)
    return data


def create_entity_vocab(datas, threshold=20):
    all_entities = []
    for data in datas:
        for split in ["train", "val", "test"]:
            split_data = data[split]
            for sample in split_data:
                sample_entities = [entity[3] for entity in sample["image_entities"]]
                all_entities.extend(sample_entities)
    entity_vocab = Counter(all_entities)
    entity_vocab = sorted(entity_vocab.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    entity_vocab = [entity[0] for entity in entity_vocab if entity[1] > threshold]

    fin = open("data/pretrain_data/train_umls.txt")
    entities = []
    relations = []
    triples = []

    for line_idx, line in tqdm(enumerate(fin)):
        line_splits = line.strip().split("\t")
        if line_splits[0] not in entity_vocab:
            continue
        if line_splits[1] not in entity_vocab:
            continue
        entities.append(line_splits[0])
        entities.append(line_splits[1])
        relations.append(line_splits[3])
        triples.append((line_splits[0], line_splits[1], line_splits[3]))

    entity_vocab = Counter(entities)
    entity_vocab = sorted(entity_vocab.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    entity2id = {k: i for i, (k, v) in enumerate(entity_vocab)}
    relation_vocab = Counter(relations)
    relation_vocab = sorted(relation_vocab.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    relation2id = {k: i for i, (k, v) in enumerate(relation_vocab)}

    os.makedirs("data/knowledge/", exist_ok=True)
    fout = open("data/knowledge/train2id.txt", "wt")
    fout.write(f"{len(triples)}\n")
    for triple in triples:
        fout.write(f"{entity2id[triple[0]]}\t{entity2id[triple[1]]}\t{relation2id[triple[2]]}\n")

    fout = open("data/knowledge/entity2id.txt", "wt")
    fout.write(f"{len(entity2id)}\n")
    for k, v in entity2id.items():
        fout.write(f"{k}\t{v}\t{linker.kb.cui_to_entity[k].canonical_name}\n")

    fout = open("data/knowledge/relation2id.txt", "wt")
    fout.write(f"{len(relation2id)}\n")
    for k, v in relation2id.items():
        fout.write(f"{k}\t{v}\n")

    return entity2id, relation2id


def filter_datas(datas, entity2id):
    for data in datas:
        for split in ["train", "val", "test"]:
            split_data = data[split]
            for sample in split_data:
                sample["image_entities"] = [entity2id[ent[-1]] for ent in sample["image_entities"]
                                            if ent[-1] in entity2id]
                sample["image_entities"] = sorted(set(sample["image_entities"]))
                sample["text_entities"] = [[[ent[0], ent[1], entity2id[ent[3]]] for ent in ents if ent[3] in entity2id]
                                           for ents in sample["text_entities"]]
    return datas


def main(cache_path="data/pretrain_data/cache_data.pkl"):
    if not os.path.exists(cache_path):
        medicat_data = prepro_medicat()
        roco_data = prepro_roco()
        mimic_cxr_data = prepro_mimic_cxr()
        fout = open(cache_path, "wb")
        cache_data = {"medicat_data": medicat_data, "roco_data": roco_data, "mimic_cxr_data": mimic_cxr_data}
        pickle.dump(cache_data, fout)
    else:
        fin = open(cache_path, "rb")
        cache_data = pickle.load(fin, encoding="bytes")
        medicat_data = cache_data["medicat_data"]
        roco_data = cache_data["roco_data"]
        mimic_cxr_data = cache_data["mimic_cxr_data"]

    if not (os.path.exists("data/knowledge/train2id.txt") and
            os.path.exists("data/knowledge/entity2id.txt") and
            os.path.exists("data/knowledge/relation2id.txt")):
        entity2id, relation2id = create_entity_vocab([medicat_data, roco_data, mimic_cxr_data])
    else:
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

    if not os.path.exists("data/knowledge/ent_embeddings.ckpt"):
        train_transe()

    datas = filter_datas([medicat_data, roco_data, mimic_cxr_data], entity2id)
    medicat_data, roco_data, mimic_cxr_data = datas[0], datas[1], datas[2]
    make_arrow(medicat_data, "medicat", "data/pretrain_arrows_umls/")
    make_arrow(roco_data, "roco", "data/pretrain_arrows_umls/")
    make_arrow_mimic_cxr(mimic_cxr_data, "mimic_cxr", "data/pretrain_arrows_umls/")


if __name__ == '__main__':
    main()
