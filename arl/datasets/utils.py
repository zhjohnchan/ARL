import torch


def record_ent_ref(encoding, txt_ents):
    encoding["txt_label"] = []
    encoding["txt_ents"] = []
    encoding["ent_ref"] = [False] * len(encoding["input_ids"])
    for ent in txt_ents:
        beg_char = ent[0]
        end_char = ent[1] - 1
        beg_token = encoding.char_to_token(beg_char)
        end_token = encoding.char_to_token(end_char)
        if beg_token is not None and end_token is not None:
            for pos in range(beg_token + 1, end_token + 1):
                encoding["ent_ref"][pos] = True
            encoding["txt_label"].append(ent[2])
            encoding["txt_ents"].append(ent)
    return encoding


def create_pos_matrix(encoding, max_text_len, max_ent_len, mlm_labels=None):
    txt_ents = encoding["txt_ents"]
    pos_matrix = torch.zeros((max_ent_len, max_text_len), dtype=torch.float)
    ent_ids = torch.ones(max_ent_len, dtype=torch.long) * (-100)
    ent_masks = torch.zeros(max_ent_len, dtype=torch.bool)

    counter = 0
    for ent_idx, ent in enumerate(txt_ents):
        beg_char = ent[0]
        end_char = ent[1] - 1
        beg_token = encoding.char_to_token(beg_char)
        end_token = encoding.char_to_token(end_char)
        if beg_token is not None and end_token is not None:
            if mlm_labels is not None and (mlm_labels[beg_token:end_token + 1] != -100).sum() != 0:
                continue
            for pos in range(beg_token, end_token + 1):
                pos_matrix[counter][pos] = True
            ent_ids[counter] = ent[2]
            ent_masks[counter] = True
            counter = counter + 1
            if counter == max_ent_len:
                break
    return pos_matrix, ent_ids, ent_masks
