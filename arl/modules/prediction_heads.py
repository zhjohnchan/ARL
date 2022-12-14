import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from arl.modules.position_embeddings import get_2d_sincos_pos_embed
from arl.modules.vision_encoders.clip_model import Transformer, LayerNorm


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MIMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.patch_size = config["patch_size"]
        self.num_patches = (config["image_size"] // config["patch_size"]) ** 2
        self.decoder_hidden_size = config["mim_decoder_hidden_size"]
        self.decoder_num_layers = config["mim_decoder_num_layers"]
        self.decoder_num_heads = config["mim_decoder_num_heads"]
        self.decoder_num_channels = 3 * config["patch_size"] ** 2

        self.decoder_embed = nn.Linear(self.hidden_size, self.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_size))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1,
                                                          self.decoder_hidden_size), requires_grad=False)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_hidden_size, int(self.num_patches ** .5), True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.decoder = Transformer(self.decoder_hidden_size, self.decoder_num_layers + 1, self.decoder_num_heads)
        self.decoder_norm = LayerNorm(self.decoder_hidden_size)
        self.decoder_pred = nn.Linear(self.decoder_hidden_size, self.patch_size ** 2 * 3, bias=True)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed.to(x.dtype)

        # apply Transformer blocks
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.decoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class AlignHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        triples = [[int(_) for _ in line.strip().split("\t")]
                   for line in open("data/knowledge/train2id.txt").read().strip().split("\n")[1:]]

        self.register_buffer("edge_index", torch.tensor(triples, dtype=torch.long).T[:2].contiguous())
        self.x = nn.Parameter(torch.load("data/knowledge/ent_embeddings.ckpt",
                                         map_location="cpu")["ent_embeddings.weight"], requires_grad=True)
        self.ent_hidden_size = self.x.shape[1]

        self.embedding_ln = nn.LayerNorm(self.ent_hidden_size, eps=1e-05)

        self.conv1 = GATConv(self.ent_hidden_size, 64, heads=8, dropout=0.2)
        self.conv2 = GATConv(64 * 8, self.ent_hidden_size, heads=1, concat=False, dropout=0.2)

        self.img_linear1 = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.img_linear2 = nn.Linear(config["hidden_size"], self.ent_hidden_size)
        self.txt_linear1 = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.txt_linear2 = nn.Linear(config["hidden_size"], self.ent_hidden_size)

        self.ent_embedding_only = config["ent_embedding_only"]

    def forward(self):
        x, edge_index = self.x, self.edge_index
        x = self.embedding_ln(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def img_classify(self, input):
        bs = input.shape[0]
        x = self.forward()

        input = F.relu(self.img_linear2(F.relu(self.img_linear1(input))))
        output = torch.bmm(input.unsqueeze(1), x.T.unsqueeze(0).repeat(bs, 1, 1)).squeeze(1)
        return output

    def txt_classify(self, input):
        bs = input.shape[0]
        x = self.forward()

        input = F.relu(self.txt_linear2(F.relu(self.txt_linear1(input))))
        output = torch.bmm(input.unsqueeze(1), x.T.unsqueeze(0).repeat(bs, 1, 1)).squeeze(1)
        return output

    def embeddings(self, ent_ids, ignore_index=-100):
        bs, ne = ent_ids.shape
        if not self.ent_embedding_only:
            x = self.forward()
        else:
            x = self.x
        embeddings = x.new_zeros((bs * ne, x.shape[1]))
        ent_ids = ent_ids.reshape(-1)
        ent_ids_ = ent_ids.clone()
        ent_ids_[ent_ids_ == ignore_index] = 0
        embeddings_ = x.index_select(0, ent_ids_)
        embeddings[ent_ids != ignore_index] = embeddings_[ent_ids != ignore_index]
        embeddings = embeddings.reshape(bs, ne, -1)
        return embeddings
