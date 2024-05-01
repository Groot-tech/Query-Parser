#! -*- coding:utf-8 -*-

import numpy as np
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from bert4torch.snippets import sequence_padding, seed_everything
from bert4torch.losses import MultilabelCategoricalCrossentropy
from bert4torch.layers import GlobalPointer, EfficientGlobalPointer
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
from tqdm import tqdm

import os
import argparse

from io import BytesIO
from pathlib import Path

# 固定seed
seed_everything(42)

categories = [
    "Audience",
    "Brand",
    "Color",
    "Design",
    "Event",
    "Function",
    "IP",
    "ItemCategory",
    "Marketing",
    "Material",
    "Pattern",
    "Place",
    "ProductModel",
    "Shape",
    "Smell",
    "Spec",
    "StopWord",
    "Style",
    "Time"
]
categories_id2label = {i: k for i, k in enumerate(categories, start=1)}
categories_label2id = {k: i for i, k in enumerate(categories, start=1)}

ner_vocab_size = len(categories_label2id)


# 模型保存
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(args, model_state_dict, model_path):
    if is_main_process():

        torch.save(model_state_dict, model_path)
        print(f"saved {args.ner_model} model to {str(model_path)}")


def save_model(args, epoch, model_without_ddp, optimizer):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    epoch_name = str(epoch)

    checkpoint_paths = [output_dir / ('checkpoint-%s.bin' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = None
        if ".pth" in str(checkpoint_path):
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
        elif ".bin" in str(checkpoint_path):
            to_save = model_without_ddp.state_dict()
        else:
            pass
        save_on_master(args, to_save, str(checkpoint_path))


# 加载数据集
class MyDataset(Dataset):
    def __init__(self, table_path, args, data_type="train"):
        self.table_path = table_path
        self.data_type = data_type
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_path)
        self.device = args.device
        self.max_seq_len_hp = args.max_seq_len_hp
        self.data = self.load_data()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self, ):
        return len(self.data)

    def load_data(self, ):
        D = []

        reader = reader #
        row_count = reader.get_row_count()
        res = []
        for _ in tqdm(range(row_count), desc='读取数据'):
            row_data = reader.read(1)
            # query = row_data[0][0].decode("utf-8")
            terms = row_data[0][1].decode("utf-8")
            term_labels = row_data[0][2].decode("utf-8")
            # industry = row_data[0][3].decode("utf-8")
            d = ['']
            start_idx = 0
            end_idx = -1
            for i, c in enumerate(zip(terms.split(";;;"), term_labels.split(";;;"))):
                char, flag = c
                d[0] += " " + char
                end_idx += len(char) + 1
                if flag[0] == 'B':
                    d.append([start_idx, end_idx, flag[2:]])
                elif flag[0] == 'I':
                    d[-1][1] = end_idx
                start_idx = end_idx + 1

            d[0] = d[0][1:]
            D.append(d)

        return D

    def collate_fn(self, batch):
        # 建立分词器

        batch_token_ids, batch_start_labels, batch_end_labels, batch_gp_labels, batch_tokens = [], [], [], [], []
        cur_max_seq_len = 0
        for d in batch:
            tokens = self.tokenizer.tokenize(d[0])
            if len(tokens) > cur_max_seq_len:
                cur_max_seq_len = len(tokens)
            mapping = self.rematch(d[0])
            # print("mapping:", mapping)
            start_mapping = {j[1][0]: i for i, j in enumerate(mapping) if j}
            # print("start_mapping:", start_mapping)
            end_mapping = {j[1][-1]: i for i, j in enumerate(mapping) if j}
            # print("end_mapping: ",end_mapping)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            start_ids = [0] * len(tokens)
            end_ids = [0] * len(tokens)

            labels = np.zeros((ner_vocab_size + 1, len(tokens), len(tokens)))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    start_ids[start] = categories_label2id[label]
                    end_ids[end] = categories_label2id[label]

                    # 构造GP标签
                    label = categories_label2id[label]
                    labels[label, start, end] = 1

            batch_gp_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            batch_token_ids.append(token_ids)
            batch_start_labels.append(start_ids)
            batch_end_labels.append(end_ids)
            batch_tokens.append(tokens)

        max_seq_len = min(self.max_seq_len_hp, cur_max_seq_len)
        # print(len(batch_gp_labels[0]))
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, length=max_seq_len), dtype=torch.long,
                                       device=self.device)
        batch_start_labels = torch.tensor(sequence_padding(batch_start_labels, length=max_seq_len), dtype=torch.long,
                                          device=self.device)
        batch_end_labels = torch.tensor(sequence_padding(batch_end_labels, length=max_seq_len), dtype=torch.long,
                                        device=self.device)
        batch_gp_labels = torch.tensor(
            sequence_padding(batch_gp_labels, seq_dims=3, length=[ner_vocab_size + 1, max_seq_len, max_seq_len]),
            dtype=torch.long, device=self.device)
        batch_mask = batch_token_ids.gt(0).long()
        batch_gp_mask = torch.unsqueeze(batch_mask.unsqueeze(1).repeat(1, batch_mask.size(1), 1) + batch_mask.unsqueeze(2).repeat(1, 1, batch_mask.size(1)), 1).repeat(1, batch_gp_labels.size(1), 1, 1)

        return {
            "batch_token_ids": batch_token_ids,
            "batch_mask": batch_mask,
            "batch_start_labels": batch_start_labels,
            "batch_end_labels": batch_end_labels,
            "batch_gp_labels": batch_gp_labels,
            "batch_tokens": batch_tokens,
            "batch_gp_mask": batch_gp_mask
        }

    def rematch(self, text):

        orig_to_tok_index = self.tokenizer.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"]

        orig_to_tok_index_new = []
        for (start, end), start_tok in zip(orig_to_tok_index, self.tokenizer.encode(text, add_special_tokens=False)):
            cur_t = self.tokenizer.convert_ids_to_tokens(start_tok)

            if cur_t[0] == "▁" and end - start < len(cur_t):
                cur_t = cur_t[1:]

            orig_to_tok_index_new.append((cur_t, (start, end)))
        return orig_to_tok_index_new


# 定义bert上的模型结构
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.xlmr = XLMRobertaModel.from_pretrained(args.model_path)
        self.mid_linear = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.start_fc = nn.Linear(768, ner_vocab_size + 1)  # 0表示没有
        self.end_fc = nn.Linear(768, ner_vocab_size + 1)

    def forward(self, token_ids, attention_mask):
        sequence_output = self.xlmr(
            input_ids=token_ids,
            attention_mask=attention_mask,
        )  # [bts, seq_len, hdsz]
        seq_out = self.mid_linear(sequence_output[0])  # [bts, seq_len, mid_dims]
        start_logits = self.start_fc(seq_out)  # [bts, seq_len, num_tags]
        end_logits = self.end_fc(seq_out)  # [bts, seq_len, num_tags]

        return start_logits, end_logits


# 定义bert上的模型结构
class ModelV2(nn.Module):
    def __init__(self, args):
        super(ModelV2, self).__init__()
        self.xlmr = XLMRobertaModel.from_pretrained(args.model_path)
        self.global_pointer = GlobalPointer(hidden_size=768, heads=ner_vocab_size + 1,
                                            head_size=args.ner_head_size)
        self.mid_linear = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.start_fc = nn.Linear(768, ner_vocab_size + 1)  # 0表示没有
        self.end_fc = nn.Linear(768, ner_vocab_size + 1)

    def forward(self, token_ids, attention_mask):
        sequence_output = self.xlmr(
            input_ids=token_ids,
            attention_mask=attention_mask,
        )  # [bts, seq_len, hdsz]
        seq_out = self.mid_linear(sequence_output[0])  # [bts, seq_len, mid_dims]
        start_logits = self.start_fc(seq_out)  # [bts, seq_len, num_tags]
        end_logits = self.end_fc(seq_out)  # [bts, seq_len, num_tags]

        gp_logits = self.global_pointer(seq_out, attention_mask)

        return (start_logits, end_logits), gp_logits


class Loss(nn.CrossEntropyLoss):
    def forward(self, outputs, labels):
        start_logits, end_logits = outputs
        mask, start_ids, end_ids = labels
        start_logits = start_logits.view(-1, ner_vocab_size + 1)
        end_logits = end_logits.view(-1, ner_vocab_size + 1)

        # 去掉padding部分的标签，计算真实 loss
        active_loss = mask.view(-1) == 1
        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]
        active_start_labels = start_ids.view(-1)[active_loss]
        active_end_labels = end_ids.view(-1)[active_loss]

        start_loss = super().forward(active_start_logits, active_start_labels)
        end_loss = super().forward(active_end_logits, active_end_labels)
        return start_loss + end_loss


class GPLoss(MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        # print(y_pred.size(), y_true.size())
        y_true = y_true.view(y_true.shape[0] * y_true.shape[1], -1)  # [btz*ner_vocab_size+1, seq_len*seq_len]
        y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)  # [btz*ner_vocab_size+1, seq_len*seq_len]

        return super().forward(y_pred, y_true)


def train(args):
    # # 转换数据集
    train_dataset = MyDataset(
        args.train_table,
        args,
        data_type="train"
    )
    valid_dataset = MyDataset(
        args.valid_table,
        args,
        data_type="valid"
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=train_dataset.collate_fn
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        collate_fn=valid_dataset.collate_fn
    )

    if args.ner_model == "pointer":
        model = Model(args).to(args.device)
    else:
        model = ModelV2(args).to(args.device)
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    loss_fn_p = Loss()
    loss_fn_gp = GPLoss()

    evaluator = Evaluator(args)
    metrics = {
        "f1_pointer": 0.,
        "precision_pointer": 0.,
        "recall_pointer": 0.,
        "f1_gp_pointer": 0.,
        "precision_gp_pointer": 0.,
        "recall_gp_pointer": 0.,
        "f1_b_pointer": 0.,
        "precision_b_pointer": 0.,
        "recall_b_pointer": 0.,
        "f1_b_gp_pointer": 0.,
        "precision_b_gp_pointer": 0.,
        "recall_b_gp_pointer": 0.,
    }
    total_step = 0
    total_loss = 0
    model.train()

    for epoch in range(args.epochs):
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar):
            batch_token_ids = batch["batch_token_ids"]
            batch_mask = batch["batch_mask"]
            batch_start_labels = batch["batch_start_labels"]
            batch_end_labels = batch["batch_end_labels"]
            batch_gp_labels = batch["batch_gp_labels"]
            # batch_gp_mask = batch["batch_gp_mask"]

            if args.ner_model == "pointer":
                logits = model(batch_token_ids, batch_mask)
                loss = loss_fn_p(logits, [batch_mask, batch_start_labels, batch_end_labels])
            else:
                p_logits, gp_logits = model(batch_token_ids, batch_mask)
                loss_p = loss_fn_p(p_logits, [batch_mask, batch_start_labels, batch_end_labels])
                loss_gp = loss_fn_gp(gp_logits, batch_gp_labels)
                loss = 1.0  * loss_p + 0. * loss_gp

            total_loss += loss
            pbar.set_description(f"Epoch[{epoch + 1}/{args.epochs}]:")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.rank == 0 and (total_step + step + 1) * args.world_size % 100 == 0:
                with torch.no_grad():
                    metrics = evaluator.on_epoch_end(
                        valid_dataloader, model.eval())
                model.train()
            pbar.set_postfix({
                "loss": f"{total_loss / (total_step + step):.03f}",
                "F1_pt": f"{metrics['f1_pointer']:.03f}",
                "p_pt": f"{metrics['precision_pointer']:.03f}",
                "r_pt": f"{metrics['recall_pointer']:.03f}",
                "F1_gp": f"{metrics['f1_gp_pointer']:.03f}",
                "p_gp": f"{metrics['precision_gp_pointer']:.03f}",
                "r_gp": f"{metrics['recall_gp_pointer']:.03f}",
                "F1_b_pt": f"{metrics['f1_b_pointer']:.03f}",
                "p_b_pt": f"{metrics['precision_b_pointer']:.03f}",
                "r_b_pt": f"{metrics['recall_b_pointer']:.03f}",
                "F1_b_gp": f"{metrics['f1_b_gp_pointer']:.03f}",
                "p_b_gp": f"{metrics['precision_b_gp_pointer']:.03f}",
                "r_b_gp": f"{metrics['recall_b_gp_pointer']:.03f}",
            })
        total_step += step + 1

        save_model(
            args=args,
            model_without_ddp=model.module,
            optimizer=optimizer,
            epoch=epoch
        )


class Evaluator(object):
    """评估与保存
    """

    def __init__(self, args):
        self.best_val_f1_pointer = 0.
        self.best_val_f1_gp_pointer = 0.
        self.strict_decode = args.strict_decode
        self.ner_model = args.ner_model
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_path)

    def on_epoch_end(self, valid_dataloader, model, logs=None):
        metrics = self.evaluate(
            valid_dataloader,
            model
        )

        return metrics

    def evaluate(self, data, model):
        X_pointer, Y_pointer, Z_pointer = 0, 1e-10, 1e-10
        X_gp_pointer, Y_gp_pointer, Z_gp_pointer = 0, 1e-10, 1e-10

        X_b_pointer, Y_b_pointer, Z_b_pointer = 0, 1e-10, 1e-10
        X_b_gp_pointer, Y_b_gp_pointer, Z_b_gp_pointer = 0, 1e-10, 1e-10

        for batch in data:
            batch_token_ids = batch["batch_token_ids"]
            batch_mask = batch["batch_mask"]
            batch_start_labels = batch["batch_start_labels"]
            batch_end_labels = batch["batch_end_labels"]
            batch_gp_labels = batch["batch_gp_labels"]
            batch_tokens = batch["batch_tokens"]

            if self.ner_model == "pointer":
                start_logit, end_logit = model(batch_token_ids, batch_mask)
            else:
                (start_logit, end_logit), gp_logits = model(batch_token_ids, batch_mask)
                # gp_pointer
                entity_pred_gp_pointer, entity_true_gp_pointer, \
                boundary_pred_gp_pointer, boundary_true_gp_pointer = self.span_decode_gp_pointer(batch_tokens,
                                                                                                 gp_logits,
                                                                                                 batch_gp_labels)
                X_gp_pointer += len(entity_pred_gp_pointer & entity_true_gp_pointer)
                Y_gp_pointer += len(entity_pred_gp_pointer)
                Z_gp_pointer += len(entity_true_gp_pointer)

                X_b_gp_pointer += len(boundary_pred_gp_pointer & boundary_true_gp_pointer)
                Y_b_gp_pointer += len(boundary_pred_gp_pointer)
                Z_b_gp_pointer += len(boundary_true_gp_pointer)

            # pointer
            entity_pred_pointer, boundary_pred_pointer = self.span_decode_pointer(
                batch_tokens,
                start_logit,
                end_logit,
                batch_mask
            )
            entity_true_pointer, boundary_true_pointer = self.span_decode_pointer(
                batch_tokens,
                batch_start_labels,
                batch_end_labels
            )

            X_pointer += len(entity_pred_pointer.intersection(entity_true_pointer))
            Y_pointer += len(entity_pred_pointer)
            Z_pointer += len(entity_true_pointer)

            X_b_pointer += len(boundary_pred_pointer.intersection(boundary_true_pointer))
            Y_b_pointer += len(boundary_pred_pointer)
            Z_b_pointer += len(boundary_true_pointer)

        f1_pointer, precision_pointer, recall_pointer = self.compute_metrics(
            X_pointer,
            Y_pointer,
            Z_pointer
        )
        f1_gp_pointer, precision_gp_pointer, recall_gp_pointer = self.compute_metrics(
            X_gp_pointer,
            Y_gp_pointer,
            Z_gp_pointer
        )

        f1_b_pointer, precision_b_pointer, recall_b_pointer = self.compute_metrics(
            X_b_pointer,
            Y_b_pointer,
            Z_b_pointer
        )
        f1_b_gp_pointer, precision_b_gp_pointer, recall_b_gp_pointer = self.compute_metrics(
            X_b_gp_pointer,
            Y_b_gp_pointer,
            Z_b_gp_pointer
        )

        return {
            "f1_pointer": f1_pointer,
            "precision_pointer": precision_pointer,
            "recall_pointer": recall_pointer,
            "f1_gp_pointer": f1_gp_pointer,
            "precision_gp_pointer": precision_gp_pointer,
            "recall_gp_pointer": recall_gp_pointer,
            "f1_b_pointer": f1_b_pointer,
            "precision_b_pointer": precision_b_pointer,
            "recall_b_pointer": recall_b_pointer,
            "f1_b_gp_pointer": f1_b_gp_pointer,
            "precision_b_gp_pointer": precision_b_gp_pointer,
            "recall_b_gp_pointer": recall_b_gp_pointer
        }

    # 严格解码 baseline
    def span_decode_pointer(self, batch_tokens, start_preds, end_preds, mask=None):
        '''返回实体的start, end
        '''
        predict_entities = set()
        predict_boundaries = set()
        if mask is not None:  # 把padding部分mask掉
            start_preds = torch.argmax(start_preds, -1) * mask
            end_preds = torch.argmax(end_preds, -1) * mask

        start_preds = start_preds.cpu().numpy()
        end_preds = end_preds.cpu().numpy()

        for bt_i in range(start_preds.shape[0]):
            start_pred = start_preds[bt_i]
            end_pred = end_preds[bt_i]
            # 统计每个样本的结果
            pre_type = None
            pre_end = -1
            tokens = batch_tokens[bt_i]
            for i, s_type in enumerate(start_pred):
                if i <= pre_end and self.strict_decode:
                    continue
                if s_type == 0:
                    continue
                for j, e_type in enumerate(end_pred[i:]):

                    #                 if s_type == e_type:
                    #                     # [样本id, 实体起点，实体终点，实体类型]
                    #                     predict_entities.add((bt_i, i, i+j, categories_id2label[s_type]))
                    #                     break
                    if s_type == e_type:
                        # if e_type != 0:
                        # # [样本id, 实体起点，实体终点，实体类型]
                        # if s_type == pre_type and pre_type in [categories_label2id["Spec"], categories_label2id["Audience"],categories_label2id["ItemCategory"]]:
                        #     pre_entity = batch_res.pop()
                        #     i = pre_entity[3]
                        #     j += pre_entity[4] + 1
                        # print(s_type, e_type, tokens[i: i+j+1])
                        # 补上0导致的漏标
                        if pre_end + 1 < i:
                            if self.tokenizer.convert_tokens_to_string(tokens[pre_end + 1: i]) != "":
                                predict_entities.add((bt_i, i, i + j, categories_id2label[s_type]))
                                predict_boundaries.add((bt_i, i, i + j))
                        predict_entities.add((bt_i, i, i + j, categories_id2label[s_type]))
                        predict_boundaries.add((bt_i, i, i + j))
                        pre_end = i + j
                        pre_type = s_type
                        break
        return predict_entities, predict_boundaries

    def span_decode_gp_pointer(self, batch_tokens, scores, label, threshold=0):
        R = set()
        T = set()
        R_b = set()
        T_b = set()
        for i, score in enumerate(scores):
            tokens = batch_tokens[i]
            # for l, start, end in zip(*np.where(score.cpu() > threshold)):
            #     R.add((start, end, categories_id2label[l]))

            score = score.cpu().numpy()
            l, start, end = [x[0] for x in np.where(score == np.max(score))]
            # print(f"type:{l}, start: {start}, end: {end}")
            if l != 0:
                R.add((i, start, end, categories_id2label[l]))
                R_b.add((i, start, end))
                while start > 0:
                    # print(f"type: {l}, start:{start}")
                    l, start_ = [x[0] for x in np.where(score[1:, :start, start] == np.max(score[1:, :start, start]))]
                    l += 1
                    R.add((i, start_, start - 1, categories_id2label[l]))
                    R_b.add((i, start_, start - 1))
                    start = start_

                while end < score.shape[2] - 1:
                    # print(f"type: {l}, end:{end}")
                    l, length = [x[0] for x in
                                 np.where(score[1:, end + 1, end + 1:] == np.max(score[1:, end + 1, end + 1:]))]
                    l += 1
                    end_ = end + 1 + length
                    if self.tokenizer.convert_tokens_to_string(tokens[end + 1: end_ + 1]) != "":
                        R.add((i, end + 1, end_, categories_id2label[l]))
                        R_b.add((i, end + 1, end_))
                    end = end_

            for l, start, end in zip(*np.where(label[i].cpu() > 0)):
                T.add((i, start, end, categories_id2label[l]))
                T_b.add((i, start, end))

        return R, T, R_b, T_b

    @staticmethod
    def compute_metrics(X, Y, Z):
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall


# 超参设置
def get_args_parser():
    parser = argparse.ArgumentParser("NER model training", add_help=False)
    parser.add_argument('--tables', default="", type=str, help='ODPS input table names')
    parser.add_argument(
        "--batch_size",
        default=1024,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    parser.add_argument("--ner_model", default='pointer', type=str)

    # Model parameters
    parser.add_argument("--model_path", default="./xlm-roberta-base", type=str, help="path of pretrained model")
    parser.add_argument("--ner_head_size", type=int, default=768, help="global pointer dim")
    parser.add_argument("--max_seq_len_hp", type=int, default=32, help="max sequence length")
    parser.add_argument("--strict_decode", type=bool, default=True)

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    # oss parameters
    parser.add_argument("--oss_access", default="access", help="auth access")
    parser.add_argument("--oss_key", default="key", help="auth key")
    parser.add_argument("--oss_bucket", default="bucket", help="bucket to save model")

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.train_table = args.tables.split(",")[0]
    args.valid_table = args.tables.split(",")[1]
    args.device = torch.device(args.device)

    torch.multiprocessing.set_start_method('spawn')
    local_rank = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    print("world size:", args.world_size, " rank:", args.rank, " local rank:", args.local_rank)

    train(args)
