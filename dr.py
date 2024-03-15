import torch.nn as nn
from transformers import XLMRobertaModel
import argparse
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from io import BytesIO
from pathlib import Path
from sklearn import metrics
from collections import defaultdict
import numpy as np


# 定义双塔模型结构
class ContraRelevanceModel(nn.Module):
    def __init__(self, xlmr_model_path, interaction=False):
        super(ContraRelevanceModel, self).__init__()
        self.xlmr = XLMRobertaModel.from_pretrained(xlmr_model_path)
        self.interaction = interaction

        self.query_mlp = nn.Linear(768, 768)
        self.item_mlp = nn.Linear(768, 768)

        self.query_proj = nn.Linear(768, 768)
        self.item_proj = nn.Linear(768, 768)

        self.output = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 2)
        )

        self.r_drop = nn.Dropout(0.1)

    def forward(self, query_input, query_attention_mask, item_input, item_attention_mask):
        # qi表征
        if not self.interaction:
            query_rep = self.xlmr(query_input, attention_mask=query_attention_mask).last_hidden_state
            item_rep = self.xlmr(item_input, attention_mask=item_attention_mask).last_hidden_state
        else:
            query_len = query_input.size(1)

            qi_input = torch.cat([query_input, item_input], dim=-1)
            qi_att_mask = torch.cat([query_attention_mask, item_attention_mask], dim=-1)

            qi_rep = self.xlmr(qi_input, qi_att_mask).last_hidden_state
            query_rep = qi_rep[:, :query_len, :]
            item_rep = qi_rep[:, query_len:, :]

        # qi任务表征
        query_rep = self.query_mlp(query_rep)
        item_rep = self.item_mlp(item_rep)

        # sentence表征
        query_rep = torch.mean(query_rep, 1)
        item_rep = torch.mean(item_rep, 1)

        logits = self.output(torch.cat([query_rep, item_rep], dim=-1))

        #
        query_rep_ori = self.r_drop(query_rep)
        item_rep_ori = self.r_drop(item_rep)

        query_rep_con = self.r_drop(query_rep)
        item_rep_con = self.r_drop(item_rep)

        # 对比学习投影
        query_rep = self.query_proj(query_rep)
        item_rep = self.item_proj(item_rep)

        return query_rep, item_rep, logits, query_rep_ori, item_rep_ori, query_rep_con, item_rep_con


# 定义对比损失函数
class ConstrastiveLoss(nn.Module):
    def __init__(self, device):
        super(ConstrastiveLoss, self).__init__()

        self.device = device
        self.temperature = 0.1
        self.ce = nn.CrossEntropyLoss()

    def forward(self, query_rep, item_rep, label=None, margin=None, group_loss=False):
        query_rep = nn.functional.normalize(query_rep, p=2, dim=1)
        item_rep = nn.functional.normalize(item_rep, p=2, dim=1)

        if not group_loss:
            logits_pos = torch.einsum('nc,nc->n', [query_rep, item_rep]).unsqueeze(-1)
            logits_neg_col = torch.einsum('nc,ck->nk', [query_rep, item_rep.transpose(0, 1)])

            logits_neg_col.masked_fill_(torch.eye(logits_neg_col.size(0)).bool().to(self.device), -1e-8)
            logits_neg_row = logits_neg_col.T
        else:
            group_batch = 128 if query_rep.size(0) % 128 == 0 else query_rep.size(0)
            query_rep = query_rep.view(-1, group_batch, query_rep.size(-1))
            item_rep = item_rep.view(-1, group_batch, item_rep.size(-1))

            logits_all = torch.matmul(query_rep, item_rep.transpose(1, 2))
            logits_pos = logits_all.diagonal(dim1=-2, dim2=-1).reshape(-1, 1)

            logits_neg_row = logits_all.transpose(1, 2).reshape(-1, item_rep.size(1))

            logits_neg_col = logits_all.view(-1, item_rep.size(1))

            logits_neg_col.masked_fill_(torch.eye(item_rep.size(1)).unsqueeze(0).repeat(item_rep.size(0), 1, 1).view(-1,
                                                                                                     item_rep.size(
                                                                                                         1)).bool().to(
                self.device), -1e-8)

            logits_neg_row.masked_fill_(torch.eye(item_rep.size(1)).unsqueeze(0).repeat(item_rep.size(0), 1, 1).view(-1,
                                                                                                     item_rep.size(
                                                                                                         1)).bool().to(
                self.device), -1e-8)

        mask_ratio = 0
        if margin:
            avg_logits = torch.mean(logits_pos)
            mask = torch.abs(logits_neg_col - avg_logits) < margin
            logits_neg_col.masked_fill_(mask, -1e-8)
            mask = torch.abs(logits_neg_row - avg_logits) < margin
            logits_neg_row.masked_fill_(mask, -1e-8)
            mask_ratio = torch.mean(torch.mean(mask.float()))

        logits_col = torch.cat([logits_pos, logits_neg_col], dim=1) / self.temperature
        logits_row = torch.cat([logits_pos, logits_neg_row], dim=1) / self.temperature

        if label is not None:
            logits_col = logits_col.masked_select(label.bool().unsqueeze(1).repeat(1, logits_col.size(1))).view(-1, logits_col.size(1))
            logits_row = logits_row.masked_select(label.bool().unsqueeze(1).repeat(1, logits_row.size(1))).view(-1, logits_row.size(1))

        labels = torch.zeros(logits_col.size(0), dtype=torch.long, device=self.device)

        loss = (self.ce(logits_col, labels) + self.ce(logits_row, labels)) / 2

        return loss, mask_ratio


# 定义数据集
class MyDataset(Dataset):
    def __init__(self, data_path, model_path, device):
        self.data_path = data_path
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        self.data = self.load_data()
        self.device = device

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self, ):
        return len(self.data)

    def load_data(self, ):
        reader = reader
        row_count = reader.get_row_count()
        res = []
        for _ in tqdm(range(row_count), desc='读取odps表'):
            row_data = reader.read(1)
            query = row_data[0][0].decode("utf-8").lower()
            title = row_data[0][1].decode("utf-8").lower()
            label = row_data[0][2]
            res.append([query, title, label])

        return res

    # @staticmethod
    def collate_fn(self, batch):
        batch_dict = {
            "query_ids": [],
            "query_attention_mask": [],
            "item_ids": [],
            "item_attention_mask": [],
            "label": [],
            "query": [],
            "item_title": [],
            "mlm_input_ids": [],
            "mlm_labels": []
        }

        query_token_list = []
        title_token_list = []
        mlm_text_list = []
        for query, title, label in batch:
            query_token_list.append(query)
            title_token_list.append(title)
            batch_dict["label"].append(label)

            mlm_text_list.append(query + " " + title)
        batch_dict["label"] = torch.tensor(batch_dict["label"], device=self.device).long()

        batch_dict["query"] = query_token_list
        batch_dict["item_title"] = title_token_list

        query_token_info = self.tokenizer(
            text=query_token_list,
            max_length=16,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        item_token_info = self.tokenizer(
            text=title_token_list,
            max_length=96,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        batch_dict["query_ids"] = query_token_info.input_ids.to(self.device)
        batch_dict["query_attention_mask"] = query_token_info.attention_mask.to(self.device)

        batch_dict["item_ids"] = item_token_info.input_ids.to(self.device)

        batch_dict["item_attention_mask"] = item_token_info.attention_mask.to(self.device)

        return batch_dict


def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""

    # print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    #
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = metrics.roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc


def compute_metrics(X, Y, Z, scores, labels, queries):
    precision, recall = X / Y, X / Z
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # auc
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # gauc
    gauc = cal_group_auc(labels, scores, queries)

    return f1, precision, recall, auc, gauc


class Evaluator(object):
    """评估与保存
    """

    def __init__(self, args):
        self.stage = args.stage
        self.best_val_f1_pointer = 0.
        self.best_val_f1_gp_pointer = 0.
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_path)

    def on_epoch_end(self, valid_dataloader, model, logs=None):
        metrics = self.evaluate(
            valid_dataloader,
            model
        )

        return metrics

    def evaluate(self, data, model):
        X, Y, Z = 0, 1e-10, 1e-10

        scores = []
        labels = []
        queries = []
        for batch in data:
            query_ids = batch["query_ids"]
            item_ids = batch["item_ids"]
            label = batch["label"]
            query_attention_mask = batch["query_attention_mask"]
            item_attention_mask = batch["item_attention_mask"]
            query = batch["query"]

            query_rep, item_rep, logits, query_rep_ori, item_rep_ori, query_rep_con, item_rep_con \
                = model(query_ids, query_attention_mask, item_ids, item_attention_mask)

            query_rep = nn.functional.normalize(query_rep, p=2, dim=1)
            item_rep = nn.functional.normalize(item_rep, p=2, dim=1)

            if self.stage == 1:
                score = torch.cosine_similarity(query_rep, item_rep, dim=-1)
                pred = torch.gt(score, 0.5)
            elif self.stage == 2:
                score = torch.softmax(logits, dim=-1)[:, 1]
                pred = torch.argmax(logits, dim=-1)

            scores.extend(score.cpu().numpy().tolist())
            labels.extend(label.cpu().numpy().tolist())
            queries.extend(query)

            X += torch.sum((pred == label) * torch.eq(label, 1))
            Y += sum(pred)
            Z += sum(label)

        f1, precision, recall, auc, gauc = compute_metrics(X, Y, Z, scores, labels, queries)

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "gauc": gauc
        }


# 模型保存oss
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


def save_on_master_local(args, model_state_dict, model_path):
    if is_main_process():
        torch.save(model_state_dict, model_path)
        print(f"saved model to {str(model_path)}")


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
        save_on_master_local(args, to_save, str(checkpoint_path))


# 定义训练流程
def train(args):
    print("加载数据集")
    train_dataset = MyDataset(
        args.train_table,
        args.model_path,
        args.device
    )

    valid_dataset = MyDataset(
        args.test_table,
        args.model_path,
        args.device
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
        batch_size=args.valid_batch_size,
        collate_fn=valid_dataset.collate_fn
    )

    print("加载模型")
    model = ContraRelevanceModel(args.model_path, interaction=args.interaction).to(args.device)

    if args.stage == 2:
        model_state_dict = torch.load(args.pretrained_model_path)
        model.load_state_dict(model_state_dict, strict=False)
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=args.warm_up_ratio * len(train_dataloader) * args.epochs,
    #     num_training_steps=len(train_dataloader) * args.epochs
    # )
    loss_fn = ConstrastiveLoss(args.device)
    loss_ce = nn.CrossEntropyLoss()

    total_step = 0
    total_loss = 0
    model.train()

    evaluator = Evaluator(args)

    metrics = {
        "f1": 0.,
        "precision": 0.,
        "recall": 0.,
        "auc": 0.,
        "gauc": 0.
    }
    mask_ratio = 0

    for epoch in range(args.epochs):
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar):
            query_ids = batch["query_ids"]
            item_ids = batch["item_ids"]
            label = batch["label"]
            query_attention_mask = batch["query_attention_mask"]
            item_attention_mask = batch["item_attention_mask"]

            query_rep, item_rep, logits, query_rep_ori, item_rep_ori, query_rep_con, item_rep_con \
                = model(query_ids, query_attention_mask, item_ids, item_attention_mask)

            if args.stage == 1:
                loss, mask_ratio = loss_fn(query_rep, item_rep, margin=args.margin, group_loss=args.group_loss)
                loss_q_r_drop, _ = loss_fn(query_rep_ori, query_rep_con, margin=args.margin)
                loss_i_r_drop, _ = loss_fn(item_rep_ori, item_rep_con, margin=args.margin)
                loss += 0.1 * (loss_q_r_drop + loss_i_r_drop) / 2
            elif args.stage == 2:
                loss_aux, _ = loss_fn(query_rep, item_rep, label=label, margin=args.margin)
                loss_q_r_drop, _ = loss_fn(query_rep_ori, query_rep_con, margin=args.margin)
                loss_i_r_drop, _ = loss_fn(item_rep_ori, item_rep_con, margin=args.margin)
                loss = loss_ce(logits, label) + 0.1 * (loss_q_r_drop + loss_i_r_drop) / 2
            total_loss += loss.item()

            pbar.set_description(f"Epoch[{epoch + 1}/{args.epochs}]:")
            pbar.set_postfix({
                "loss": f"{total_loss / (total_step + step + 1):.03f}",
                "F1": f"{metrics['f1']:.03f}",
                "Precision": f"{metrics['precision']:.03f}",
                "Recall": f"{metrics['recall']:.03f}",
                "AUC": f"{metrics['auc']:.03f}",
                "GAUC": f"{metrics['gauc']:.03f}",
                "mask_ratio": f"{mask_ratio:.03f}"
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if args.rank == 0 and (total_step + step + 1) * args.world_size % 100 == 0:
                with torch.no_grad():
                    metrics = evaluator.on_epoch_end(valid_dataloader, model.eval())
                model.train()

                pbar.set_postfix({
                    "loss": f"{total_loss / (total_step + step + 1):.03f}",
                    "F1": f"{metrics['f1']:.03f}",
                    "Precision": f"{metrics['precision']:.03f}",
                    "Recall": f"{metrics['recall']:.03f}",
                    "AUC": f"{metrics['auc']:.03f}",
                    "GAUC": f"{metrics['gauc']:.03f}",
                    "mask_ratio": f"{mask_ratio:.03f}"
                })

        total_step += step + 1
        try:
            save_model(
                args=args,
                model_without_ddp=model.module,
                optimizer=optimizer,
                epoch=epoch
            )
        except:
            pass


# 超参数设置
def get_args_parser():
    parser = argparse.ArgumentParser("Revelance model training", add_help=False)

    # 样本
    parser.add_argument('--tables', default="", type=str, help="placeholder")
    parser.add_argument('--train_table',
                        default="",
                        type=str, help='')
    parser.add_argument('--test_table',
                        default="",
                        type=str, help='')

    # 模型
    parser.add_argument(
        "--batch_size",
        default=196,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--valid_batch_size",
        default=1024,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--model_path", default="xlm-roberta-base", type=str,
                        help="path of pretrained model")
    parser.add_argument("--pretrained_model_path", default="", type=str)
    parser.add_argument("--margin", default=None, type=float)
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--warm_up_ratio", default=0.1, type=float)
    parser.add_argument("--stage", default=1, type=int)
    parser.add_argument("--group_loss", action='store_true', help="")


    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--interaction", action='store_true', help="")

    return parser


# 主函数
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    args.train_table = args.train_table
    args.test_table = args.test_table
    args.device = torch.device(args.device)

    torch.multiprocessing.set_start_method('spawn')
    local_rank = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    print("world size:", args.world_size, " rank:", args.rank, " local rank:", args.local_rank)

    train(args)
