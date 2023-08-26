
import argparse
from transformers import BertJapaneseTokenizer, get_scheduler, BertForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from dataset import generate_dataset, Dataset
from tqdm import tqdm
from collections import OrderedDict
import json
from accelerate import Accelerator
import numpy as np
import random
from typing import List
from modeling import WrapperForTrain

def R2(ref: List[float], hyp: List[float]):
    mean_r = sum(ref) / len(ref)
    x = sum((r - h)**2 for r, h in zip(ref, hyp))
    xx = sum((r - mean_r)**2 for r in ref)
    return 1 - x/xx

def complexity_wise_MAE(ref: List[float], hyp: List[float]):
    results = {
        'Zero': [0, 0], # [sum of MAE, # of instances]
        'Easy': [0, 0],
        'Not Easy': [0, 0],
        'Difficult': [0, 0]
    }
    for r, h in zip(ref, hyp):
        category = None
        if r == 0:
            category = 'Zero'
        elif r <= 0.165:
            category = 'Easy'
        elif r <= 0.5:
            category = 'Not Easy'
        else:
            category = 'Difficult'
        results[category][0] += abs(r - h)
        results[category][1] += 1
    for k, v in results.items():
        try:
            results[k] = v[0] / v[1]
        except ZeroDivisionError:
            results[k] = 0
    return results

def evaluation(
    model: WrapperForTrain,
    valid_loader: Dataset,
):
    model.eval()
    ref = []
    hyp = []
    with torch.no_grad():
        for batch in valid_loader:
            outputs = model(**batch)
            ref += batch['labels'].view(-1).cpu().tolist()
            hyp += outputs.logits.view(-1).cpu().tolist()
    print(ref)
    print(hyp)
    r2 = R2(ref, hyp)
    categorical_MAE = complexity_wise_MAE(ref, hyp)
    return {
        'R2': r2,
        'MAE': categorical_MAE
    }
    
def train(
    model,
    loader: DataLoader,
    optimizer,
    epoch: int,
    accelerator: Accelerator,
    lr_scheduler
) -> float:
    model.train()
    log = {
        'loss': 0
    }
    with tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_main_process) as pbar:
        for _, batch in pbar:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                log['loss'] += loss.item()
                if accelerator.is_main_process:
                    pbar.set_description(f'[Epoch {epoch}] [TRAIN]')
                    pbar.set_postfix(OrderedDict(
                        loss=loss.item(),
                        lr=optimizer.optimizer.param_groups[0]['lr']
                    ))
    return {k: v/len(loader) for k, v in log.items()}

def valid(model,
    loader: DataLoader,
    epoch: int,
    accelerator: Accelerator
) -> float:
    model.eval()
    log = {
        'loss': 0
    }
    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_main_process) as pbar:
            for _, batch in pbar:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    log['loss'] += loss.item()
                    if accelerator.is_main_process:
                        pbar.set_description(f'[Epoch {epoch}] [VALID]')
                        pbar.set_postfix(OrderedDict(
                            loss=loss.item()
                        ))
    return {k: v/len(loader) for k, v in log.items()}

def main(args):
    state_config = json.load(open(os.path.join(args.restore_dir, 'training_state.json'))) if args.restore_dir else dict()
    current_epoch = state_config.get('current_epoch', -1) + 1
    min_valid_loss = state_config.get('min_valid_loss', float('inf'))
    seed = state_config.get('argparse', {'seed': args.seed}).get('seed')
    max_len = state_config.get('argparse', {'max_len': args.max_len}).get('max_len')
    log_dict = json.load(open(os.path.join(args.restore_dir, '../log.json'))) if args.restore_dir else dict()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    if args.restore_dir is not None:
        base_model = BertForSequenceClassification.from_pretrained(
            args.restore_dir
        )
        tokenizer = BertJapaneseTokenizer.from_pretrained(args.restore_dir)
    else:
        base_model = BertForSequenceClassification.from_pretrained(
            args.model_id,
            classifier_dropout=args.drop_out,
            num_labels=1
        )
        tokenizer = BertJapaneseTokenizer.from_pretrained(args.model_id)
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<unused0>', '<unused1>']}
        )
    # ModelForTrain is a wrapper of BertForSequenceClassification to use custom loss function
    model = WrapperForTrain(base_model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    cv_split_data = json.load(open('dataset/cv_splits.json'))
    split = cv_split_data[args.cv_split-1]
    train_dataset, valid_dataset = generate_dataset(
        input_file=args.input_file,
        cv_split=split,
        tokenizer=tokenizer,
        max_len=max_len
    )
    print(len(train_dataset))
    print(len(valid_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # train_loader = valid_loader
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.accumulation,
        num_training_steps=len(train_loader) * args.epochs,
    )
    tokenizer.save_pretrained(args.outdir)
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation)
    model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, lr_scheduler
    )
    # print(evaluation(args.outdir, valid_dataset, max_len, batch_size=args.batch_size))
    # return
    accelerator.wait_for_everyone()
    for epoch in range(current_epoch, args.epochs):
        train_log = train(model, train_loader, optimizer, epoch, accelerator, lr_scheduler)
        # valid_log = valid(model, valid_loader, epoch, accelerator)
        valid_log = {}
        log_dict[f'Epoch {epoch}'] = {
            'train_log': train_log,
            'valid_log': valid_log
        }
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Save checkpoint as the last checkpoint for each epoch
            accelerator.unwrap_model(base_model).save_pretrained(args.outdir)
            state_dict = {
                'current_epoch': epoch,
                'min_valid_loss': min_valid_loss,
                'argparse': args.__dict__
            }
            with open(os.path.join(args.outdir, 'training_state.json'), 'w') as fp:
                json.dump(state_dict, fp, indent=4)
            if epoch == args.epochs-1:
                log_dict['evaluation'] = evaluation(
                    model,
                    valid_loader
                )
            with open(os.path.join(args.outdir, 'log.json'), 'w') as fp:
                json.dump(log_dict, fp, indent=4)
    print('Finish')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--model_id', default='cl-tohoku/bert-base-japanese-v2')
    parser.add_argument('--outdir', default='models/sample/')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--restore_dir', default=None)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--cv_split', type=int, choices=[1,2,3,4,5])
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
