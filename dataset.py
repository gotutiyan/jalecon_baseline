import random
import math
import torch
from typing import List, Tuple
from transformers import PreTrainedTokenizer, BertJapaneseTokenizer
import json

class Dataset():
    def __init__(
        self,
        srcs: List[str],
        labels: List[float],
        target_tokens: List[str],
        tokenizer: PreTrainedTokenizer,
        max_len: int
    ) -> None:
        self.srcs = srcs
        self.labels = labels
        self.target_tokens = target_tokens
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __getitem__(self, idx: int) -> dict:
        src = self.srcs[idx]
        label = self.labels[idx]
        target_token = self.target_tokens[idx]
        # encoded sentence will be "[CLS] src [SEP] target_token [SEP]"
        # Of course the src includes <unused1> <unused2> span.
        encode = self.tokenizer(
            src, target_token,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encode['input_ids'].squeeze(),
            'attention_mask': encode['attention_mask'].squeeze(),
            'token_type_ids': encode['token_type_ids'].squeeze(),
            'labels': torch.tensor(label)
        }

    def __len__(self):
        return len(self.srcs)

def process_sentence(
    content: str
):
    srcs = []
    labels = []
    target_tokens = []
    sent_info, *token_info_list = content.split('\n')
    sent_id, number_of_sent, src = sent_info.split('\t')
    for token_info in token_info_list:
        target_token, position, *others = token_info.split('\t')
        if others[-1] == '-':
            continue
        complexity = float(others[-1])
        start, end = position.split(':')
        start = int(start)
        end = int(end)
        tokens = src.split('|')
        fixed_tokens = tokens[:start] + ['<unused0>'] + [target_token] + ['<unused1>'] + tokens[end:]
        fixed_src = ''.join(fixed_tokens)
        srcs.append(fixed_src)
        labels.append(complexity)
        target_tokens.append(target_token)
    return sent_id, srcs, labels, target_tokens

def generate_dataset(
    input_file: str,
    cv_split: List[List[int]],
    tokenizer: PreTrainedTokenizer,
    max_len: int,
) -> Tuple[Dataset, Dataset]:
    '''
    This function recieves input file path(s) and returns a Dataset instance.
    '''
    contents = open(input_file).read().rstrip().split('\n\n')
    train_srcs = []
    valid_srcs = []
    train_labels = []
    valid_labels = []
    train_target_tokens = []
    valid_target_tokens = []
    for content in contents:
        sent_id, srcs, labels, target_tokens = process_sentence(
            content
        )
        if sent_id in cv_split[0]: # If the sent-id in the train split
            train_srcs += srcs
            train_labels += labels
            train_target_tokens += target_tokens
        else: # otherwise
            valid_srcs += srcs
            valid_labels += labels
            valid_target_tokens += target_tokens
    train_dataset = Dataset(
        train_srcs,
        train_labels,
        train_target_tokens,
        tokenizer=tokenizer,
        max_len=max_len
    )
    valid_dataset = Dataset(
        valid_srcs,
        valid_labels,
        valid_target_tokens,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return train_dataset, valid_dataset

def test():
    # We use the first instance of official dataset (ck.txt) for the following example.
    # https://github.com/naist-nlp/jalecon/blob/main/ck.txt
    content = '''gov-1-1	1	Q|：|今回|長官|に|就任|さ|れ|まし|た|が|、|とりわけ|在任|中|に|成し遂げ|たい|こと|など|の|抱負|を|お|聞か|せ|ください|。
Q	0:1	-	7	0	0	0	0.0
：	1:2	-	7	0	0	0	0.0
今回	2:3	-	7	0	0	0	0.0
長官	3:4	-	6	0	1	0	0.096
に	4:5	-	7	0	0	0	0.0
就任さ	5:7	-	7	0	0	0	0.0
れ	7:8	-	7	0	0	0	0.0
まし	8:9	-	7	0	0	0	0.0
た	9:10	-	7	0	0	0	0.0
が	10:11	-	7	0	0	0	0.0
、	11:12	-	7	0	0	0	0.0
とりわけ	12:13	-	4	2	1	0	0.19
在任	13:14	-	7	0	0	0	0.0
中	14:15	-	7	0	0	0	0.0
に	15:16	-	7	0	0	0	0.0
成し遂げ	16:17	-	6	0	1	0	0.096
たい	17:18	-	6	0	1	0	0.096
こと	18:19	-	6	0	1	0	0.096
など	19:20	-	7	0	0	0	0.0
の	20:21	-	7	0	0	0	0.0
抱負	21:22	-	7	0	0	0	0.0
を	22:23	-	7	0	0	0	0.0
お聞かせください	23:27	-	7	0	0	0	0.0
。	27:28	-	7	0	0	0	0.0'''
    sent_id, srcs, labels, target_token = process_sentence(content)
    print(f'{sent_id=}')
    for s, l, t in zip(srcs, labels, target_token):
        print(s)
        print(l, t)
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<unused0>', '<unused1>']}
    )
    encode = tokenizer(srcs[0], target_token[0])
    print(tokenizer.convert_ids_to_tokens(encode['input_ids']))
    
    

if __name__ == '__main__':
    test()