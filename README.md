# JaLeCon Baseline

This is a reproduction of the model for JaLeCon which is introduced by the following [paper](https://aclanthology.org/2023.bea-1.40):

```bibtex
@inproceedings{ide-etal-2023-japanese,
    title = "{J}apanese Lexical Complexity for Non-Native Readers: A New Dataset",
    author = "Ide, Yusuke  and
      Mita, Masato  and
      Nohejl, Adam  and
      Ouchi, Hiroki  and
      Watanabe, Taro",
    booktitle = "Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.bea-1.40",
    doi = "10.18653/v1/2023.bea-1.40",
    pages = "477--487"
}
```

# Install
Confirmed that it works on python 3.8.10.
```sh
pip install -r requirements.txt
git clone https://github.com/naist-nlp/jalecon.git dataset
```

After that, the directory structure looks like the following:
```
jalecon_baseline
├── dataset
│   ├── ck.txt
│   ├── cv_splits.json
│   ├── LICENSE
│   ├── mwe_list
│   ├── non_ck.txt
│   └── README.md
├── dataset.py
├── LICENSE
├── modeling.py
├── README.md
└── train.py
```

# Train

```sh
accelerate launch train.py \
    --input_file dataset/ck.txt \
    --outdir models/sample \
    --cv_split 1
```

- `--input_file` can be either `dataset/ck.txt` or `dataset/non_ck.txt`
- `--cv_split` can be one of 1, 2, 3, 4, and 5. This is the index of the split in `dataset/cv_split.json`.

### Evaluation

The training script includes evaluation scripts.  
After training, please refer to `<--outdir>/log.json`. It has "evaluation" value that contains like this:
```json
"evaluation": {
    "R2": 0.3909226117501584,
    "MAE": {
        "Zero": 0.0034583500462579274,
        "Easy": 0.06634619035699134,
        "Not Easy": 0.18918060780475465,
        "Difficult": 0.3481951270077843
    }
}
```
`"R2"` is a coefficient of determination and `"MAE"` is the MAE by gold complexity score tier.

# Usage as API

The model is trained with `BertForSequenceClassification` as a regression task (i.e. num_labels=1).

- Please encode a sentence carefully. Our trained models assume that sentences are entered in the format described in the [paper](https://aclanthology.org/2023.bea-1.40/).
    - Surround the target token with `<unused0>` and `<unused1>`.
    - Pass the sentence and the target token to the tokenizer as two arguments to obtain `token_type_ids` appropriately.
- Pass the output logits to sigmoid function.

### Example
```python
from transformers import BertForSequenceClassification, AutoTokenizer
import torch
path = '<--outdir>'
model = BertForSequenceClassification.from_pretrained(path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(path)
# This example uses the first instance of https://github.com/naist-nlp/jalecon/blob/main/ck.txt.
src = 'Q：今回長官に就任されましたが、<unused0>とりわけ<unused1>在任中に成し遂げたいことなどの抱負をお聞かせください。'
target_tokens = 'とりわけ'
encode = tokenizer(src, target_tokens, return_tensors='pt')
with torch.no_grad():
    outputs = model(**encode)
complexity = torch.sigmoid(outputs.logits).view(-1).cpu().tolist()
print(complexity)
```

# Performances obtained

We performed expriments on both CK and non-CK datasets for five splits.  
The following results are their average score.

Non-CK results are competitive with the results of the paper, but CK is a little low.

### CK

||↓Zero|↓Easy|↓Not easy|↓(Very) Difficult|↑R2|
|:--|:-:|:-:|:-:|:-:|:-:|
|Paper|0.0034|0.0676|0.1913|0.2954|0.4351|
|Ours|0.0034|0.0721|0.2170|0.3479|0.3216|

### Non-CK

||↓Zero|↓Easy|↓Not easy|↓(Very) Difficult|↑R2|
|:--|:-:|:-:|:-:|:-:|:-:|
|Paper|0.0066|0.0510|0.1169|0.2932|0.6142|
|Ours|0.0059|0.0503|0.1201|0.3043|0.6024|