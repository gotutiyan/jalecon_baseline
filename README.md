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

```sh
pip install torch transformers accelerate
git clone https://github.com/naist-nlp/jalecon.git dataset
```

After this, the directory structure looks like this:
```
jalecon_baseline/
├── dataset
│   ├── ck.txt
│   ├── cv_splits.json
│   ├── LICENSE
│   ├── mwe_list
│   ├── non_ck.txt
│   └── README.md
├── modeling.py
├── train.py
....
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

Most of configurations is already set the official one (as mentioned in Appendix G Experimental Setting in the paper). However, we use "constant" setting for the learning rate scheduler while the official uses linear decay (because it is not clear what factor and iteration are used).

### Evalaution

The training script also has evaluation scripts.  
After training, please refer to `<--outdir>/log.json`. It has "evaluation" value that contains like this:
```json
"evaluation": {
    "R2": 0.43230069464738186,
    "MAE": {
        "Zero": 0.0035927612345890317,
        "Easy": 0.07501028705967677,
        "Not Easy": 0.2129422956611961,
        "Difficult": 0.3131373669538233
    }
}
```
`"R2"` is a coefficient of determination and `"MAE"` is MAE by gold complexity score tier.

# Usage as API

The model is trained with `BertForSequenceClassification` as a regression task (i.e. num_labels=1).

- Please encode the sentence carefully. Our trained models assume that the input sentence is the format as described in the [paper](https://aclanthology.org/2023.bea-1.40/).
    - Surround the target token with `<unused0>` and `<unused1>`.
    - Pass sentence and target token to the tokenizer as two arguments to obtain `token_type_ids` appropriately.
    
- In addition, do not forget to pass the output logits to sigmoid function.

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
complexity = torch.sigmoid(outputs.logits)
print(complexity.view(-1).cpu().tolist())
```

# Performances obtained

We performed expriments on both CK and non-CK datasets for five splits.  
The following results is their average score.

Non-CK results are competitive with the results of the paper, but CK is a little low.

### CK

||Zero|Easy|Not easy|(Very) Difficult|R2|
|:--|:-:|:-:|:-:|:-:|:-:|
|Paper|0.0034|0.0676|0.1913|0.2954|0.4351|
|Ours|0.0033|0.0742|0.2154|0.3562|0.3272|

### Non-CK

||Zero|Easy|Not easy|(Very) Difficult|R2|
|:--|:-:|:-:|:-:|:-:|:-:|
|Paper|0.0066|0.0510|0.1169|0.2932|0.6142|
|Ours|0.0058|0.0508|0.1215|0.3032|0.6030|