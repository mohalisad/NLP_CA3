import copy
import tqdm
import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel
from tokenizers.processors import TemplateProcessing
from torch.utils.data import Dataset


SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
LM_TRAINING_DATA = ['./t.txt']
WORD_TOKENIZER_FILE_NAME = './wtoken.json'
BPE_TOKENIZER_FILE_NAME = './bpetoken.json'

word_tokenizer = Tokenizer.from_file(WORD_TOKENIZER_FILE_NAME)
bpe_tokenizer = Tokenizer.from_file(BPE_TOKENIZER_FILE_NAME)

def add_post_processor_to(tokenizer: Tokenizer):
    tokenizer.post_processor = TemplateProcessing(
        single=f"{SOS_TOKEN} $0 {EOS_TOKEN}",
        special_tokens=[
            (X, tokenizer.token_to_id(X)) for X in [SOS_TOKEN, EOS_TOKEN]
        ]
    )
add_post_processor_to(word_tokenizer)
add_post_processor_to(bpe_tokenizer)

#_______________________________________________

class TextDataset(Dataset):
    def __init__(self, corpus_files):
        dataset_lines = []

        for file_name in LM_TRAINING_DATA:
            with open(file_name, 'r') as f:
                dataset_lines += f.readlines()
        dataset_lines = [line.strip() for line in dataset_lines]
                
        self.__lines = dataset_lines
        
    def __len__(self):
        return len(self.__lines)
    
    def __getitem__(self, idx):
        return self.__lines[idx]
    
    def get_tokenized(self, tokenizer, **tokenizer_args):
        return TokenizedTextDataset(self.__lines, tokenizer, tokenizer_args)

dataset = TextDataset(LM_TRAINING_DATA)
#_______________________________________________


def calc_ppl(dataset, model, tokenizer):
    nlls = []
    
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.enable_truncation(128)

    sum_len = 0

    for line in tqdm.tqdm(dataset):
        ids = tokenizer.encode(line).ids
        input_ids = torch.tensor(ids).to('cuda')
        target_ids = input_ids.clone()
        trg_len = len(ids)
        sum_len += trg_len
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / sum_len)

bpe_model = GPT2LMHeadModel.from_pretrained('gpt2_bpe').to('cuda')
print(f"PPL IS: {calc_ppl(dataset, bpe_model, bpe_tokenizer)}")
