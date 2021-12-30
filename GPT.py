#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install transformers accelerate tokenizers datasets
# pip install hazm


# In[2]:


# get_ipython().run_cell_magic('html', '', '<link href="https://v1.fontapi.ir/css/Vazir" rel="stylesheet">\n<link rel="stylesheet" href="style.css">')


# # <div class="farsi center">بسم الله الرحمن الرحیم</div>

# In[3]:


import typing
import copy
import torch
import tqdm


# In[4]:


TRAIN_TOKENIZERS = False

WORD_TOKENIZER_FILE_NAME = './wtoken.json'
BPE_TOKENIZER_FILE_NAME = './bpetoken.json'

BPE_VOCAB_SIZE = 10000
WORD_LEVEL_VOCAB_SIZE = 5000

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
ALL_TOKENS = [UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]

ALL_TRAINING_DATA = [
    './cultural.txt',
    './economics.txt',
    './politics.txt',
    './sports.txt'
]

LM_TRAINING_DATA = ['./t.txt'] #ALL_TRAINING_DATA[:1]


# # <div class="green">Tokenization</div>

# In[5]:


from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.processors import TemplateProcessing


# ## <span class="blue">Word Tokenizer</span>

# In[6]:


if TRAIN_TOKENIZERS:
    word_tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
    word_tokenizer.pre_tokenizer = Whitespace()
    word_trainer = WordLevelTrainer(vocab_size=WORD_LEVEL_VOCAB_SIZE, special_tokens=ALL_TOKENS)
    word_tokenizer.train(ALL_TRAINING_DATA, word_trainer)
    word_tokenizer.enable_padding(pad_token=PAD_TOKEN)
    word_tokenizer.save(WORD_TOKENIZER_FILE_NAME)
else:
    word_tokenizer = Tokenizer.from_file(WORD_TOKENIZER_FILE_NAME)


# In[ ]:


word_tokenizer.id_to_token


# ## <span class="blue">BPE Tokenizer</span>

# In[7]:


if TRAIN_TOKENIZERS:
    bpe_tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    bpe_tokenizer.pre_tokenizer = Whitespace()
    bpe_trainer = BpeTrainer(vocab_size=BPE_VOCAB_SIZE, special_tokens=ALL_TOKENS)
    bpe_tokenizer.train(ALL_TRAINING_DATA, bpe_trainer)
    bpe_tokenizer.enable_padding(pad_token=PAD_TOKEN)
    bpe_tokenizer.save(BPE_TOKENIZER_FILE_NAME)
else:
    bpe_tokenizer = Tokenizer.from_file(BPE_TOKENIZER_FILE_NAME)


# ## <span class="blue">Post Processing</span>

# In[8]:


def add_post_processor_to(tokenizer: Tokenizer):
    tokenizer.post_processor = TemplateProcessing(
        single=f"{SOS_TOKEN} $0 {EOS_TOKEN}",
        special_tokens=[
            (X, tokenizer.token_to_id(X)) for X in [SOS_TOKEN, EOS_TOKEN]
        ]
    )
add_post_processor_to(word_tokenizer)
add_post_processor_to(bpe_tokenizer)


# ## <div class="blue right farsi">تست عملکرد توکنایزیشن</div>

# In[9]:


sample = 'سلاااااام حالت خوب است؟'
print(f'Word Tokenizer: {word_tokenizer.encode(sample).tokens}')
print(f'BPE Tokenizer: {bpe_tokenizer.encode(sample).tokens}')


# # <div class="green">Preparing Data For LM</div>

# In[10]:


from torch.utils.data import Dataset

class TokenizedTextDataset(Dataset):
    def __init__(self, lines, tokenizer, tokenizer_args):                
        self.__lines = [tokenizer(line, **tokenizer_args) for line in lines]
        
    def __len__(self):
        return len(self.__lines)
    
    def __getitem__(self, idx):
        return self.__lines[idx]

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


# In[11]:


dataset = TextDataset(LM_TRAINING_DATA)
dataset[1]


# # <div class="green">Transformer LM</div>

# In[12]:


TRANSFORMER_EPOCHS = 300
MAX_LENGTH = 128
LENGTH_OF_EMBEDINGS = 120
NUMBER_OF_LAYERS = 3


# In[13]:


from transformers import (
    AutoModel,
    PreTrainedModel,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
#     LineByLineTextDataset, #REMOVE THIS
)


# In[14]:


def create_gpt_model(dataset: Dataset, tokenizer: Tokenizer) -> typing.Tuple[PreTrainedModel, Trainer]:
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=copy.deepcopy(tokenizer))
    fast_tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    
    prepared_dataset = dataset.get_tokenized(fast_tokenizer, 
                padding='max_length', truncation='longest_first', return_tensors="pt", max_length=MAX_LENGTH)
        
    config = GPT2Config(vocab_size=tokenizer.get_vocab_size(), n_layer=NUMBER_OF_LAYERS, n_embd=LENGTH_OF_EMBEDINGS, n_positions=MAX_LENGTH)
    model = GPT2LMHeadModel(config)
    
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=fast_tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir="./GPT2",
#         save_strategy='no',
        overwrite_output_dir=True,
        num_train_epochs=TRANSFORMER_EPOCHS,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=1_0000,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=prepared_dataset
    )
    return model, trainer

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
            outputs = word_model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / sum_len)


# ## <span class="blue">Word Level</span>

# In[15]:


word_model, trainer = create_gpt_model(dataset, word_tokenizer)
print(word_model.num_parameters())


# In[16]:


trainer.train()
word_model.save_pretrained('gpt2_word')


# In[7]:


# print(f"PPL IS: {calc_ppl(dataset, word_model, word_tokenizer)}")


# ## <span class="blue">BPE Level</span>

# In[18]:


bpe_model, trainer = create_gpt_model(dataset, bpe_tokenizer)
print(bpe_model.num_parameters())


# In[19]:


trainer.train()
bpe_model.save_pretrained('gpt2_bpe')


# In[20]:


# print(f"PPL IS: {calc_ppl(dataset, bpe_model, bpe_tokenizer)}")

