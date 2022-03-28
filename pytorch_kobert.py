import os
from zipfile import ZipFile
import torch
from transformers import BertModel
import gluonnlp as nlp


def get_pytorch_kobert_model(ctx="cpu"):
    def get_kobert_model(model_path, vocab_file, ctx="cpu"):
        bertmodel = BertModel.from_pretrained(model_path, return_dict=False)
        device = torch.device(ctx)
        bertmodel.to(device)
        bertmodel.eval()
        vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(
            vocab_file, padding_token="[PAD]"
        )
        return bertmodel, vocab_b_obj

    model_path = './model_weight/kobert_from_pretrained'
    vocab_path = './model_weight/kobert_news_wiki_ko_cased-1087f8699e.spiece'
    return get_kobert_model(model_path, vocab_path, ctx)


def get_tokenizer():
    model_path = './model_weight/kobert_news_wiki_ko_cased-1087f8699e.spiece'
    return model_path
