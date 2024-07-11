from transformers import BertTokenizer


def bert_tokenizer(pretrained_model_name_or_path):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                              truncation_side='right')
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    return tokenizer
