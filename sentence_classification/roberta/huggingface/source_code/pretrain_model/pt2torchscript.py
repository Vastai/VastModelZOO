import os
import argparse
from packaging import version

import torch
import torch.nn as nn
from transformers import BertForTokenClassification


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

        self.seq_length = config.max_seq_length

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        # seq_length = input_shape[1]
        seq_length = self.seq_length
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    
def modify_bert_embeddings(model, max_seq=128):
    model.config.max_seq_length = max_seq
    embeddings_new = BertEmbeddings(model.config)
    embeddings_new.word_embeddings.weight = model.bert.embeddings.word_embeddings.weight
    embeddings_new.position_embeddings.weight = model.bert.embeddings.position_embeddings.weight
    embeddings_new.token_type_embeddings.weight = model.bert.embeddings.token_type_embeddings.weight
    embeddings_new.LayerNorm.weight = model.bert.embeddings.LayerNorm.weight
    embeddings_new.dropout = model.bert.embeddings.dropout
    embeddings_new.position_embedding_type = model.bert.embeddings.position_embedding_type
    model.bert.embeddings = embeddings_new
    
    return model


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="FINETUNE MODEL")
    parse.add_argument(
        "--model_name_or_path", 
        type=str,
        default="/home/jies/code/nlp/transformers/runs/bert_test/checkpoint-2300",
        help="pretrain model path"
    )
    parse.add_argument(
        "--seq_length", 
        type=int, 
        default=128, 
        help="max sequence length "
    )
    parse.add_argument(
        "--save_path",
        type=str,
        default="/home/jies/code/nlp/transformers/runs/bert_test/checkpoint-2300/cls_bert_base.torchscript.pt",
        help="finetune model result dir.",
    )
    args = parse.parse_args()
    
    model_name_or_path = args.model_name_or_path
    seq_length = args.seq_length
    
    model = BertForTokenClassification.from_pretrained(model_name_or_path, return_dict=False)
    model = modify_bert_embeddings(model, seq_length)
    input = torch.randint(0, 2, (1, seq_length), dtype=torch.long)
    out = model(input, input, input)
    print(out[0].shape)
    
    model.eval()
    scripted_model = torch.jit.trace(model, (input, input, input) , strict=False).eval()
    torch.jit.save(scripted_model, args.save_path)



