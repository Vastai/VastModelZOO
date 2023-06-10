import argparse
import json
import os

import evaluate
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer


def tokenizer_preprocess(examples):
    tokenized_examples = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=seq_length, padding='max_length', return_tensors='pt')
    tokenized_examples['label'] = examples['label']
    return tokenized_examples


def get_tpfn(pre:int, label:int, tpfn: list):
    if pre == 1 and label == 1:
        tpfn[0] += 1
    elif pre == 1 and label == 0:
        tpfn[1] += 1
    elif pre == 0 and label == 1:
        tpfn[2] += 1
    else:
        tpfn[3] += 1
        
def get_f1(tpfn: list):
    precision = tpfn[0] / (tpfn[0] + tpfn[1] + 1E-6)
    recall = tpfn[0] / (tpfn[0] + tpfn[2] + 1E-6)
    f1 = 2. * precision * recall / (precision + recall + 1E-6)
    acc = ( tpfn[0] +  tpfn[3]) / ( tpfn[0] +  tpfn[1] +  tpfn[2] +  tpfn[3])
    return f1, acc
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    tpfn = [0., 0., 0., 0.]
    for pre, lbl in zip(predictions, labels):
      get_tpfn(pre, lbl, tpfn)
    
    f1, acc = get_f1(tpfn)
    # score = accuracy_metric.compute(predictions=predictions, references=labels)
    # return score
    return {
              "eval_f1": f1,
              "eval_accuracy": acc
           }


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="FINETUNE MODEL")
    parse.add_argument(
        "--data_name",
        type=str,
        default="mrpc",
        help="the name of the task to execute.",
    )
    parse.add_argument(
        "--model_name_or_path", 
        type=str,
        default="/home/jies/code/nlp/transformers/runs/bert_test/checkpoint-2300",
        help="pretrain model path"
    )
    parse.add_argument(
        "--num_labels", 
        type=int, 
        default=2, 
        help="class num"
    )
    parse.add_argument(
        "--seq_length", 
        type=int, 
        default=128, 
        help="max sequence length "
    )
    parse.add_argument(
        "--output_dir",
        type=str,
        default="/home/jies/code/nlp/transformers/runs/bert_test",
        help="finetune model result dir.",
    )
    parse.add_argument(
        "--do_train",
        action='store',
        help="whether train model.",
    )
    parse.add_argument(
        "--do_eval",
        action='store',
        help="whether eval model.",
    )
    parse.add_argument(
        "--do_predict",
        action='store',
        help="whether predict.",
    )
    args = parse.parse_args()
    
    output_dir = args.output_dir
    model_name_or_path = args.model_name_or_path  # model_name 
    num_labels = args.num_labels
    seq_length = args.seq_length

    # load datasets
    datasets = load_dataset('glue', args.data_name)
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    # build the data set through the batch process method
    datasets = datasets.map(tokenizer_preprocess, batched=True)

    # build the accuracy evaluation method
    accuracy_metric = evaluate.load("accuracy")

    # load seq cls pretrain model
    model = BertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)

    # create train args
    args = TrainingArguments(
                            learning_rate=2e-5,
                            per_device_train_batch_size=8,
                            per_device_eval_batch_size=128,
                            num_train_epochs=20,
                            weight_decay=0.01,
                            output_dir=output_dir,
                            logging_steps=10,
                            evaluation_strategy = "epoch",
                            save_strategy = "epoch",
                            load_best_model_at_end=True,
                            metric_for_best_model="accuracy",
                            fp16=True,
                            )
    # create trainner
    trainer = Trainer(
                    model,
                    args=args,
                    train_dataset=datasets["train"],
                    eval_dataset=datasets["validation"],
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
                    )

    # train
    if args.do_train:
      out = trainer.train()
      
    # eval
    if args.do_eval:
      precision = trainer.evaluate()
      with open(os.path.join(output_dir,'eval.json'), 'w') as json_file:
          json_file.write(json.dumps(precision, ensure_ascii=False, indent=4))

    if args.do_predict:
      # predict and write results
      result = trainer.predict(datasets["test"])
      predictions = result.predictions
      predictions = predictions.argmax(axis=-1)
      label_ids = result.label_ids
      
      with open(os.path.join(output_dir, 'predict.txt'), 'w') as fw:
        for pre, lab in zip(predictions, label_ids):
          fw.write(str(pre) + ' ' + str(lab) + '\n')