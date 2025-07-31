# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
# from modelscope.utils.hub import read_config
from modelscope.metainfo import Metrics


model_id = 'damo/nlp_structbert_sentiment-classification_chinese-base'
dataset_id = 'jd'

WORK_DIR = 'workspace'

max_epochs = 2
def cfg_modify_fn(cfg):
    cfg.train.max_epochs = max_epochs
    cfg.train.hooks = [{
            'type': 'TextLoggerHook',
            'interval': 100
        }, {
            "type": "CheckpointHook",
            "interval": 1
        }]
    cfg.evaluation.metrics = [Metrics.seq_cls_metric]
    cfg['dataset'] = {
        'train': {
            'labels': ['负面', '正面'],
            'first_sequence': 'sentence',
            'label': 'label',
        }
    }
    cfg.train.optimizer.lr = 3e-5
    return cfg


eval_dataset = MsDataset.load(dataset_id, namespace='DAMO_NLP', split='validation').to_hf_dataset()
eval_dataset = eval_dataset.filter(lambda x: x["label"] != None and x["sentence"] != None)

# map float to index
def map_labels(examples):
    map_dict = {0: "负面", 1: "正面"}
    examples['label'] = map_dict[int(examples['label'])]
    return examples

# train_dataset = train_dataset.map(map_labels)
eval_dataset = eval_dataset.map(map_labels)

kwargs = dict(
    model=model_id,
    # train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=WORK_DIR,
    cfg_modify_fn=cfg_modify_fn)


trainer = build_trainer(name='nlp-base-trainer', default_args=kwargs)
eval_results = trainer.evaluate()
print(eval_results)


print('===============================================================')
print('evaluate success')
print('===============================================================')



#######################################
# {'accuracy': 0.9200080192461909, 'binary-f1': 0.9185548071034906, 'f1': 0.9185548071034906}
# ===============================================================
# evaluate success
# ===============================================================