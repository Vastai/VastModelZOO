import os
import numpy as np
import argparse


def get_predcit(pre_dir):
    file_names = os.listdir(pre_dir)
    file_names.sort()
    predict_dict = {}
    for i, name in enumerate(file_names):
        path = os.path.join(pre_dir, name)
        npz_data = np.load(path)
        predict = npz_data['output_0'].squeeze(0)
        predict_dict[i] = predict
    
    return predict_dict


def evaluate(
    predicts, 
    eval_path: str, 
    output_dir: str = 'output', 
    **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    fr = open(eval_path, 'r')
    contents = fr.readlines()[1:]
    label_dict = {}

    s = ''
    for index, line in enumerate(contents):
        label_dict[index] = int(line.strip().split('\t')[0])
    
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

    tpfn = [0., 0., 0., 0.]
    for key, value in predicts.items():
        predict = value
        if predict[0] < predict[1]:
            result = 1
        else:
            result = 0
        s += str(key) + ' Reference: ' + str(label_dict[key]) + ' Predicted: ' + str(result)
        get_tpfn(result, label_dict[key], tpfn)
        s += ' results_0: ' + str(predict[0]) + ' results_1: ' + str(predict[1]) + ' right: ' + str(int(tpfn[0] + tpfn[3])) + ' wrong: ' + str(int(tpfn[1] + tpfn[2])) + '\n'

    f1, acc = get_f1(tpfn)

    print("F1: ", round(f1, 4), "Accuracy:", round(acc, 4))
    s += "F1: " + str(round(f1, 5)) + " Accuracy: " + str(round(acc, 5))
    fw = open(os.path.join(output_dir, 'evaluate.txt'), 'w')
    fw.write(s)
    fw.close()
    fr.close()
        
        
if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="MPRC EVAL")
    parse.add_argument(
        "--result_dir",
        type=str,
        default="./out_result",
        help="vamp output *.npz results path",
    )
    parse.add_argument(
        "--eval_path", 
        type=str,
        default="./datasets/MRPC/dev.tsv",
        help="MRPC-dev file path "
    )
    args = parse.parse_args()

    evaluate(get_predcit(args.result_dir), args.eval_path)
    