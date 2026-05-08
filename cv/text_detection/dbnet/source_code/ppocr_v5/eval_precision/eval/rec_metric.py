import argparse
import string
from pathlib import Path
from typing import Dict, List, Union
from tqdm import tqdm
import Levenshtein


class TextRecMetric:
    def __init__(self, is_filter=False, ignore_space=True, **kwargs):
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-15

    def __call__(self, pred_txt_path: Union[str, Path, None] = None):
        if pred_txt_path is None:
            raise ValueError("pred_txt_path can not be None.")

        data = read_txt(pred_txt_path)
        preds, gts, elapses = [], [], []
        for v in tqdm(data, desc="Evaluating images"):
            pred, gt, elapse = v.split("\t")
            preds.append(pred)
            gts.append(gt)
            elapses.append(float(elapse))

        result = self.get_metric(preds, gts)
        avg_elapse = sum(elapses) / len(elapses)
        result["avg_elapse"] = round(avg_elapse, 4)
        return result

    def get_metric(
        self, preds: Union[List[str], str], gts: Union[List[str], str]
    ) -> Dict[str, float]:
        if preds is None or gts is None:
            return ValueError("preds or gts can not be both None.")

        if isinstance(preds, str):
            preds = [preds]

        if isinstance(gts, str):
            gts = [gts]

        correct_num, all_num = 0, 0
        char_match_list = []
        for pred, target in zip(preds, gts):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")

            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)

            cur_distance = Levenshtein.distance(pred, target)
            char_match_one = 1 - cur_distance / (max(len(pred), len(target)) + self.eps)
            char_match_list.append(char_match_one)

            if pred == target:
                correct_num += 1

            all_num += 1

        exact_match = 1.0 * correct_num / (all_num + self.eps)
        char_match = sum(char_match_list) / len(char_match_list)
        return {
            "ExactMatch": round(exact_match, 4),
            "CharMatch": round(char_match, 4),
        }

    def _normalize_text(self, text):
        text = "".join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text)
        )
        return text.lower()


def read_txt(txt_path: Union[Path, str]) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        data = [v.rstrip("\n") for v in f]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred", "--pred_path", type=str, required=True)
    args = parser.parse_args()

    evaluator = TextRecMetric()
    metrics = evaluator(args.pred_path)
    print(metrics)


if __name__ == "__main__":
    main()