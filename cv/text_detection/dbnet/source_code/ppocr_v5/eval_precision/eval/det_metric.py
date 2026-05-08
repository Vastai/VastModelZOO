import argparse
import ast
import copy
import json
from tqdm import tqdm
from typing import Dict, List, Tuple

from shapely.geometry import Polygon


class TextDetMetric:
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def __call__(self, pred_txt_path: str):
        preds, gts, elapses = self.read_pred_txt(pred_txt_path)

        results = []
        for gt, pred in tqdm(zip(gts, preds), total=len(gts), desc="Evaluating images"):
            results.append(self.evaluate_image(gt, pred))

        avg_elapse = sum(elapses) / len(elapses)
        metrics = self.combine_results(results)
        metrics["avg_elapse"] = round(avg_elapse, 4)
        return metrics

    def read_pred_txt(self, txt_path: str) -> Tuple[List, List]:
        pred_boxes, gt_boxes, elapses = [], [], []
        datas = self.read_txt(txt_path)
        for data in datas:
            cur_pred_boxes, cur_gt_boxes, elapse = data.split("\t")
            cur_pred_boxes = ast.literal_eval(cur_pred_boxes)
            cur_pred_boxes = [
                {"points": p, "text": "", "ignore": False} for p in cur_pred_boxes
            ]
            pred_boxes.append(cur_pred_boxes)

            cur_gt_boxes = ast.literal_eval(cur_gt_boxes)
            tmp_gt_boxes = copy.deepcopy(cur_gt_boxes)
            for i, points in enumerate(tmp_gt_boxes):
                if len(points) == 2:
                    x0, y0 = points[0]
                    x1, y1 = points[1]

                    points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                cur_gt_boxes[i] = points

            cur_gt_boxes = [
                {"points": p, "text": "", "ignore": False} for p in cur_gt_boxes
            ]
            gt_boxes.append(cur_gt_boxes)

            elapses.append(float(elapse))
        return pred_boxes, gt_boxes, elapses

    @staticmethod
    def parse_single_gt(json_path: str) -> List[Dict]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = []
        for v in data["shapes"]:
            if v["shape_type"] == "rectangle":
                x0, y0 = v["points"][0]
                x1, y1 = v["points"][1]
                v["points"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

            dict_tmp = {
                "points": v["points"],
                "text": v["label"],
                "ignore": False,
            }
            result.append(dict_tmp)
        return result

    def evaluate_image(self, gts, preds):
        gtPols, gtPolPoints, gtDontCarePolsNum = self.filter_gts(gts)

        filter_pred_res = self.filter_preds(preds, gtDontCarePolsNum, gtPols)
        detPols, detPolPoints, detDontCarePolsNum = filter_pred_res

        pairs = []
        detMatched = 0
        if len(gtPols) > 0 and len(detPols) > 0:
            for gtNum, pG in enumerate(gtPols):
                for detNum, pD in enumerate(detPols):
                    if gtNum in gtDontCarePolsNum or detNum in detDontCarePolsNum:
                        continue

                    iou = self.get_intersection_over_union(pD, pG)
                    if iou > self.iou_constraint:
                        detMatched += 1
                        pairs.append({"gt": gtNum, "det": detNum})

        recall, precision, hmean = 0, 0, 0
        numGtCare = len(gtPols) - len(gtDontCarePolsNum)
        numDetCare = len(detPols) - len(detDontCarePolsNum)
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare

        hmean = (
            0
            if (precision + recall) == 0
            else 2.0 * precision * recall / (precision + recall)
        )

        perSampleMetrics = {
            "precision": precision,
            "recall": recall,
            "hmean": hmean,
            "pairs": pairs,
            "gtPolPoints": gtPolPoints,
            "detPolPoints": detPolPoints,
            "gtCare": numGtCare,
            "detCare": numDetCare,
            "gtDontCare": gtDontCarePolsNum,
            "detDontCare": detDontCarePolsNum,
            "detMatched": detMatched,
        }
        return perSampleMetrics

    def filter_gts(self, gts: List[Dict]) -> Tuple[List, List, List]:
        gtPols, gtPolPoints, gtDontCarePolsNum = [], [], []
        for gt in gts:
            points = gt["points"]
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPols.append(points)
            gtPolPoints.append(points)
            if gt["ignore"]:
                gtDontCarePolsNum.append(len(gtPols) - 1)
        return gtPols, gtPolPoints, gtDontCarePolsNum

    def filter_preds(
        self, preds: List[Dict], gtDontCarePolsNum: List, gtPols: List
    ) -> Tuple[List, List, List]:
        detPols, detPolPoints, detDontCarePolsNum = [], [], []
        for pred in preds:
            points = pred["points"]
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)

            if not gtDontCarePolsNum:
                continue

            for dontCarePol in gtDontCarePolsNum:
                dontCarePol = gtPols[dontCarePol]
                intersected_area = self.get_intersection(dontCarePol, detPol)
                pdDimensions = Polygon(detPol).area

                precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                if precision > self.area_precision_constraint:
                    detDontCarePolsNum.append(len(detPols) - 1)
                    break
        return detPols, detPolPoints, detDontCarePolsNum

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result["gtCare"]
            numGlobalCareDet += result["detCare"]
            matchedSum += result["detMatched"]

        methodRecall = (
            0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
        )
        methodPrecision = (
            0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
        )
        methodHmean = (
            0
            if methodRecall + methodPrecision == 0
            else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
        )

        methodMetrics = {
            "precision": round(methodPrecision, 4),
            "recall": round(methodRecall, 4),
            "hmean": round(methodHmean, 4),
        }
        return methodMetrics

    @staticmethod
    def get_intersection(pD, pG):
        return Polygon(pD).intersection(Polygon(pG)).area

    @staticmethod
    def get_union(pD, pG):
        return Polygon(pD).union(Polygon(pG)).area

    def get_intersection_over_union(self, pD, pG):
        return self.get_intersection(pD, pG) / self.get_union(pD, pG)

    @staticmethod
    def read_txt(txt_path: str) -> List[str]:
        with open(txt_path, "r", encoding="utf-8") as f:
            data = f.readlines()
        data = list(map(lambda x: x.rstrip("\n"), data))
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred", "--pred_path", type=str, required=True)
    args = parser.parse_args()

    evaluator = TextDetMetric()
    metrics = evaluator(args.pred_path)
    print(metrics)


if __name__ == "__main__":
    main()

'''
from: https://github.com/SWHL/TextDetMetric/blob/main/text_det_metric/main.py
'''