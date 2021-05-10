
from typing import Optional, List

import torch
from allennlp.training.metrics import Metric


@Metric.register("siamese_measure")
class SiameseMeasure(Metric):

    def __init__(self, vocab) -> None:
        self._correct_cnt = []
        self._total_cnt = []
        self._voacb = vocab

        self.sample_num = {}
        self.sample_vote = {}

        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0

    def __call__(self,
                 predictions: torch.Tensor,
                 pair_labels: torch.Tensor,
                 meta_eval: List = None,
                 mask: Optional[torch.Tensor] = None):
        predictions, pair_labels, mask = self.unwrap_to_tensors(predictions, pair_labels, mask)
        num_classes = predictions.size(-1)
        predictions = predictions.view((-1, num_classes))
        predictions = predictions.max(-1)[1].unsqueeze(-1)

        same_label = self._voacb.get_token_index("same", "labels")
        diff_label = self._voacb.get_token_index("diff", "labels")
        ff_label = self._voacb.get_token_index("feature@feature", "label_tags")
        fo_label = self._voacb.get_token_index("feature@other", "label_tags")
        of_label = self._voacb.get_token_index("other@feature", "label_tags")
        oo_label = self._voacb.get_token_index("other@other", "label_tags")

        if meta_eval == None:
            for p, pair_label in zip(predictions, pair_labels):
                if same_label == p:
                    if ff_label == pair_label:
                        self.true_positive += 1
                    if oo_label == pair_label:
                        self.true_negative += 1
                    if fo_label == pair_label:
                        self.false_positive += 1
                    if of_label == pair_label:
                        self.false_negative += 1
                if diff_label == p:
                    if fo_label == pair_label:
                        self.true_negative += 1
                    if of_label == pair_label:
                        self.true_positive += 1
                    if ff_label == pair_label:
                        self.false_negative += 1
                    if oo_label == pair_label:
                        self.false_positive += 1
        else:
            iter_number = int(meta_eval[0]["num"])
            for p, pair_label, meta in zip(predictions, pair_labels, meta_eval):

                d_id = meta["dialog2_id"]

                if self.sample_num.get(d_id) == None:
                    self.sample_num[d_id] = 0
                    self.sample_vote[d_id] = [0, 0, 0, 0]  # TP,TN,FP,FN

                self.sample_num[d_id] += 1

                if same_label == p:
                    if ff_label == pair_label:
                        self.sample_vote[d_id][0] += 1
                    if oo_label == pair_label:
                        self.sample_vote[d_id][1] += 1
                    if fo_label == pair_label:
                        self.sample_vote[d_id][2] += 1
                    if of_label == pair_label:
                        self.sample_vote[d_id][3] += 1
                if diff_label == p:
                    if fo_label == pair_label:
                        self.sample_vote[d_id][1] += 1
                    if of_label == pair_label:
                        self.sample_vote[d_id][0] += 1
                    if ff_label == pair_label:
                        self.sample_vote[d_id][3] += 1
                    if oo_label == pair_label:
                        self.sample_vote[d_id][2] += 1

                if self.sample_num[d_id] == iter_number * 2 + 1:

                    if self.sample_vote[d_id][0] > iter_number:
                        self.true_positive += 1
                        #print("======FR_ID:", d_id)
                    elif self.sample_vote[d_id][1] > iter_number:
                        self.true_negative += 1

                    elif self.sample_vote[d_id][2] > iter_number:
                        self.false_positive += 1
                        #print("------FALSE_FR_ID:", d_id)
                    else:
                        self.false_negative += 1
                        #print("++++++FALSE_NONFR_ID:", d_id)


    def get_metric(self, reset: bool):
        precision = self.true_positive * 1.0 / (self.true_positive + self.false_positive + 1e-6)
        recall = self.true_positive * 1.0 / (self.true_positive + self.false_negative + 1e-6)
        fmeasure = (2.0 * precision * recall) / (precision + recall + 1e-6)
        # 每一次训练都reset
        if reset:
            self.reset()

        return precision, recall, fmeasure

    def reset(self) -> None:
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0

        self.sample_num = {}
        self.sample_vote = {}

