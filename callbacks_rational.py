from torchmetrics import Metric
from torchmetrics.classification import BinaryRecall, BinarySpecificity
import torch
from callbacks import BinaryACC_Callback, BinaryAUC_Callback, EER_Callback


class BinaryFPR(Metric):
    def __init__(self, threshold=0.0):  # 0.0 for centered logits
        super().__init__()
        self.threshold = threshold
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, preds, target):
        p = (preds >= self.threshold).int()
        self.fp += ((p == 1) & (target == 0)).sum()
        self.tn += ((p == 0) & (target == 0)).sum()
    def compute(self): return self.fp.float() / (self.fp + self.tn + 1e-8)

class BinaryFNR(Metric):
    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, preds, target):
        p = (preds >= self.threshold).int()
        self.fn += ((p == 0) & (target == 1)).sum()
        self.tp += ((p == 1) & (target == 1)).sum()
    def compute(self): return self.fn.float() / (self.fn + self.tp + 1e-8)

class TPR_Callback(BinaryACC_Callback):
    @property
    def metric_name(self): return "tpr"
    def build_metric_funcs(self, *args, **kwargs): return BinaryRecall(threshold=0.0)

class TNR_Callback(BinaryACC_Callback):
    @property
    def metric_name(self): return "tnr"
    def build_metric_funcs(self, *args, **kwargs): return BinarySpecificity(threshold=0.0)

class FPR_Callback(BinaryACC_Callback):
    @property
    def metric_name(self): return "fpr"
    def build_metric_funcs(self, *args, **kwargs): return BinaryFPR(threshold=0.0)

class FNR_Callback(BinaryACC_Callback):
    @property
    def metric_name(self): return "fnr"
    def build_metric_funcs(self, *args, **kwargs): return BinaryFNR(threshold=0.0)