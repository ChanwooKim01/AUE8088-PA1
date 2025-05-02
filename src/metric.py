from torchmetrics import Metric
import torch
import src.config as cfg

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("TP", default=torch.zeros(cfg.NUM_CLASSES), dist_reduce_fx="sum")
        self.add_state("FP", default=torch.zeros(cfg.NUM_CLASSES), dist_reduce_fx="sum")
        self.add_state("FN", default=torch.zeros(cfg.NUM_CLASSES), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)

        assert preds.shape == target.shape, \
        f'preds and target do not have equal shape! \npred shape: {preds.shape} \ntarget shape: {target.shape}'

        for cls_ in range(cfg.NUM_CLASSES):
            pred_cls    = (preds == cls_)
            target_cls  = (target == cls_)

            self.TP[cls_] += torch.sum(pred_cls & target_cls)
            self.FP[cls_] += torch.sum(pred_cls & (~target_cls))
            self.FN[cls_] += torch.sum((~pred_cls) & target_cls)

    def compute(self):
        f1_scores   = torch.zeros(cfg.NUM_CLASSES)
        precision   = torch.zeros(cfg.NUM_CLASSES)
        recall      = torch.zeros(cfg.NUM_CLASSES)
        
        for cls_ in range(cfg.NUM_CLASSES):
            if (self.TP[cls_] + self.FP[cls_]) > 0:
                precision[cls_] = self.TP[cls_] / (self.TP[cls_] + self.FP[cls_])
            else:
                precision[cls_] = 0

            if (self.TP[cls_] + self.FN[cls_]) > 0:
                recall[cls_] = self.TP[cls_] / (self.TP[cls_] + self.FN[cls_])
            else:
                recall[cls_] = 0
            
            if (precision[cls_] + recall[cls_]) > 0:
                f1_scores[cls_] = 2 * precision[cls_] * recall[cls_] / (precision[cls_] + recall[cls_])
            else:
                f1_scores[cls_] = 0  
        
        return f1_scores
    
    
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        assert (preds.shape == target.shape), \
        f'preds and target do not have equal shape! \npred shape: {preds.shape} \ntarget shape:{target.shape}'

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
