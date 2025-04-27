from torchmetrics import Metric
import torch
import src.config as cfg

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("TP", default=torch.zeros(cfg.NUM_CLASSES), dist_reduce_fx="sum")
        self.add_state("TN", default=torch.zeros(cfg.NUM_CLASSES), dist_reduce_fx="sum")
        self.add_state("FP", default=torch.zeros(cfg.NUM_CLASSES), dist_reduce_fx="sum")
        self.add_state("FN", default=torch.zeros(cfg.NUM_CLASSES), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)

        assert preds.shape == target.shape, \
        f'preds and target do not have equal shape! \npred shape: {preds.shape} \ntarget shape: {target.shape}'

        for cls in range(cfg.NUM_CLASSES):
            pred_cls    = (preds == cls)
            target_cls  = (target == cls)

            self.TP[cls] += torch.sum(pred_cls & target_cls)
            self.FP[cls] += torch.sum(pred_cls & (~target_cls))
            self.FN[cls] += torch.sum((~pred_cls) & target_cls)

    def compute(self):
        f1_scores   = torch.zeros(cfg.NUM_CLASSES)
        precision   = torch.zeros(cfg.NUM_CLASSES)
        recall      = torch.zeros(cfg.NUM_CLASSES)
        
        for cls in range(cfg.NUM_CLASSES):
            precision[cls] = self.TP[cls] / (self.TP[cls] + self.FP[cls])
            recall[cls] = self.TP[cls] / (self.TP[cls] + self.FN[cls])
            f1_scores[cls] = (2 * precision[cls] * recall[cls]) / (precision[cls] + recall[cls])   
        
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
        f'preds and target does not have equal shape! \npred shape: {preds.shape} \ntarget shape:{target.shape}'

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
