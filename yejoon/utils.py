import torch

class ConfusionMatrix:
    def __init__(self, confusion_matrix):
        '''confusion_matrix[i][j] = real label is i, predicted label is j
        '''
        # length of matrix (matrix need to be square)
        length = confusion_matrix.shape[0]

        # accuracy
        self.accuracy = (confusion_matrix.diag().sum() / confusion_matrix.sum()).item()
        
        # precisions, recalls, f1s
        precisions = torch.zeros(length, dtype=torch.float32)
        recalls = torch.zeros(length, dtype=torch.float32)
        f1s = torch.zeros(length, dtype=torch.float32)

        for i in range(length):
            precision = confusion_matrix[i][i] / confusion_matrix[:,i].sum()
            recall = confusion_matrix[i][i] / confusion_matrix[i,:].sum()

            precisions[i] = precision
            recalls[i] = recall
            f1s[i] = 2 * precision * recall / (precision + recall)

        # macro precision, recall, f1
        self.precision = precisions.mean().item()
        self.recall = recalls.mean().item()
        self.f1 = f1s.mean().item()