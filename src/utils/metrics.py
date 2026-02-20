import torch
import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score, accuracy_score, roc_curve, f1_score

#def dice_score_coefficient(y_pred, y_true, eps=1e-7):
#    """
#    Sørensen–Dice coefficient, also known as the Dice similarity index.
#    The dice score is defined as 1 - dice coefficient.
#    """
#    y_pred = y_pred.cpu()
#    y_true = y_true.cpu()
#    intersection = torch.sum(y_true * y_pred)
#    union = torch.sum(y_true) + torch.sum(y_pred)
#    dice = (2 * intersection + eps) / (union + eps)
#    return dice


def dice_score_coefficient(y_pred, y_true, threshold=None, eps=1e-7):
    """
    Sørensen–Dice coefficient, also known as the Dice similarity index.
    The dice score is defined as 1 - dice coefficient.
    """
    # Convert inputs to float for calculations
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)

    # Apply threshold if specified (binary Dice)
    if threshold is not None:
        y_pred = (y_pred > threshold).astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2 * intersection + eps) / (union + eps)
    return dice

def jaccard_index(y_pred, y_true, eps=1e-7):
    """
    Jaccard index, also known as the intersection-over-union.
    """
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    jaccard = (intersection + eps) / (union + eps)
    return jaccard

def visualize_predictions(x, y, y_pred):    
    x = x[0].permute(1, 2, 0).cpu().numpy()
    y = y[0].permute(1, 2, 0).cpu().numpy()
    y_pred = y_pred[0].permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(x, cmap='gray', interpolation='none')
    ax[1].imshow(x, cmap='gray')
    ax[1].imshow(y, cmap='jet', interpolation='none', alpha=0.5)
    ax[2].imshow(x, cmap='gray')
    ax[2].imshow(y_pred, cmap='jet', interpolation='none', alpha=0.5)

    return fig, ax

def show_confmat(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    return fig, ax

def show_reconstruction(y_true, y_pred):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
    ax[0].imshow(y_true[0, 0], cmap='gray')
    ax[0].set_title("Ground Truth")
    ax[1].imshow(y_pred[0, 0], cmap='gray')
    ax[1].set_title("MAE Prediction")
    return fig, ax

def compute_conceptwise_metrics(c_true, c_pred, selected_concepts=None, desc='test', dataset='breast_us'):
    """
    Evaluate accuracy and AUROC for each concept separately.

    Args:
        c_true (torch.Tensor): Ground-truth binary labels of shape (batch_size, num_categories).
        c_pred (torch.Tensor): Predicted probabilities or logits of shape (batch_size, num_categories).

    Returns:
        dict: A dictionary containing accuracy and AUROC for each concept.
              Format: {"accuracy": [acc1, acc2, ...], "auroc": [auroc1, auroc2, ...]}
    """
    # round c_true (pseudo) to 0 or 1
    c_pred = torch.from_numpy(c_pred)
    if isinstance(c_true, np.ndarray):
        c_true = (torch.from_numpy(c_true) > 0.5).int()
    elif isinstance(c_true, list):
        c_true = torch.tensor(c_true, dtype=torch.float32)
        c_true = (c_true > 0.5).int()

    num_categories = c_true.shape[1]
    accuracy_per_concept = []
    bal_accuracy_per_concept = []
    auroc_per_concept = []

    if dataset in ['BREAST_US', 'BrEaST', 'BUSBRA']:
        concept_names = {
            0: 'shadowing',
            1: 'enhancement',
            2: 'halo',
            3: 'calcifications',
            4: 'skin_thickening',
            5: 'circumscribed_margins',
            6: 'spiculated_margins',
            7: 'indistinct_margins',
            8: 'angular_margins',
            9: 'microlobulated_margins',
            10: 'regular_shape',
            11: 'echo_hyperechoic',
            12: 'echo_hypoechoic',
            13: 'echo_heterogeneous',
            14: 'echo_cystic'
        }
    
    if dataset == 'DDSM':
        concept_names = {
            0: 'regular_shape', 1: 'irregular_shape', 2: 'lobulated_shape',
            3: 'circumscribed_margins', 4: 'ill_defined_margins', 5: 'spiculated_margins',
            6: 'obscured_margins', 7: 'microlobulated_margins', 
            8: 'pleomorphic_calc', 9: 'amorphous_calc', 10: 'fine_linear_calc',
            11: 'branching_calc', 12: 'vascular_calc', 13: 'coarse_calc',
            14: 'punctate_calc', 15: 'lucent_calc', 16: 'eggshell_calc',
            17: 'round_calc', 18: 'regular_calc', 19: 'dystrophic_calc',
            20: 'clustered_calc_dist', 21: 'segmental_calc_dist', 22: 'linear_calc_dist',
            23: 'scattered_calc_dist', 24: 'regional_calc_dist',
            25: 'low_breast_density', 26: 'moderate_breast_density', 27: 'high_breast_density',
            28: 'architecture_distortion', 29: 'asymmetry', 30: 'lymph_node'
        }


    if dataset == 'CUB': 
        from src.utils.concept_bank.CUB import CUB_CONCEPT_BANK
        concept_names = {
            idx: f"{concept}"
            for idx, concept in enumerate(CUB_CONCEPT_BANK)
        }

    if selected_concepts is None:
        selected_concepts = [list(range(num_categories))] * num_categories

    # Evaluate metrics for each concept
    for i in range(num_categories):
        pred_concept = c_pred[:, i]
        label_concept = c_true[:, i]

        # Accuracy (convert predictions to binary)
        binary_preds = (pred_concept >= 0.5).float()
        accuracy = (binary_preds == label_concept).float().mean().item()

        fpr, tpr, t = roc_curve(label_concept, pred_concept)

        # select the threshold that yields the highest bal. accuracy
        bal_acc = [balanced_accuracy_score(label_concept, pred_concept > x) for x in t]
        max_bal_acc = max(bal_acc)
        closest_thresh = t[bal_acc.index(max_bal_acc)]
        print("Max balanced accuracy: {:.2f} at threshold {:.2f}".format(max_bal_acc, closest_thresh))

        bal_accuracy = balanced_accuracy_score(label_concept, binary_preds >= closest_thresh)
        accuracy_per_concept.append(accuracy)
        bal_accuracy_per_concept.append(bal_accuracy)

        # AUROC (handle cases where labels are not both 0 and 1)
        if len(torch.unique(label_concept)) > 1:
            auroc = roc_auc_score(label_concept.cpu().numpy(), pred_concept.cpu().numpy())
        else:
            auroc = float('nan')  # Cannot compute AUROC if only one class is present
        auroc_per_concept.append(auroc)

    

    # Return results as a dictionary
    metrics = {}
    for i in range(len(accuracy_per_concept)):
        selected_concept = concept_names[selected_concepts[i]]
        metrics[f'{selected_concept}_accuracy'] = accuracy_per_concept[i]
    for i in range(len(accuracy_per_concept)):
        selected_concept = concept_names[selected_concepts[i]]
        metrics[f'{selected_concept}_bal_accuracy'] = bal_accuracy_per_concept[i]
    for i in range(len(auroc_per_concept)):
        selected_concept = concept_names[selected_concepts[i]]
        metrics[f'{selected_concept}_auroc'] = auroc_per_concept[i]


    metrics['concept_auc'] = np.mean(auroc_per_concept)
    metrics['concept_acc'] = np.mean(accuracy_per_concept)
    metrics['concept_bacc'] = np.mean(bal_accuracy_per_concept)

    acc_only_6 = 0
    acc_imp_6 = 0
    clin_rel_cons = ['shadowing', 'spiculated_margins', 'angular_margins', 'echo_hyperechoic', 'regular_shape', 'circumscribed_margins']
    imp_weights = [1 if concept in clin_rel_cons else 0.5 for concept in concept_names.values()]
    for i in range(len(accuracy_per_concept)):
        if concept_names[i] in clin_rel_cons:
            acc_only_6 += accuracy_per_concept[i]
        acc_imp_6 += accuracy_per_concept[i] * imp_weights[i]
    
    acc_only_6 /= len(clin_rel_cons)
    acc_imp_6 /= len(imp_weights)
    metrics['concept_acc_only_6'] = acc_only_6
    metrics['concept_acc_imp_6'] = acc_imp_6

    return metrics

def compute_classification_metrics(y_true, y_prob, tune_threshold=True, multi_class=False):
    """
    y_prob: predicted probabilities of shape (batch_size, num_classes)
        example: y_prob = [0.6, 0.2, 
    y_true: ground truth labels of shape (batch_size,)
    """
    if multi_class:
        # Convert probabilities to predicted labels
        y_prob = F.softmax(torch.from_numpy(y_prob)).numpy()
        y_pred = np.argmax(y_prob, axis=1)
        
        if type(y_true) != list:
            y_true = y_true.flatten()

        print('debugging metrics function')
        print(y_pred)
        print(y_pred.shape)
        print(y_true)
        #print(y_true.shape)

        # AUROC: one-vs-rest
        try:
            auroc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError as e:
            print(f"Warning: AUROC calculation failed: {e}")
            auroc = None

        accuracy = accuracy_score(y_true, y_pred)
        bal_accuracy = balanced_accuracy_score(y_true, y_pred)

        return {
            'auc': auroc,
            'accuracy': accuracy,
            'bal_accuracy': bal_accuracy,
        }

    y_prob = F.sigmoid(torch.from_numpy(y_prob)).numpy()
    fpr, tpr, t = roc_curve(y_true, y_prob[:, 1])

    # select the threshold that yields the highest bal. accuracy
    bal_acc = [balanced_accuracy_score(y_true, y_prob[:, 1] > x) for x in t]
    max_bal_acc = max(bal_acc)
    closest_thresh = t[bal_acc.index(max_bal_acc)]

    thresh = 0.5 if not tune_threshold else closest_thresh

    # select the threshold that yields the specificity closest to 40%
    thresh40 = min(t, key=lambda x: abs(recall_score(y_true, y_prob[:, 1] > x, pos_label=0) - 0.6))
    sens_at_40 = recall_score(y_true, y_prob[:, 1] > thresh40, pos_label=1)

    # select the threshold that yields the specificity closest to 60%
    thresh60 = min(t, key=lambda x: abs(recall_score(y_true, y_prob[:, 1] > x, pos_label=0) - 0.4))
    sens_at_60 = recall_score(y_true, y_prob[:, 1] > thresh60, pos_label=1)

    # select the threshold that yields the specificity closest to 80%
    thresh80 = min(t, key=lambda x: abs(recall_score(y_true, y_prob[:, 1] > x, pos_label=0) - 0.2))
    sens_at_80 = recall_score(y_true, y_prob[:, 1] > thresh80, pos_label=1)

    auroc = roc_auc_score(y_true, y_prob[:, 1])
    bal_accuracy = balanced_accuracy_score(y_true, y_prob[:, 1] > closest_thresh)
    sensitivity = recall_score(y_true, y_prob[:, 1] > 0.5, pos_label=1)
    specificity = recall_score(y_true, y_prob[:, 1] > 0.5, pos_label=0)

    return {
        'auc': auroc,
        'bal_accuracy': bal_accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'sensitivity_at_40': sens_at_40,
        'sensitivity_at_60': sens_at_60,
        'sensitivity_at_80': sens_at_80,
    }

def compute_multiclass_metrics(b_true, b_prob, tune_threshold=True):
    """
    Compute multiclass classification metrics by using a one-vs-rest approach.
    """
    # Convert tensors to numpy if needed
    if hasattr(b_true, 'cpu'):
        b_true = b_true.cpu().numpy()
    if hasattr(b_prob, 'cpu'):
        b_prob = b_prob.cpu().numpy()

    metrics = {}
    num_classes = b_prob.shape[1]

    # Compute metrics for each class in a one-vs-rest fashion
    for i in range(num_classes):
        # Construct binary labels for class i
        y_true_bin = (b_true == i).astype(int)
        y_score_bin = b_prob[:, i]

        # ROC curve
        fpr, tpr, threshold_list = roc_curve(y_true_bin, y_score_bin)

        # Optional: tune threshold for best balanced accuracy
        if tune_threshold:
            bal_acc_list = [balanced_accuracy_score(y_true_bin, y_score_bin > thr) for thr in threshold_list]
            best_thr = threshold_list[bal_acc_list.index(max(bal_acc_list))]
        else:
            best_thr = 0.5

        # Compute AUROC
        auroc = roc_auc_score(y_true_bin, y_score_bin)

        # Balanced accuracy at best threshold vs. a default 0.5 threshold
        bal_acc = balanced_accuracy_score(y_true_bin, (y_score_bin > best_thr))
        bal_acc_05 = balanced_accuracy_score(y_true_bin, (y_score_bin > 0.5))

        # Store metrics
        metrics[f"birads_{i}_auroc"] = auroc
        metrics[f"birads_{i}_bal_accuracy"] = bal_acc
        metrics[f"birads_{i}_bal_accuracy_05"] = bal_acc_05

    return metrics