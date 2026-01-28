"""
Fairness Metrics 계산 모듈
Stage 2 Linear Probing 평가를 위한 공정성 메트릭

Metrics:
- F_FPR: False Positive Rate Fairness (서브그룹 간 FPR 차이)
- F_OAE: Overall Accuracy Equality (서브그룹 간 정확도 차이)
- F_DP: Demographic Parity (서브그룹 간 양성 예측률 차이)
- F_MEO: Maximum Equality of Opportunity (서브그룹 간 TPR/FPR/TNR/FNR 최대 차이)
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from collections import defaultdict


def compute_fairness_metrics(predictions, labels, subgroup_ids, attribute_name='subgroup'):
    """
    Fairness 메트릭 계산

    Args:
        predictions: numpy array of predicted probabilities [N]
        labels: numpy array of ground truth labels [N]
        subgroup_ids: numpy array of subgroup IDs [N]
        attribute_name: 속성 이름 (로깅용)

    Returns:
        dict: Fairness 메트릭 딕셔너리
    """
    unique_subgroups = np.unique(subgroup_ids)

    # 서브그룹별 메트릭 계산
    subgroup_metrics = {}

    for sg_id in unique_subgroups:
        if sg_id < 0:  # Unknown 제외
            continue

        mask = subgroup_ids == sg_id
        sg_preds = predictions[mask]
        sg_labels = labels[mask]

        if len(sg_labels) < 10:  # 최소 샘플 수 체크
            continue

        try:
            metrics = compute_classification_metrics(sg_labels, sg_preds)
            subgroup_metrics[sg_id] = metrics
        except Exception as e:
            print(f"Warning: Failed to compute metrics for {attribute_name}={sg_id}: {e}")
            continue

    if len(subgroup_metrics) < 2:
        return {'error': 'Not enough subgroups with valid metrics'}

    # Fairness 메트릭 계산
    fprs = [m['fpr'] for m in subgroup_metrics.values()]
    tprs = [m['tpr'] for m in subgroup_metrics.values()]
    fnrs = [m['fnr'] for m in subgroup_metrics.values()]
    tnrs = [m['tnr'] for m in subgroup_metrics.values()]
    accs = [m['acc'] for m in subgroup_metrics.values()]
    pos_rates = [m['positive_rate'] for m in subgroup_metrics.values()]
    neg_rates = [m['negative_rate'] for m in subgroup_metrics.values()]

    fairness_results = {
        'subgroup_metrics': subgroup_metrics,
        # F_FPR: 평균 FPR과의 차이 합
        'F_FPR': sum([abs(fpr - np.mean(fprs)) for fpr in fprs]) * 100,
        # F_FPR_maxmin: max-min 방식
        'F_FPR_maxmin': (max(fprs) - min(fprs)) * 100,
        # F_OAE: 정확도 최대-최소 차이
        'F_OAE': (max(accs) - min(accs)) * 100,
        # F_DP: Demographic Parity (양성/음성 예측률 차이)
        'F_DP': max(max(pos_rates) - min(pos_rates), max(neg_rates) - min(neg_rates)) * 100,
        # F_MEO: Maximum Equality of Opportunity
        'F_MEO': max(
            max(fprs) - min(fprs),
            max(fnrs) - min(fnrs),
            max(tnrs) - min(tnrs),
            max(tprs) - min(tprs)
        ) * 100,
        # 추가 통계
        'num_subgroups': len(subgroup_metrics),
        'avg_fpr': np.mean(fprs),
        'avg_acc': np.mean(accs),
    }

    return fairness_results


def compute_classification_metrics(labels, predictions, threshold=0.5):
    """
    분류 메트릭 계산

    Args:
        labels: Ground truth labels (0 or 1)
        predictions: Predicted probabilities (0~1)
        threshold: Classification threshold

    Returns:
        dict: 분류 메트릭 딕셔너리
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    pred_binary = (predictions >= threshold).astype(int)

    # Confusion Matrix
    try:
        CM = confusion_matrix(labels, pred_binary)
        TN = CM[0][0] if len(CM) > 0 else 0
        FN = CM[1][0] if len(CM) > 1 else 0
        TP = CM[1][1] if len(CM) > 1 else 0
        FP = CM[0][1] if len(CM) > 0 else 0
    except Exception:
        TN, FN, TP, FP = 0, 0, 0, 0

    # 메트릭 계산
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0

    positive_rate = (TP + FP) / len(labels) if len(labels) > 0 else 0
    negative_rate = (TN + FN) / len(labels) if len(labels) > 0 else 0

    try:
        auc = roc_auc_score(labels, predictions)
    except Exception:
        auc = 0.5

    try:
        acc = accuracy_score(labels, pred_binary)
    except Exception:
        acc = 0.0

    return {
        'auc': auc,
        'acc': acc,
        'fpr': fpr,
        'tpr': tpr,
        'fnr': fnr,
        'tnr': tnr,
        'positive_rate': positive_rate,
        'negative_rate': negative_rate,
        'num_samples': len(labels),
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
    }


def compute_eer(labels, predictions):
    """
    Equal Error Rate (EER) 계산

    Args:
        labels: Ground truth labels
        predictions: Predicted probabilities

    Returns:
        float: EER 값
    """
    from sklearn.metrics import roc_curve

    try:
        fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
        fnr = 1 - tpr
        diff = np.abs(fpr - fnr)
        min_idx = np.argmin(diff)
        eer = (fpr[min_idx] + fnr[min_idx]) / 2.0
        return eer
    except Exception:
        return 0.5


def print_fairness_report(fairness_results, attribute_name='subgroup'):
    """
    Fairness 결과 리포트 출력

    Args:
        fairness_results: compute_fairness_metrics의 결과
        attribute_name: 속성 이름
    """
    print("\n" + "=" * 60)
    print(f"FAIRNESS METRICS REPORT ({attribute_name})")
    print("=" * 60)

    if 'error' in fairness_results:
        print(f"Error: {fairness_results['error']}")
        return

    print(f"\nNumber of valid subgroups: {fairness_results['num_subgroups']}")

    # 서브그룹별 메트릭
    if 'subgroup_metrics' in fairness_results:
        print("\nSubgroup Performance:")
        print("-" * 60)
        print(f"{'Subgroup':<12} {'Samples':>8} {'AUC':>8} {'ACC':>8} {'FPR':>8} {'TPR':>8}")
        print("-" * 60)
        for sg_id, metrics in fairness_results['subgroup_metrics'].items():
            print(f"{sg_id:<12} {metrics['num_samples']:>8} "
                  f"{metrics['auc']:>8.4f} {metrics['acc']:>8.4f} "
                  f"{metrics['fpr']:>8.4f} {metrics['tpr']:>8.4f}")

    # Fairness 메트릭
    print("\nFairness Metrics:")
    print("-" * 60)
    print(f"F_FPR (Paper):  {fairness_results.get('F_FPR', 0):>8.3f}%")
    print(f"F_FPR (MaxMin): {fairness_results.get('F_FPR_maxmin', 0):>8.3f}%")
    print(f"F_OAE:          {fairness_results.get('F_OAE', 0):>8.3f}%")
    print(f"F_DP:           {fairness_results.get('F_DP', 0):>8.3f}%")
    print(f"F_MEO:          {fairness_results.get('F_MEO', 0):>8.3f}%")

    print("\nOverall Statistics:")
    print(f"Average FPR: {fairness_results.get('avg_fpr', 0):.4f}")
    print(f"Average ACC: {fairness_results.get('avg_acc', 0):.4f}")
    print("=" * 60)


def compute_gender_race_fairness(predictions, labels, genders, races):
    """
    Gender와 Race 각각에 대한 Fairness 계산

    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        genders: Gender labels (0: Male, 1: Female)
        races: Race labels (0: Asian, 1: Black, 2: White, 3: Other)

    Returns:
        dict: Gender 및 Race 별 Fairness 메트릭
    """
    results = {}

    # Gender Fairness
    gender_fairness = compute_fairness_metrics(predictions, labels, genders, 'gender')
    results['gender'] = gender_fairness

    # Race Fairness
    race_fairness = compute_fairness_metrics(predictions, labels, races, 'race')
    results['race'] = race_fairness

    # Intersection Fairness (Gender x Race)
    intersection_ids = genders * 4 + races
    intersection_fairness = compute_fairness_metrics(predictions, labels, intersection_ids, 'intersection')
    results['intersection'] = intersection_fairness

    return results
