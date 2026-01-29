"""
Intersection Fairness Metrics 계산 모듈
Fairness-Generalization 프로젝트 방식과 유사한 교차 그룹 분석 구현

주요 기능:
1. Intersection 그룹 생성 (예: male,asian, female,white)
2. Inter-group과 Intra-group fairness 동시 계산
3. 기존 fairness_metrics_advanced와 호환 가능한 인터페이스
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score
from collections import defaultdict
import itertools


def classification_metrics(label, prediction):
    """
    기본 분류 메트릭 계산

    Args:
        label: Ground truth labels (0 or 1)
        prediction: Predicted probabilities (0~1)

    Returns:
        tuple: 다양한 메트릭들
    """
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    auc = roc_auc_score(label, prediction)
    acc = accuracy_score(label, prediction >= 0.5)
    precision = precision_score(label, prediction >= 0.5, zero_division=0)

    CM = confusion_matrix(label, prediction >= 0.5)
    TN = CM[0][0] if len(CM) > 0 else 0
    FN = CM[1][0] if len(CM) > 1 else 0
    TP = CM[1][1] if len(CM) > 1 else 0
    FP = CM[0][1] if len(CM) > 0 else 0

    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

    f_g = (TP + FP) / len(label) if len(label) > 0 else 0
    positive_rate = (TP + FP) / len(label) if len(label) > 0 else 0
    negative_rate = (TN + FN) / len(label) if len(label) > 0 else 0

    er = 1 - ((TPR + (1 - FPR)) / 2)

    return auc, er, FPR, TPR, acc, precision, f_g, eer, positive_rate, negative_rate, FNR, TNR


def compute_fairness_metrics_intersection(predictions_dict, labels_dict, attribute_groups):
    """
    Intersection Fairness Metrics 계산 (교차 그룹 분석 포함)

    Args:
        predictions_dict: dict[key] -> np.array of predictions
                         예: {'gender_male': [...], 'gender_female': [...],
                              'race_asian': [...], 'race_white': [...]}
        labels_dict: dict[key] -> np.array of labels
        attribute_groups: list of attribute group names
                         예: ['gender', 'race']

    Returns:
        dict: {
            'single_group_metrics': {...},      # 단일 서브그룹 메트릭
            'intersection_metrics': {...},      # 교차 그룹 메트릭
            'group_overall_metrics': {...},     # 그룹별 전체 메트릭
            'intra_fairness': {...},           # Intra-group fairness
            'inter_fairness': {...},           # Inter-group fairness
            'overall_fairness': {...},         # 종합 fairness 메트릭
            'skipped_groups': {...}            # 계산 실패한 그룹 정보
        }
    """

    # Step 1: 단일 서브그룹 메트릭 계산 (기존과 동일)
    single_metrics = {}
    skipped_groups = {}  # 계산 실패한 그룹과 이유 저장

    for key in predictions_dict:
        if len(labels_dict[key]) == 0:
            continue

        try:
            auc, er, fpr, tpr, acc, precision, f_g, eer, pr, nr, fnr, tnr = classification_metrics(
                labels_dict[key], predictions_dict[key]
            )

            single_metrics[key] = {
                'auc': auc,
                'error_rate': er,
                'FPR': fpr,
                'TPR': tpr,
                'acc': acc,
                'precision': precision,
                'F_G': f_g,
                'eer': eer,
                'positive_rate': pr,
                'negative_rate': nr,
                'FNR': fnr,
                'TNR': tnr,
                'num_samples': len(labels_dict[key])
            }
        except Exception as e:
            # 계산 실패한 그룹 정보 저장
            error_msg = str(e)
            num_samples = len(labels_dict[key])

            # 라벨 분포 확인
            unique_labels = np.unique(labels_dict[key])
            label_counts = {int(label): int(np.sum(labels_dict[key] == label)) for label in unique_labels}

            skipped_groups[key] = {
                'reason': error_msg,
                'num_samples': num_samples,
                'label_distribution': label_counts,
                'error_type': type(e).__name__
            }

            print(f"Warning: Skipping {key} - {error_msg} (samples={num_samples}, labels={label_counts})")
            continue

    # Step 2: Intersection 그룹 메트릭 계산 (이미 생성된 intersection 데이터 사용)
    intersection_metrics = {}

    # predictions_dict에서 이미 intersection인 항목 추출 (쉼표가 있는 key)
    for key in predictions_dict:
        if ',' in key:  # Intersection 키 (예: gender_male,race_asian)
            # 이미 single_metrics에 포함되어 있으므로 intersection_metrics로 분리
            if key in single_metrics:
                intersection_metrics[key] = single_metrics[key]

    # Step 3: Group별 전체 메트릭 계산
    group_metrics = {}
    for group_name in attribute_groups:
        group_preds = []
        group_labels = []

        for key in predictions_dict:
            if key.startswith(f"{group_name}_"):
                group_preds.extend(predictions_dict[key])
                group_labels.extend(labels_dict[key])

        if len(group_labels) > 0:
            try:
                auc, er, fpr, tpr, acc, precision, f_g, eer, pr, nr, fnr, tnr = classification_metrics(
                    np.array(group_labels), np.array(group_preds)
                )

                group_metrics[group_name] = {
                    'auc': auc,
                    'error_rate': er,
                    'FPR': fpr,
                    'TPR': tpr,
                    'acc': acc,
                    'precision': precision,
                    'F_G': f_g,
                    'eer': eer,
                    'positive_rate': pr,
                    'negative_rate': nr,
                    'FNR': fnr,
                    'TNR': tnr,
                }
            except Exception as e:
                print(f"Warning: Failed to compute group metrics for {group_name}: {e}")

    # Step 4: Intra-group Fairness 계산 (단일 그룹 내 공정성)
    intra_fairness = {}

    for group_name in attribute_groups:
        subgroup_metrics = {}
        for key, value in single_metrics.items():
            # 단일 그룹만 사용 (intersection 그룹 제외)
            if key.startswith(f"{group_name}_") and ',' not in key:
                subgroup_name = key.replace(f"{group_name}_", "")
                subgroup_metrics[subgroup_name] = value

        if len(subgroup_metrics) >= 2:
            fprs = [m['FPR'] for m in subgroup_metrics.values()]
            tprs = [m['TPR'] for m in subgroup_metrics.values()]
            fnrs = [m['FNR'] for m in subgroup_metrics.values()]
            tnrs = [m['TNR'] for m in subgroup_metrics.values()]
            accs = [m['acc'] for m in subgroup_metrics.values()]
            prs = [m['positive_rate'] for m in subgroup_metrics.values()]
            nrs = [m['negative_rate'] for m in subgroup_metrics.values()]

            # Intra-group Overall FPR/TPR 계산 (전체 샘플 가중 — PG-FDD 방식)
            group_all_labels = []
            group_all_preds = []
            for key in predictions_dict:
                if key.startswith(f"{group_name}_") and ',' not in key:
                    group_all_labels.extend(labels_dict[key])
                    group_all_preds.extend(predictions_dict[key])

            # Overall FPR/TPR (가중 평균)
            if len(group_all_labels) > 0:
                try:
                    _, _, overall_fpr_weighted, overall_tpr_weighted, _, _, _, _, _, _, _, _ = \
                        classification_metrics(np.array(group_all_labels), np.array(group_all_preds))
                except Exception:
                    overall_fpr_weighted = np.mean(fprs)
                    overall_tpr_weighted = np.mean(tprs)
            else:
                overall_fpr_weighted = np.mean(fprs)
                overall_tpr_weighted = np.mean(tprs)

            # Intra-group fairness metrics
            # F_FPR (Mean): 서브그룹 FPR 평균과의 차이 합 (비가중)
            mean_fpr = np.mean(fprs)
            intra_fairness[f'{group_name}_F_FPR'] = sum([abs(fpr - mean_fpr) for fpr in fprs]) * 100

            # F_FPR_overall: 전체 샘플 overall FPR과의 차이 합 (가중, PG-FDD 방식)
            intra_fairness[f'{group_name}_F_FPR_overall'] = \
                sum([abs(fpr - overall_fpr_weighted) for fpr in fprs]) * 100

            # F_FPR_maxmin: max-min 방식
            intra_fairness[f'{group_name}_F_FPR_maxmin'] = (max(fprs) - min(fprs)) * 100

            intra_fairness[f'{group_name}_F_OAE'] = (max(accs) - min(accs)) * 100
            intra_fairness[f'{group_name}_F_DP'] = max(max(prs) - min(prs), max(nrs) - min(nrs)) * 100
            intra_fairness[f'{group_name}_F_MEO'] = max(
                max(fprs) - min(fprs),
                max(fnrs) - min(fnrs),
                max(tnrs) - min(tnrs),
                max(tprs) - min(tprs)
            ) * 100

            # F_EO: Equalized Odds (PG-FDD 방식) — FPR+TPR 차이 합계
            intra_fairness[f'{group_name}_F_EO'] = \
                sum([abs(fpr - overall_fpr_weighted) + abs(tpr - overall_tpr_weighted)
                     for fpr, tpr in zip(fprs, tprs)]) * 100

    # Step 5: Inter-group Fairness 계산 (교차 그룹 간 공정성)
    inter_fairness = {}

    if len(intersection_metrics) >= 2:
        # 모든 intersection 그룹의 메트릭 수집
        inter_fprs = [m['FPR'] for m in intersection_metrics.values()]
        inter_tprs = [m['TPR'] for m in intersection_metrics.values()]
        inter_fnrs = [m['FNR'] for m in intersection_metrics.values()]
        inter_tnrs = [m['TNR'] for m in intersection_metrics.values()]
        inter_accs = [m['acc'] for m in intersection_metrics.values()]
        inter_prs = [m['positive_rate'] for m in intersection_metrics.values()]
        inter_nrs = [m['negative_rate'] for m in intersection_metrics.values()]

        if len(inter_fprs) >= 2:
            # Inter-group Overall FPR/TPR 계산 (전체 샘플 가중 — PG-FDD 방식)
            all_inter_labels = []
            all_inter_preds = []
            for key in intersection_metrics:
                if key in predictions_dict:
                    all_inter_labels.extend(labels_dict[key])
                    all_inter_preds.extend(predictions_dict[key])

            if len(all_inter_labels) > 0:
                try:
                    _, _, overall_inter_fpr_weighted, overall_inter_tpr_weighted, \
                        _, _, _, _, _, _, _, _ = \
                        classification_metrics(np.array(all_inter_labels), np.array(all_inter_preds))
                except Exception:
                    overall_inter_fpr_weighted = np.mean(inter_fprs)
                    overall_inter_tpr_weighted = np.mean(inter_tprs)
            else:
                overall_inter_fpr_weighted = np.mean(inter_fprs)
                overall_inter_tpr_weighted = np.mean(inter_tprs)

            # Inter-group fairness metrics
            # F_FPR (Mean): 서브그룹 FPR 평균과의 차이 합 (비가중)
            mean_inter_fpr = np.mean(inter_fprs)
            inter_fairness['inter_F_FPR'] = sum([abs(fpr - mean_inter_fpr) for fpr in inter_fprs]) * 100

            # F_FPR_overall: 전체 샘플 overall FPR과의 차이 합 (가중, PG-FDD 방식)
            inter_fairness['inter_F_FPR_overall'] = \
                sum([abs(fpr - overall_inter_fpr_weighted) for fpr in inter_fprs]) * 100

            # F_FPR_maxmin: max-min 방식
            inter_fairness['inter_F_FPR_maxmin'] = (max(inter_fprs) - min(inter_fprs)) * 100

            inter_fairness['inter_F_OAE'] = (max(inter_accs) - min(inter_accs)) * 100
            inter_fairness['inter_F_DP'] = max(
                max(inter_prs) - min(inter_prs),
                max(inter_nrs) - min(inter_nrs)
            ) * 100
            inter_fairness['inter_F_MEO'] = max(
                max(inter_fprs) - min(inter_fprs),
                max(inter_fnrs) - min(inter_fnrs),
                max(inter_tnrs) - min(inter_tnrs),
                max(inter_tprs) - min(inter_tprs)
            ) * 100

            # F_EO: Equalized Odds (PG-FDD 방식) — FPR+TPR 차이 합계
            inter_fairness['inter_F_EO'] = \
                sum([abs(fpr - overall_inter_fpr_weighted) + abs(tpr - overall_inter_tpr_weighted)
                     for fpr, tpr in zip(inter_fprs, inter_tprs)]) * 100

    # Step 6: 추가 메트릭 계산 (F_S, F_A_inter 등)
    overall_fairness = {}

    # F_S 계산 (Statistical Parity 차이)
    if len(intersection_metrics) >= 2:
        f_g_values = [m['F_G'] for m in intersection_metrics.values()]
        if len(f_g_values) >= 2:
            # 모든 쌍의 차이 중 최대값
            max_diff = 0
            for i in range(len(f_g_values)):
                for j in range(i+1, len(f_g_values)):
                    diff = abs(f_g_values[i] - f_g_values[j])
                    if diff > max_diff:
                        max_diff = diff
            overall_fairness['F_S'] = max_diff * 100

    # F_A_inter 계산 (평균 inter-group 차이)
    if 'F_S' in overall_fairness and len(group_metrics) > 0:
        # 각 intersection과 전체 평균의 차이 계산
        overall_f_g = np.mean([m['F_G'] for m in group_metrics.values()])
        inter_diffs = []
        for metrics in intersection_metrics.values():
            inter_diffs.append(abs(metrics['F_G'] - overall_f_g))

        if inter_diffs:
            max_inter_diff = max(inter_diffs)
            overall_fairness['F_A_inter'] = (overall_fairness['F_S'] + max_inter_diff * 100) / 2

    # single_group_metrics에서 intersection 제외 (쉼표가 없는 키만)
    single_group_only = {k: v for k, v in single_metrics.items() if ',' not in k}

    return {
        'single_group_metrics': single_group_only,
        'intersection_metrics': intersection_metrics,
        'group_overall_metrics': group_metrics,
        'intra_fairness': intra_fairness,
        'inter_fairness': inter_fairness,
        'overall_fairness': overall_fairness,
        'skipped_groups': skipped_groups  # 계산 실패한 그룹 정보
    }


def compute_fairness_with_indices(data_with_indices, attribute_columns, target_column='label',
                                 prediction_column='prediction'):
    """
    인덱스 기반 데이터로 정확한 intersection 계산

    Args:
        data_with_indices: DataFrame 또는 dict with columns:
                          - prediction: 예측값
                          - label: 실제 라벨
                          - gender: 성별 (male/female)
                          - race: 인종 (asian/white/black/others)
                          - 기타 attributes
        attribute_columns: list of attribute column names ['gender', 'race']
        target_column: 라벨 컬럼명
        prediction_column: 예측 컬럼명

    Returns:
        완전한 fairness 분석 결과
    """
    import pandas as pd

    # DataFrame으로 변환 (필요시)
    if isinstance(data_with_indices, dict):
        df = pd.DataFrame(data_with_indices)
    else:
        df = data_with_indices.copy()

    # 단일 그룹별 데이터 분리
    single_preds = {}
    single_labels = {}

    for attr in attribute_columns:
        unique_values = df[attr].unique()
        for value in unique_values:
            key = f"{attr}_{value}"
            mask = df[attr] == value
            single_preds[key] = df.loc[mask, prediction_column].values
            single_labels[key] = df.loc[mask, target_column].values

    # Intersection 그룹별 데이터 분리
    intersection_preds = {}
    intersection_labels = {}

    # 2개 attribute intersection만 계산 (gender × race)
    if len(attribute_columns) >= 2:
        for attr1, attr2 in itertools.combinations(attribute_columns, 2):
            for val1 in df[attr1].unique():
                for val2 in df[attr2].unique():
                    key = f"{attr1}_{val1},{attr2}_{val2}"
                    mask = (df[attr1] == val1) & (df[attr2] == val2)
                    if mask.sum() > 0:
                        intersection_preds[key] = df.loc[mask, prediction_column].values
                        intersection_labels[key] = df.loc[mask, target_column].values

    # 3-way 이상의 intersection은 계산하지 않음 (2-way만 사용)
    # 이유: gender × race 조합만 필요 (gender_male,race_asian 등)

    # 통합 딕셔너리 생성
    all_preds = {**single_preds, **intersection_preds}
    all_labels = {**single_labels, **intersection_labels}

    # 메트릭 계산
    return compute_fairness_metrics_intersection(all_preds, all_labels, attribute_columns)


def print_intersection_fairness_report(results):
    """
    Intersection Fairness 리포트 출력

    Args:
        results: compute_fairness_metrics_intersection 결과
    """
    print("\n" + "="*80)
    print("📊 INTERSECTION FAIRNESS METRICS REPORT")
    print("="*80)

    # 단일 서브그룹 성능
    if 'single_group_metrics' in results:
        print("\n🔹 SINGLE SUBGROUP PERFORMANCE")
        print("-" * 80)
        print(f"{'Group':<25} {'Samples':>8} {'AUC':>8} {'ACC':>8} {'EER':>8} {'FPR':>8} {'TPR':>8}")
        print("-" * 80)
        for key, metrics in results['single_group_metrics'].items():
            print(f"{key:<25} {metrics['num_samples']:>8} "
                  f"{metrics['auc']:>8.4f} {metrics['acc']:>8.4f} {metrics['eer']:>8.4f} "
                  f"{metrics['FPR']:>8.4f} {metrics['TPR']:>8.4f}")

    # Intersection 그룹 성능
    if 'intersection_metrics' in results and results['intersection_metrics']:
        print("\n🔸 INTERSECTION GROUP PERFORMANCE (Gender × Race)")
        print("-" * 80)
        print(f"{'Group':<35} {'Samples':>8} {'AUC':>8} {'ACC':>8} {'EER':>8} {'FPR':>8} {'TPR':>8}")
        print("-" * 80)
        for key, metrics in results['intersection_metrics'].items():
            print(f"{key:<35} {metrics['num_samples']:>8} "
                  f"{metrics['auc']:>8.4f} {metrics['acc']:>8.4f} {metrics['eer']:>8.4f} "
                  f"{metrics['FPR']:>8.4f} {metrics['TPR']:>8.4f}")

    # Intra-group Fairness
    if 'intra_fairness' in results and results['intra_fairness']:
        print("\n⚖️  INTRA-GROUP FAIRNESS (Within Single Attribute)")
        print("-" * 80)
        groups = sorted(set([k.split('_')[0] for k in results['intra_fairness'].keys() if not k.endswith('_maxmin')]))

        print(f"{'Attribute':<12} {'F_FPR(Mean)':>13} {'F_FPR(Over)':>13} {'F_FPR(MM)':>12} {'F_OAE':>9} {'F_DP':>9} {'F_MEO':>9} {'F_EO':>9}")
        print("-" * 100)
        for group in groups:
            f_fpr = results['intra_fairness'].get(f'{group}_F_FPR', 0)
            f_fpr_ov = results['intra_fairness'].get(f'{group}_F_FPR_overall', 0)
            f_fpr_mm = results['intra_fairness'].get(f'{group}_F_FPR_maxmin', 0)
            f_oae = results['intra_fairness'].get(f'{group}_F_OAE', 0)
            f_dp = results['intra_fairness'].get(f'{group}_F_DP', 0)
            f_meo = results['intra_fairness'].get(f'{group}_F_MEO', 0)
            f_eo = results['intra_fairness'].get(f'{group}_F_EO', 0)
            print(f"{group.upper():<12} {f_fpr:>12.3f}% {f_fpr_ov:>12.3f}% {f_fpr_mm:>11.3f}% {f_oae:>8.3f}% {f_dp:>8.3f}% {f_meo:>8.3f}% {f_eo:>8.3f}%")

    # Inter-group Fairness
    if 'inter_fairness' in results and results['inter_fairness']:
        print("\n⚖️  INTER-GROUP FAIRNESS (Across All Intersections)")
        print("-" * 80)
        inter = results['inter_fairness']
        print(f"{'Metric':<20} {'Value':>12}")
        print("-" * 80)
        if 'inter_F_FPR' in inter:
            print(f"{'F_FPR (Mean)':<20} {inter['inter_F_FPR']:>11.3f}%")
        if 'inter_F_FPR_overall' in inter:
            print(f"{'F_FPR (Overall)':<20} {inter['inter_F_FPR_overall']:>11.3f}%")
        if 'inter_F_FPR_maxmin' in inter:
            print(f"{'F_FPR (MaxMin)':<20} {inter['inter_F_FPR_maxmin']:>11.3f}%")
        for key, value in inter.items():
            if key not in ['inter_F_FPR', 'inter_F_FPR_overall', 'inter_F_FPR_maxmin']:
                metric_name = key.replace('inter_', 'F_')
                print(f"{metric_name:<20} {value:>11.3f}%")

    # Overall Fairness
    if 'overall_fairness' in results and results['overall_fairness']:
        print("\n📈 OVERALL FAIRNESS METRICS")
        print("-" * 80)
        print(f"{'Metric':<25} {'Value':>12}")
        print("-" * 80)
        for key, value in results['overall_fairness'].items():
            print(f"{key:<25} {value:>11.3f}%")

    # Fairness 비교 (Intra vs Inter)
    if 'intra_fairness' in results and 'inter_fairness' in results:
        print("\n🔍 FAIRNESS COMPARISON (Intra vs Inter)")
        print("-" * 80)

        # 평균 계산
        intra_values = [v for k, v in results['intra_fairness'].items() if 'F_FPR' in k]
        inter_value = results['inter_fairness'].get('inter_F_FPR', 0)

        if intra_values:
            avg_intra = np.mean(intra_values)
            print(f"{'Metric':<30} {'Value':>12}")
            print("-" * 80)
            print(f"{'Average Intra-group F_FPR':<30} {avg_intra:>11.3f}%")
            print(f"{'Inter-group F_FPR':<30} {inter_value:>11.3f}%")
            print("-" * 80)

            if inter_value > avg_intra:
                ratio = inter_value / avg_intra
                print(f"⚠️  Inter-group bias is {ratio:.1f}x higher than intra-group")
            else:
                print(f"✓ Inter-group fairness is comparable to intra-group")

    # Skipped Groups 정보 출력
    if 'skipped_groups' in results and results['skipped_groups']:
        print("\n⚠️  SKIPPED GROUPS (Computation Failed)")
        print("-" * 80)
        print(f"{'Group':<35} {'Samples':>8} {'Label Dist':>20} {'Error':>15}")
        print("-" * 80)
        for group_name, info in results['skipped_groups'].items():
            label_dist = str(info['label_distribution'])
            if len(label_dist) > 20:
                label_dist = label_dist[:17] + "..."
            print(f"{group_name:<35} {info['num_samples']:>8} {label_dist:>20} {info['error_type']:>15}")
            print(f"  └─ Reason: {info['reason']}")

    print("\n" + "="*80)


def backward_compatible_wrapper(predictions_dict, labels_dict, attribute_groups,
                               use_intersection=False):
    """
    기존 코드와의 호환성을 위한 wrapper 함수

    Args:
        predictions_dict: 예측값 딕셔너리
        labels_dict: 라벨 딕셔너리
        attribute_groups: 속성 그룹 리스트
        use_intersection: True면 intersection 분석, False면 기존 방식

    Returns:
        기존 형식과 호환되는 결과
    """
    if not use_intersection:
        # 기존 방식 사용 (단일 서브그룹만)
        from utils.fairness_metrics_advanced import compute_fairness_metrics_advanced
        return compute_fairness_metrics_advanced(predictions_dict, labels_dict, attribute_groups)

    # Intersection 분석 사용
    results = compute_fairness_metrics_intersection(predictions_dict, labels_dict, attribute_groups)

    # 기존 형식으로 변환
    backward_results = {
        'per_group_metrics': results['single_group_metrics'],
        'group_overall_metrics': results['group_overall_metrics'],
        'fairness_metrics': {}
    }

    # Fairness metrics 통합
    backward_results['fairness_metrics'].update(results.get('intra_fairness', {}))

    # Inter fairness를 추가 (접두사 유지)
    if 'inter_fairness' in results:
        backward_results['fairness_metrics'].update(results['inter_fairness'])

    # Overall fairness 추가
    if 'overall_fairness' in results:
        backward_results['fairness_metrics'].update(results['overall_fairness'])

    return backward_results


if __name__ == "__main__":
    # 테스트 예제
    print("Intersection Fairness Metrics Module")
    print("\nUsage Examples:")
    print("\n1. Basic intersection analysis:")
    print("  from utils.fairness_metrics_intersection import compute_fairness_metrics_intersection")
    print("  results = compute_fairness_metrics_intersection(preds_dict, labels_dict, ['gender', 'race'])")
    print("  print_intersection_fairness_report(results)")

    print("\n2. With DataFrame input (accurate intersection):")
    print("  from utils.fairness_metrics_intersection import compute_fairness_with_indices")
    print("  df = pd.DataFrame({")
    print("      'prediction': [...],")
    print("      'label': [...],")
    print("      'gender': ['male', 'female', ...],")
    print("      'race': ['asian', 'white', ...]")
    print("  })")
    print("  results = compute_fairness_with_indices(df, ['gender', 'race'])")

    print("\n3. Backward compatible mode:")
    print("  from utils.fairness_metrics_intersection import backward_compatible_wrapper")
    print("  results = backward_compatible_wrapper(preds_dict, labels_dict, ['gender', 'race'], use_intersection=True)")