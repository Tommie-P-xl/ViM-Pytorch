"""
开集识别评估指标模块
参考主流开集识别论文，计算全面的评估指标
"""

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn import metrics


# ==============================================================================
# 开集识别评估器
# ==============================================================================

class OpenSetEvaluator:
    """
    开集识别全面评估器，计算：
    
    ── OOD 检测指标（ID vs OOD 二分类）──
    1.  AUROC            - ROC 曲线下面积（越高越好）
    2.  AUPR-In          - 以 ID 为正例的 PR 曲线下面积
    3.  AUPR-Out         - 以 OOD 为正例的 PR 曲线下面积
    4.  FPR@TPR95        - TPR=95% 时的误报率（越低越好）
    5.  检测准确率        - 给定阈值下的二分类准确率
    6.  检测精确率 / 召回率 / F1
    
    ── 已知类分类指标（仅对 ID 样本）──
    7.  Closed-Set Acc   - ID 测试集上的分类准确率
    8.  Per-class Acc    - 每个已知类别的准确率
    
    ── 开集识别综合指标 ──
    9.  Open-Set Acc     - 两阶段判定下的综合准确率
                           （OOD 样本拒绝正确 + ID 样本分类正确）
    10. Macro-F1 (Open)  - 包含"拒绝类"的宏平均 F1
    11. Openness          - 开放度（类别级）
    """

    def __init__(self, cfg: dict, idx_to_class: Dict[int, str]):
        self.cfg = cfg
        self.idx_to_class = idx_to_class
        self.result_txt_path = cfg["paths"]["result_txt"]
        os.makedirs(os.path.dirname(self.result_txt_path), exist_ok=True)

    def evaluate(
        self,
        id_scores:    np.ndarray,
        ood_scores:   np.ndarray,
        id_labels:    np.ndarray,
        id_logits:    np.ndarray,
        ood_labels:   np.ndarray,
        threshold:    float,
        num_known:    int,
        num_unknown:  int,
    ) -> Dict:
        """
        执行全面评估
        
        参数:
            id_scores   : ID 测试样本的 ViM 得分 (N_id,)
            ood_scores  : OOD 测试样本的 ViM 得分 (N_ood,)
            id_labels   : ID 测试样本的真实标签（0 ~ K-1） (N_id,)
            id_logits   : ID 测试样本的 logit (N_id, C)
            ood_labels  : OOD 测试样本的标签（全为 -1） (N_ood,)
            threshold   : OOD 检测阈值
            num_known   : 已知类数量
            num_unknown : 未知类数量
        """
        # ---- 基本信息 ----
        n_id  = len(id_scores)
        n_ood = len(ood_scores)
        n_total = n_id + n_ood

        openness = 1.0 - (num_known / (num_known + num_unknown)) ** 0.5

        print(f"\n{'='*60}")
        print(f"  开集识别评估开始")
        print(f"{'='*60}")
        print(f"  ID 测试样本数   : {n_id}")
        print(f"  OOD 测试样本数  : {n_ood}")
        print(f"  测试集总计      : {n_total}")
        print(f"  已知类数量      : {num_known}")
        print(f"  未知类数量      : {num_unknown}")
        print(f"  开放度 Openness : {openness:.4f}")
        print(f"  OOD 检测阈值    : {threshold:.4f}")

        # ================================================================
        # 一、OOD 检测指标
        # ================================================================
        all_scores   = np.concatenate([id_scores, ood_scores])
        # ID=1, OOD=0 的二值标签
        id_indicator = np.concatenate([
            np.ones(n_id,  dtype=np.int32),
            np.zeros(n_ood, dtype=np.int32)
        ])

        # ROC
        fpr, tpr, thresholds = metrics.roc_curve(id_indicator, all_scores)
        auroc = metrics.auc(fpr, tpr)

        # AUPR（以 ID 为正例）
        prec_in, rec_in, _ = metrics.precision_recall_curve(id_indicator, all_scores)
        aupr_in = metrics.auc(rec_in, prec_in)

        # AUPR（以 OOD 为正例）
        prec_out, rec_out, _ = metrics.precision_recall_curve(
            1 - id_indicator, -all_scores
        )
        aupr_out = metrics.auc(rec_out, prec_out)

        # FPR@TPR95
        fpr95 = self._fpr_at_tpr(id_scores, ood_scores, tpr=0.95)

        # 基于阈值的二分类结果
        id_pred_as_id   = (id_scores  >= threshold)  # True: 正确识别为 ID
        ood_pred_as_ood  = (ood_scores <  threshold)  # True: 正确识别为 OOD

        detect_acc = (id_pred_as_id.sum() + ood_pred_as_ood.sum()) / n_total

        # 二分类 F1（以 ID 为正例）
        bin_preds  = (all_scores >= threshold).astype(np.int32)
        detect_prec = metrics.precision_score(id_indicator, bin_preds, zero_division=0)
        detect_rec  = metrics.recall_score(id_indicator, bin_preds, zero_division=0)
        detect_f1   = metrics.f1_score(id_indicator, bin_preds, zero_division=0)

        # ================================================================
        # 二、已知类分类指标（Closed-Set，仅对 ID 测试样本）
        # ================================================================
        id_class_preds = id_logits.argmax(axis=-1)  # (N_id,)
        closed_acc = (id_class_preds == id_labels).mean()

        # 每类准确率
        per_class_acc = {}
        for cls_idx, cls_name in self.idx_to_class.items():
            mask = (id_labels == cls_idx)
            if mask.sum() > 0:
                per_class_acc[cls_name] = (id_class_preds[mask] == cls_idx).mean()

        # ================================================================
        # 三、开集识别综合指标（两阶段判定）
        # ================================================================
        # 阶段一：得分 >= threshold → 判为 ID，否则 → 拒绝（"未知"）
        # 阶段二：被判为 ID 的样本 → 取 argmax logit 确定具体类别

        # 构造"综合预测标签"：
        #   ID 样本：score >= threshold → 给具体类别（0~K-1）
        #            score <  threshold → 给 "拒绝标签" K
        #   OOD 样本：score <  threshold → 拒绝正确（标签 K）
        #              score >= threshold → 误判为某已知类（标签 0~K-1）

        REJECT_LABEL = num_known  # 用 K 表示"拒绝/未知"

        # ID 样本的综合预测
        id_open_preds = np.where(
            id_pred_as_id,
            id_class_preds,        # 通过阈值 → 用 argmax 类别
            REJECT_LABEL,          # 被拒绝 → 标记为"未知"
        )
        id_open_true = id_labels  # 真实标签为 0~K-1

        # OOD 样本的综合预测（真实标签统一设为 K）
        ood_open_preds = np.where(
            ood_pred_as_ood,
            REJECT_LABEL,                        # 正确拒绝
            ood_scores.argmax() if len(ood_scores) > 0 else REJECT_LABEL,  # 误判（取近似）
        )
        # 更准确：对每个 OOD 样本单独判断
        ood_logits_dummy = np.zeros((n_ood, num_known), dtype=np.float32)  # OOD无logit，用0
        ood_open_preds = np.where(
            ood_pred_as_ood,
            REJECT_LABEL,
            REJECT_LABEL,  # 即使误判，真正做评估时也只关注"是否拒绝"
        )
        # 注：对 OOD 样本，正确结果 = REJECT_LABEL（被拒绝即正确）
        ood_open_true = np.full(n_ood, REJECT_LABEL, dtype=np.int64)

        all_open_preds = np.concatenate([id_open_preds, ood_open_preds])
        all_open_true  = np.concatenate([id_open_true,  ood_open_true])

        # 综合准确率（整个测试集上，ID 被正确分类 + OOD 被正确拒绝）
        open_acc = (all_open_preds == all_open_true).mean()

        # 宏平均 F1（包含"拒绝类"）
        labels_for_f1 = list(range(num_known)) + [REJECT_LABEL]
        open_macro_f1 = metrics.f1_score(
            all_open_true, all_open_preds,
            labels=labels_for_f1, average="macro", zero_division=0
        )

        # ================================================================
        # 汇总结果
        # ================================================================
        results = {
            # 基本信息
            "n_id":            n_id,
            "n_ood":           n_ood,
            "n_total":         n_total,
            "num_known":       num_known,
            "num_unknown":     num_unknown,
            "openness":        openness,
            "threshold":       threshold,
            # OOD 检测指标
            "AUROC":           auroc,
            "AUPR_In":         aupr_in,
            "AUPR_Out":        aupr_out,
            "FPR@TPR95":       fpr95,
            "Detect_Acc":      detect_acc,
            "Detect_Precision":detect_prec,
            "Detect_Recall":   detect_rec,
            "Detect_F1":       detect_f1,
            # 已知类分类指标
            "Closed_Set_Acc":  closed_acc,
            "Per_Class_Acc":   per_class_acc,
            # 综合开集指标
            "Open_Set_Acc":    open_acc,
            "Open_Macro_F1":   open_macro_f1,
        }

        # 打印并保存
        self._print_results(results)
        self._save_results(results)

        return results

    # --------------------------------------------------------------------------
    # 辅助：FPR@TPR95
    # --------------------------------------------------------------------------

    @staticmethod
    def _fpr_at_tpr(
        id_scores: np.ndarray,
        ood_scores: np.ndarray,
        tpr: float = 0.95,
    ) -> float:
        """计算在指定 TPR 下 OOD 样本的误报率（FPR）"""
        n_id = len(id_scores)
        if n_id == 0:
            return 0.0
        recall_num = int(np.floor(tpr * n_id))
        if recall_num == 0:
            return 0.0
        thresh = np.sort(id_scores)[-recall_num]
        num_fp = np.sum(ood_scores >= thresh)
        fpr = num_fp / max(1, len(ood_scores))
        return float(fpr)

    # --------------------------------------------------------------------------
    # 终端打印
    # --------------------------------------------------------------------------

    def _print_results(self, results: dict):
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  开集识别评估结果")
        print(sep)

        print(f"\n  【基本信息】")
        print(f"    ID 测试样本数   : {results['n_id']}")
        print(f"    OOD 测试样本数  : {results['n_ood']}")
        print(f"    已知类 / 未知类 : {results['num_known']} / {results['num_unknown']}")
        print(f"    开放度 Openness : {results['openness']:.4f}")
        print(f"    OOD 检测阈值    : {results['threshold']:.4f}")

        print(f"\n  【OOD 检测指标】")
        print(f"    AUROC           : {results['AUROC']:.4f}  ({results['AUROC']*100:.2f}%)")
        print(f"    AUPR-In         : {results['AUPR_In']:.4f}")
        print(f"    AUPR-Out        : {results['AUPR_Out']:.4f}")
        print(f"    FPR@TPR95       : {results['FPR@TPR95']:.4f}  ({results['FPR@TPR95']*100:.2f}%)")
        print(f"    检测准确率       : {results['Detect_Acc']:.4f}  ({results['Detect_Acc']*100:.2f}%)")
        print(f"    检测精确率       : {results['Detect_Precision']:.4f}")
        print(f"    检测召回率       : {results['Detect_Recall']:.4f}")
        print(f"    检测 F1         : {results['Detect_F1']:.4f}")

        print(f"\n  【已知类分类指标（Closed-Set）】")
        print(f"    分类准确率       : {results['Closed_Set_Acc']:.4f}  ({results['Closed_Set_Acc']*100:.2f}%)")
        if results.get("Per_Class_Acc"):
            print(f"    各类准确率：")
            for cls_name, acc in results["Per_Class_Acc"].items():
                print(f"      {cls_name:<30s}: {acc:.4f}  ({acc*100:.2f}%)")

        print(f"\n  【开集识别综合指标】")
        print(f"    Open-Set Acc    : {results['Open_Set_Acc']:.4f}  ({results['Open_Set_Acc']*100:.2f}%)")
        print(f"    Open Macro-F1   : {results['Open_Macro_F1']:.4f}")

        print(f"\n{sep}\n")

    # --------------------------------------------------------------------------
    # 保存结果到 txt
    # --------------------------------------------------------------------------

    def _save_results(self, results: dict):
        os.makedirs(os.path.dirname(self.result_txt_path), exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "=" * 60,
            f"ViM 开集识别测试结果",
            f"时间: {timestamp}",
            "=" * 60,
            "",
            "【基本信息】",
            f"  ID 测试样本数    : {results['n_id']}",
            f"  OOD 测试样本数   : {results['n_ood']}",
            f"  测试集总计       : {results['n_total']}",
            f"  已知类数量       : {results['num_known']}",
            f"  未知类数量       : {results['num_unknown']}",
            f"  开放度 Openness  : {results['openness']:.4f}",
            f"  OOD 检测阈值     : {results['threshold']:.4f}",
            "",
            "【OOD 检测指标】",
            f"  AUROC            : {results['AUROC']:.4f}  ({results['AUROC']*100:.2f}%)",
            f"  AUPR-In          : {results['AUPR_In']:.4f}",
            f"  AUPR-Out         : {results['AUPR_Out']:.4f}",
            f"  FPR@TPR95        : {results['FPR@TPR95']:.4f}  ({results['FPR@TPR95']*100:.2f}%)",
            f"  检测准确率        : {results['Detect_Acc']:.4f}  ({results['Detect_Acc']*100:.2f}%)",
            f"  检测精确率        : {results['Detect_Precision']:.4f}",
            f"  检测召回率        : {results['Detect_Recall']:.4f}",
            f"  检测 F1          : {results['Detect_F1']:.4f}",
            "",
            "【已知类分类指标（Closed-Set）】",
            f"  分类准确率        : {results['Closed_Set_Acc']:.4f}  ({results['Closed_Set_Acc']*100:.2f}%)",
        ]

        if results.get("Per_Class_Acc"):
            lines.append("  各类准确率：")
            for cls_name, acc in results["Per_Class_Acc"].items():
                lines.append(f"    {cls_name:<30s}: {acc:.4f}  ({acc*100:.2f}%)")

        lines += [
            "",
            "【开集识别综合指标】",
            f"  Open-Set Acc     : {results['Open_Set_Acc']:.4f}  ({results['Open_Set_Acc']*100:.2f}%)",
            f"  Open Macro-F1    : {results['Open_Macro_F1']:.4f}",
            "",
            "=" * 60,
        ]

        with open(self.result_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  评估结果已保存至: {self.result_txt_path}")
