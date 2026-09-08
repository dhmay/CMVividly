from typing import Dict, Tuple
import numpy as np
from cmvividly.data.access import load_cmv_ecocluster_2026
from abc import ABC, abstractmethod


class AbstractBreadthClassifier(ABC):
    """Classifies some status from TCR repertoire breadth on some list of TCRs
    (fraction of TCRs matching the CMV ECOcluster).

    The call threshold is fixed at the max-F1 point of the breadth ROC curve
    (see cells above). Confidence, though, is computed *per sample*: each
    sample's own breadth value is located on the ROC curve, and the FPR/TPR
    at that point are converted to precision (for a + call) or NPV (for a
    - call) -- rather than using one fixed confidence for every call.

    All ROC-derived data below (threshold, FPR/TPR curve, and label counts)
    are baked in as constants from the breadth vs. status ROC analysis on
    the Emerson cohorts, rather than recomputed from data at call time.
    """

    def initialize(self, pdf_tcrs, breadth_roc_thresholds: list,
                   breadth_roc_tpr: list, breadth_roc_fpr: list,
                   breadth_max_f1_threshold: float,
                   n_pos_breadth: int, n_neg_breadth: int
                   ) -> None:
        """Initialize the classifier with the specified TCRs, so
        classify_repertoire() can look up matches against them.

        Args:
            pdf_tcrs: DataFrame of TCRs, with a "tcr" column
            breadth_roc_thresholds: list of thresholds from the ROC curve for breadth vs. label 
            breadth_roc_tpr: list of TPR values from the ROC curve for breadth vs. label
            breadth_roc_fpr: list of FPR values from the ROC curve for breadth vs. label
            breadth_max_f1_threshold: threshold for breadth that maximizes F1 score
            n_pos_breadth: number of positive samples in the ROC analysis
            n_neg_breadth: number of negative samples in the ROC analysis
        """
        self.pdf_tcrs = pdf_tcrs
        self.tcrs = set(pdf_tcrs["tcr"])
        self.breadth_roc_thresholds = breadth_roc_thresholds
        self.breadth_max_f1_threshold = breadth_max_f1_threshold
        self.breadth_roc_tpr = breadth_roc_tpr
        self.breadth_roc_fpr = breadth_roc_fpr
        self.n_pos_breadth = n_pos_breadth
        self.n_neg_breadth = n_neg_breadth
 
    def classify_breadth(self, breadth: float) -> tuple:
        """Classify status from a breadth value.

        Args:
            breadth: breadth value for a sample

        Returns:
            (prediction, confidence): prediction is 1 (label+) or 0 (label-); confidence
                is the estimated probability that this specific prediction is correct,
                derived from the FPR/TPR at the ROC curve point nearest this sample's
                own breadth value (precision if predicted positive, NPV if negative).
        """
        prediction = int(breadth >= self.breadth_max_f1_threshold)

        thresholds = np.asarray(self.breadth_roc_thresholds)
        idx = int(np.argmin(np.abs(thresholds - breadth)))

        fpr_at_breadth = self.breadth_roc_fpr[idx]
        tpr_at_breadth = self.breadth_roc_tpr[idx]

        tp = tpr_at_breadth * self.n_pos_breadth
        fp = fpr_at_breadth * self.n_neg_breadth
        fn = self.n_pos_breadth - tp
        tn = self.n_neg_breadth - fp

        p = tp + fp
        n = tn + fn

        if prediction == 1:
            confidence = tp / p if p > 0 else 1.0
        else:
            confidence = tn / n if n > 0 else 1.0
        return prediction, confidence

    def count_matches(self, pdf_onerepertoire_tcrs) -> int:
        """Count the rows in pdf_onerepertoire_tcrs whose "tcr" value is in
        the set of TCRs in self.tcrs.

        Args:
            pdf_onerepertoire_tcrs: DataFrame of TCRs for one repertoire, with a "tcr" column

        Returns:
            number of rows whose "tcr" value is in self.tcrs
        """
        return int(pdf_onerepertoire_tcrs["tcr"].isin(self.tcrs).sum())

    def calc_nmatches_and_breadth(self, pdf_onerepertoire_tcrs) -> Tuple[int, float]:
        """Count the number of matches and compute breadth for one repertoire.

        Args:
            pdf_onerepertoire_tcrs: DataFrame of TCRs for one repertoire, with a "tcr" column

        Returns:
            (n_matches, breadth): number of matches and breadth (fraction of TCRs that match)
        """
        n_tcrs = len(pdf_onerepertoire_tcrs)
        n_matches = self.count_matches(pdf_onerepertoire_tcrs)
        breadth = n_matches / n_tcrs if n_tcrs > 0 else 0.0
        return n_matches, breadth

    def classify_repertoire(self, pdf_onerepertoire_tcrs) -> Dict:
        """Classify status for one repertoire

        Args:
            pdf_onerepertoire_tcrs: DataFrame of all TCRs for one repertoire, with a "tcr" column

        Returns:
            dict with keys "prediction", "prediction_confidence", "n_matches", "n_tcrs", "cmv_breadth"
        """
        n_tcrs = len(pdf_onerepertoire_tcrs)
        n_matches, breadth = self.calc_nmatches_and_breadth(pdf_onerepertoire_tcrs)
        prediction, prediction_confidence = self.classify_breadth(breadth)
        return {
            "prediction": prediction,
            "prediction_confidence": prediction_confidence,
            "n_matches": n_matches,
            "total_tcrs": n_tcrs,
            "breadth": breadth,
        }
