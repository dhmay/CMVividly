from typing import Dict, Tuple
import numpy as np
from cmvividly.data.access import load_cmv_ecocluster_2026

BREADTH_ROC_FPR = [0.0, 0.002242152466367713, 0.002242152466367713, 0.004484304932735426, 0.004484304932735426, 0.006726457399103139, 0.006726457399103139, 0.008968609865470852, 0.008968609865470852, 0.011210762331838564, 0.011210762331838564, 0.013452914798206279, 0.013452914798206279, 0.01569506726457399, 0.01569506726457399, 0.017937219730941704, 0.017937219730941704, 0.020179372197309416, 0.020179372197309416, 0.02242152466367713, 0.02242152466367713, 0.02466367713004484, 0.02466367713004484, 0.026905829596412557, 0.026905829596412557, 0.02914798206278027, 0.02914798206278027, 0.03139013452914798, 0.03139013452914798, 0.033632286995515695, 0.033632286995515695, 0.03587443946188341, 0.03587443946188341, 0.03811659192825112, 0.03811659192825112, 0.04035874439461883, 0.04035874439461883, 0.042600896860986545, 0.042600896860986545, 0.04484304932735426, 0.04484304932735426, 0.04708520179372197, 0.04708520179372197, 0.04932735426008968, 0.04932735426008968, 0.0515695067264574, 0.0515695067264574, 0.05605381165919283, 0.05605381165919283, 0.06950672645739911, 0.06950672645739911, 0.07399103139013453, 0.07399103139013453, 0.1412556053811659, 0.1412556053811659, 0.1547085201793722, 0.1547085201793722, 0.16143497757847533, 0.16143497757847533, 0.1681614349775785, 0.1681614349775785, 0.1860986547085202, 0.1860986547085202, 0.21973094170403587, 0.21973094170403587, 0.24887892376681614, 0.24887892376681614, 0.2623318385650224, 0.2623318385650224, 0.30493273542600896, 0.30493273542600896, 0.37668161434977576, 0.37668161434977576, 0.5, 0.5, 0.5493273542600897, 0.5493273542600897, 0.7443946188340808, 0.7443946188340808, 0.7600896860986547, 0.7600896860986547, 0.8475336322869955, 0.8475336322869955, 1.0]
BREADTH_ROC_TPR = [0.0, 0.0, 0.029411764705882353, 0.029411764705882353, 0.03823529411764706, 0.03823529411764706, 0.06470588235294118, 0.06470588235294118, 0.06764705882352941, 0.06764705882352941, 0.08235294117647059, 0.08235294117647059, 0.11176470588235295, 0.11176470588235295, 0.1264705882352941, 0.1264705882352941, 0.13823529411764707, 0.13823529411764707, 0.2529411764705882, 0.2529411764705882, 0.34411764705882353, 0.34411764705882353, 0.36764705882352944, 0.36764705882352944, 0.40588235294117647, 0.40588235294117647, 0.5, 0.5, 0.5735294117647058, 0.5735294117647058, 0.6764705882352942, 0.6764705882352942, 0.7617647058823529, 0.7617647058823529, 0.7735294117647059, 0.7735294117647059, 0.8176470588235294, 0.8176470588235294, 0.9235294117647059, 0.9235294117647059, 0.9294117647058824, 0.9294117647058824, 0.9323529411764706, 0.9323529411764706, 0.9352941176470588, 0.9352941176470588, 0.9382352941176471, 0.9382352941176471, 0.95, 0.95, 0.9529411764705882, 0.9529411764705882, 0.9558823529411765, 0.9558823529411765, 0.9588235294117647, 0.9588235294117647, 0.961764705882353, 0.961764705882353, 0.9647058823529412, 0.9647058823529412, 0.9676470588235294, 0.9676470588235294, 0.9705882352941176, 0.9705882352941176, 0.9735294117647059, 0.9735294117647059, 0.9764705882352941, 0.9764705882352941, 0.9794117647058823, 0.9794117647058823, 0.9823529411764705, 0.9823529411764705, 0.9852941176470589, 0.9852941176470589, 0.9882352941176471, 0.9882352941176471, 0.9911764705882353, 0.9911764705882353, 0.9941176470588236, 0.9941176470588236, 0.9970588235294118, 0.9970588235294118, 1.0, 1.0]
BREADTH_ROC_THRESHOLDS = [float("inf"), 0.007057312061667884, 0.0033188843644642936, 0.0032893053718011504, 0.0030272372930538237, 0.0029951940749615388, 0.0027278221525293606, 0.0026933074441765057, 0.0026776748856409682, 0.0025505983565874374, 0.002450225320427858, 0.0024302180252742675, 0.0022846871819303984, 0.0022811603567617312, 0.002235898344106969, 0.0022353241219976896, 0.002174940549163257, 0.002172842001541202, 0.001703152909899057, 0.0016891144972010223, 0.0014928272717047884, 0.001489900078188971, 0.0014547587622818596, 0.0014399356887618761, 0.0013742785037855127, 0.0013708976206040836, 0.0012989644467279846, 0.0012955556807300174, 0.0012324377618930244, 0.0012314374454042512, 0.001116504642653094, 0.0011159198663257895, 0.0010164757460048085, 0.0010139526822081637, 0.0009954202949993634, 0.0009937206137094778, 0.0009406013923863131, 0.0009369729527140984, 0.0008242155492360341, 0.0008169934640522876, 0.0008064451093136346, 0.000800668384042331, 0.0007957298180384203, 0.0007913669064748202, 0.0007888628089982089, 0.0007829597160808374, 0.0007811747041427623, 0.000775750610544462, 0.0007679196868386058, 0.0007558869500602147, 0.0007552175683307722, 0.0007523724000679563, 0.000751561839447602, 0.0006899471563200728, 0.0006886156287187624, 0.0006831436844895795, 0.0006830365194300867, 0.0006810578301736499, 0.0006804972290712145, 0.000678136655473766, 0.0006772773450728074, 0.0006691387626509048, 0.0006690345981094134, 0.0006598601638803394, 0.000659467480009892, 0.0006536577310508065, 0.0006533404571640959, 0.0006519199608094359, 0.0006518362365472096, 0.0006391394529931022, 0.0006375139593573859, 0.0006225973178902845, 0.0006223479490806223, 0.0006016629092938798, 0.0006016485169364058, 0.00059108059383897, 0.0005906328337751808, 0.0005610558536827401, 0.0005583923681891269, 0.0005554499330850584, 0.0005552045641644171, 0.0005347159560196126, 0.0005339206460439817, 0.0]


class BreadthCmvClassifier:
    """Classifies CMV status from TCR repertoire breadth (fraction of TCRs
    matching the CMV ECOcluster).

    The call threshold is fixed at the max-F1 point of the breadth ROC curve
    (see cells above). Confidence, though, is computed *per sample*: each
    sample's own breadth value is located on the ROC curve, and the FPR/TPR
    at that point are converted to precision (for a CMV+ call) or NPV (for a
    CMV- call) -- rather than using one fixed confidence for every call.

    All ROC-derived data below (threshold, FPR/TPR curve, and label counts)
    are baked in as constants from the breadth vs. cmv_status ROC analysis on
    the Emerson cohorts, rather than recomputed from data at call time.
    """

    # Call threshold: the max-F1 point on the breadth ROC curve
    breadth_max_f1_threshold = 0.0007679196868386058

    # Count of labeled positive (CMV+) and negative (CMV-) samples underlying the ROC curve
    n_pos_breadth = 340
    n_neg_breadth = 446

    # Thresholds, FPR and TPR from the ROC curve for breadth vs. cmv_status
    # (as returned by sklearn.metrics.roc_curve: thresholds descending, FPR/TPR co-indexed)

    def __init__(self) -> None:
        """Initialize the classifier with the CMV ECOcluster's TCRs, so
        classify_repertoire() can look up matches against them.

        Args:
            pdf_cmv_ecocluster: DataFrame of the CMV ECOcluster, with a "tcr" column
        """
        pdf_cmv_ecocluster = load_cmv_ecocluster_2026()
        self.cmv_ecocluster_tcrs = set(pdf_cmv_ecocluster["tcr"])
 
    def classify_breadth(self, breadth: float) -> tuple:
        """Classify CMV status from a breadth value.

        Args:
            breadth: breadth value for a sample

        Returns:
            (prediction, confidence): prediction is 1 (CMV+) or 0 (CMV-); confidence
                is the estimated probability that this specific prediction is correct,
                derived from the FPR/TPR at the ROC curve point nearest this sample's
                own breadth value (precision if predicted positive, NPV if negative).
        """
        prediction = int(breadth >= self.breadth_max_f1_threshold)

        thresholds = np.asarray(BREADTH_ROC_THRESHOLDS)
        idx = int(np.argmin(np.abs(thresholds - breadth)))
        fpr_at_breadth = BREADTH_ROC_FPR[idx]
        tpr_at_breadth = BREADTH_ROC_TPR[idx]

        tp = tpr_at_breadth * self.n_pos_breadth
        fp = fpr_at_breadth * self.n_neg_breadth
        fn = self.n_pos_breadth - tp
        tn = self.n_neg_breadth - fp

        if prediction == 1:
            confidence = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        else:
            confidence = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
        return prediction, confidence

    def count_matches(self, pdf_onerepertoire_tcrs) -> int:
        """Count the rows in pdf_onerepertoire_tcrs whose "tcr" value is in
        the CMV ECOcluster.

        Args:
            pdf_onerepertoire_tcrs: DataFrame of TCRs for one repertoire, with a "tcr" column

        Returns:
            number of rows whose "tcr" value is in cmv_ecocluster_tcrs
        """
        return int(pdf_onerepertoire_tcrs["tcr"].isin(self.cmv_ecocluster_tcrs).sum())

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
        """Classify CMV status for one repertoire

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
            "cmv_breadth": breadth,
        }
