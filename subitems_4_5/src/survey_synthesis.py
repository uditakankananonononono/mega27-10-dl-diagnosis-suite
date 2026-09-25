#!/usr/bin/env python3
"""Insert per-cluster synthesis paragraphs into the generated lit surveys."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYN = {
'malaria': {
 'Deep learning for diagnosis': (
  "Synthesis. This cluster has grown monotonically since 2018 and is now "
  "dominated by incremental architecture swaps --- YOLO variants, MobileNet "
  "variants, transformer backbones --- evaluated on the same NIH collection "
  "with near-identical protocols. Two observations matter for positioning. "
  "First, reported accuracies cluster tightly around the mid-90s, which is "
  "consistent with our finding of 11.44\\% label noise: a contaminated "
  "ceiling sits under any clean-model gain, and most papers neither measure "
  "nor discuss it. Second, none of the 44 records performs a label-noise "
  "census or publishes per-image flags; our census is, to the extent this "
  "screen can establish, the first for this collection."),
 'Computer-aided detection and segmentation': (
  "Synthesis. The largest cluster covers the detection-and-segmentation "
  "pipeline literature: cell localization, watershed and learned "
  "segmentation, patch extraction, and counting for parasitemia "
  "estimation. These systems operate upstream of the per-cell "
  "classification task this paper studies, and their errors compound into "
  "exactly the kind of label-side inconsistency our census quantifies --- "
  "a cell cropped at the boundary of two erythrocytes is a natural "
  "mislabel candidate. The cluster's scale (121 records) against the "
  "label-noise cluster's two records is the literature's blind spot in "
  "one ratio."),
 'Classical machine learning': (
  "Synthesis. The classical cluster is small but contains the collection's "
  "reference study (Rajaraman et al., PMID 29682411), whose cell-level "
  "94.0\\% is the benchmark this paper beats. Its feature-extractor "
  "framing --- pretrained networks as fixed embedders with shallow "
  "classifiers --- is the regime our compact end-to-end models are "
  "compared against in the benchmark chapter."),
 'Dataset and benchmark studies': (
  "Synthesis. A single record studies the benchmark itself. That the "
  "standard collection has one benchmark-analysis paper and zero "
  "label-noise audits after eight years of use is the gap this paper's "
  "census and verification chapters address."),
 'Clinical and diagnostic context': (
  "Synthesis. The clinical cluster frames deployment: smartphone "
  "microscopy, field diagnostics, point-of-care triage. Its constraint "
  "profile --- cheap hardware, no connectivity, operator variability --- "
  "is precisely the regime our 2-CPU-budget models are designed for, and "
  "the dxtool engineering chapter borrows its requirements from this "
  "literature rather than from datacenter assumptions."),
 'Label noise and robustness': (
  "Synthesis. Two records, and neither performs a label-noise census of "
  "the standard collection; both use `robust' in the architectural sense. "
  "The confident-learning literature (Northcutt et al.) has not been "
  "applied to this benchmark in any record our screen found --- the "
  "novelty claim of the census rests on this absence, stated with the "
  "screen's own limits (title-keyword clustering, PubMed coverage)."),
},
'pneumonia': {
 'Deep learning for diagnosis': (
  "Synthesis. The dominant cluster reproduces the Kermany transfer-learning "
  "recipe --- ImageNet-initialized backbones fine-tuned on the official "
  "split --- with reported accuracies mostly between 90 and 96\\%. Two "
  "positioning observations. First, nearly all of these works train at "
  "datacenter scale with input resolutions of 224px or above; our compact "
  "128px models make the resource-constrained regime explicit and show "
  "where its ceiling sits. Second, none of the 112 records audits the "
  "training labels, despite the collection being assembled by automated "
  "report parsing --- a known contamination path our 21.15\\% census "
  "estimate quantifies directly."),
 'Computer-aided detection and segmentation': (
  "Synthesis. The CAD cluster covers lung-field segmentation, ROI "
  "extraction, and multi-view fusion. These methods are orthogonal to the "
  "classification head but shape input quality; our flagged-film forensics "
  "(blurry, low-contrast acquisitions concentrate in the flagged set) is "
  "the label-noise shadow of this cluster's concerns."),
 'Classical machine learning': (
  "Synthesis. Hand-crafted texture and shape features with SVM or ensemble "
  "classifiers; mostly pre-2018 or small-scale. Useful as a floor: our "
  "from-scratch compact CNN exceeds this regime by a wide margin, as "
  "expected, and the comparison is included for completeness in the "
  "benchmark framing."),
 'Dataset and benchmark studies': (
  "Synthesis. Seven records examine the collection or its splits, "
  "including discussions of patient-level leakage when the official split "
  "is ignored. Our protocol preserves the official patient-level split "
  "exactly for this reason, and the TorchXRayVision head-to-head is "
  "conducted on the identical films."),
 'Clinical and diagnostic context': (
  "Synthesis. The clinical cluster covers paediatric triage, antibiotic "
  "stewardship, and radiologist-workload framing. Its operating point "
  "preference --- high sensitivity even at specificity cost --- matches "
  "the behaviour of our class-weighted tuned configuration, which we "
  "report with its full sensitivity/specificity trade-off rather than a "
  "single accuracy."),
 'Label noise and robustness': (
  "Synthesis. One record, architectural-robustness sense only. No record "
  "in the screen performs a confident-learning census of ChestXRay2017; "
  "the pneumonia label-noise literature concentrates on the much larger "
  "NIH ChestX-ray14 (a different collection with its own mining-noise "
  "discussion), leaving this paediatric benchmark unaudited."),
}}

POS = {
'malaria': (
 "\\subsection*{Positioning of this work}\n"
 "Against 197 screened records, this study is (to the screen's knowledge) "
 "the first label-noise census of the NIH malaria collection, the first "
 "publication of per-image falsifiable flags for it, and --- at "
 "96.08\\% cell-level accuracy with a 140k-parameter model --- above the "
 "source-verified 94.0\\% reference at two orders of magnitude fewer "
 "parameters than the reference's feature-extractor regime. The screen's "
 "limits: title-keyword clustering, PubMed coverage as of the audit date, "
 "and no full-text screening of every record.\n"),
'pneumonia': (
 "\\subsection*{Positioning of this work}\n"
 "Against 150 screened records, this study contributes the first "
 "confident-learning census of the ChestXRay2017 training set (21.15\\% "
 "estimated noise, 155 published flags), a demonstration that cleaning "
 "improves the strongest configuration by +5.9 points, and a same-split "
 "head-to-head against the strongest runnable external tool "
 "(TorchXRayVision DenseNet-121), which our models beat decisively on the "
 "identical 624 official test films. The screen's limits: title-keyword "
 "clustering and PubMed coverage as of the audit date.\n"),
}

for d in ('malaria', 'pneumonia'):
    for side in ('paper', f'paper_{d}'):
        p = os.path.join(ROOT, side, f'lit_survey_{d}.tex')
        if not os.path.exists(p): continue
        s = open(p).read()
        for cluster, syn in SYN[d].items():
            marker = f"\\subsection*{{{cluster}"
            i = s.find(marker)
            if i < 0:
                print('MISS', d, cluster); continue
            j = s.find('\\par', i)
            s = s[:j+4] + '\n\n' + syn + '\n' + s[j+4:]
        j = s.find('\\subsection*{Complete record list')
        s = s[:j] + POS[d] + '\n' + s[j:]
        open(p, 'w').write(s)
        print('synthesis injected:', p, len(s))
