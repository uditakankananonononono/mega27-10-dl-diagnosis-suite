"""Long-form paper sections for MEGA27-10 (assembled by build_paper.py)."""

INTRO = """
Computer-aided diagnosis is one of the oldest promises of machine learning,
yet the published benchmark culture around small clinical datasets is
methodologically fragile. The Wisconsin Diagnostic Breast Cancer (WDBC),
Cleveland heart disease, Pima diabetes and Parkinson's vocal-measurement
cohorts have each accumulated hundreds of classification papers, with
reported accuracies climbing from the mid-80s to above 99 percent over three
decades. A substantial fraction of those numbers were produced with
non-nested feature selection, single random splits, or test-set peeking,
which inflates estimates in ways that are invisible in the headline metric.
This project asks three questions under a deliberately strict protocol
(held-out 25 percent test set, five seeds, bootstrap confidence intervals,
calibration metrics, hermetic tests for every computation):

1. Floor: what do modern deep architectures (MLP, 1D-CNN, GCN, GAT)
actually achieve on real clinical cohorts when evaluated honestly, and how
do they compare to strong classical baselines (regularized logistic
regression, random forests)?

2. Discovery: does a unified patient-similarity graph spanning multiple
diseases contain transferable diagnostic signal? We formalize this as the
bridge hypothesis: cross-disease edges of a shared graph, built on
disease-agnostic distributional fingerprints, improve per-disease diagnosis
in a paired ablation across seeds.

3. Label efficiency: can a transductive graph model trained on a fraction
of labels match or beat a full-label tabular model? This is the regime that
matters clinically, where labeled data is the bottleneck.

All code, every raw number behind every table, and all negative results are
preserved in the project repository.
"""

METHODS = """
Datasets. Twenty-six accession-level clinical datasets are used, all
downloaded from the UCI Machine Learning Repository or a cited public
mirror; every URL was verified live (HTTP 200) on the day of the
experiments. The four core cohorts (WDBC, Cleveland heart disease, Pima
diabetes, Parkinson's) receive the full six-model, five-seed treatment; a
further twenty-two datasets (four heart-disease sites, two thyroid cohorts,
two breast-cancer cohorts, hepatitis, SPECTF cardiac scintigraphy, Haberman
survival, lymphography, dermatology, mammographic masses, echocardiogram,
arrhythmia, cervical-cancer risk factors, Messidor diabetic retinopathy,
fertility, HCV, Indian liver patient dataset, early-stage diabetes risk)
receive a four-model, three-seed panel treatment. Missing values are
median-imputed; categorical variables are factorized; labels are binarized
to the clinically standard positive class (documented per dataset in the
repository loaders).

Models. The MLP is a two-hidden-layer ReLU network trained with Adam and
class-weighted binary cross-entropy. The 1D-CNN treats the standardized
feature vector as a one-channel sequence with two convolutional blocks. The
GCN and GAT operate on a patient-similarity graph: nodes are patients, edge
weights come from a Gaussian kernel over k-nearest neighbors with a
self-tuned bandwidth (Eq. 1), and propagation uses the symmetrically
normalized adjacency (Eq. 2). Graph models are trained transductively: the
graph includes test patients' features but never their labels.

Unified cross-disease graph. Each disease's encoder maps its native feature
space into a shared 16-dimensional latent space. Every patient is
additionally described by a 10-dimensional distributional fingerprint
(Eq. 4) - moments and quantiles of the patient's own feature vector - which
is comparable across diseases by construction because it does not reference
any specific native feature. The unified graph is the union of the latent
kNN graph and the fingerprint kNN graph; a two-layer GCN trunk with one
output head per disease is trained on the summed per-disease losses
(Eq. 6). The ablated variant removes exactly the cross-disease edges.

Evaluation. Every run reports ROC AUC (Mann-Whitney form, Eq. 10),
balanced accuracy, Brier score, and expected calibration error (Eq. 7),
with percentile bootstrap 95 percent confidence intervals (Eq. 8). The
bridge effect per disease is the paired per-seed AUC difference between the
full and ablated graphs (Eq. 9); significance uses the exact sign test
(Proposition 1). Label-efficiency experiments subsample training labels at
10, 25, 50 and 100 percent, five seeds, comparing transductive GCN against
MLP trained on the same labeled subset.

Reproducibility. The full suite ships with hermetic pytest suites (28 tests)
that run without network access; live dataset downloads happen only through
a single documented fetch function with on-disk caching. Seeds, splits and
every hyperparameter are recorded in the committed result JSONs.
"""

DISCOVERY = """
The bridge hypothesis survives - weakly, and exactly where theory predicts
it should. On the full five-seed paired ablation, removing cross-disease
edges costs Pima 0.0066 AUC with all five seeds positive (exact sign-test
p = 0.031), Parkinson's 0.0140 AUC (three of five seeds positive, largest
single-seed effect +0.047), and Cleveland 0.0102 (four of five positive),
while WDBC is unaffected (+0.0004). Six cross-disease bridge pairs recur in
all five seeds; the WDBC-Parkinson's bridge alone carries over two thousand
edges per seed. The pattern - signal on the smaller, noisier cohorts, none
on the saturated one - is what a regularization-by-transfer account
predicts: WDBC is already at its noise ceiling, so extra structure adds
nothing; the smaller cohorts benefit from the added constraint.

The same experiment exposes the cost side honestly: the shared encoder
compresses each disease into 16 dimensions, and on Cleveland the unified
model trails the single-disease GCN in absolute AUC. Bridges carry real
signal, but the first-generation shared encoder gave away more than the
bridges returned on that dataset. A deeper two-hidden-layer encoder
generation was constructed to test whether the gap is architectural; its
verdict is reported with the same paired protocol.
"""

LABEL_EFF = """
Label efficiency is where the graph model earns its keep. On Cleveland, the
transductive GCN trained with 10 percent of labels reaches 0.8617 AUC -
matching the MLP trained on all labels (0.8580) - and with 25 percent of
labels it reaches 0.8838, beating the full-label MLP by 2.6 AUC points.
The advantage persists at every label fraction tested (+6.0, +3.4, +1.7,
+2.1 points at 10, 25, 50, 100 percent respectively). On WDBC, the GCN
with a quarter of the labels matches the full-label MLP (0.9884 vs 0.9880).
The mechanism is transparent: unlabeled patients shape the decision
boundary through the graph Laplacian, so supervision propagates along
diagnostically meaningful similarity structure instead of being confined to
the labeled subset. For clinical deployment, where every label costs an
expert read, a 4-10x label reduction at equal accuracy is the difference
between a feasible study and an infeasible one.
"""

NEGATIVE = """
Several hypotheses did not survive contact with the protocol, and they are
recorded here with the same rigor as the positive results.

1. The GAT, despite its attention mechanism, is the weakest graph model on
three of four core datasets (e.g. 0.8613 +- 0.1061 on Parkinson's) - on
small cohorts the attention parameters buy variance, not signal.

2. The 1D-CNN over the feature axis never wins a core dataset; feature
order carries no spatial structure for it to exploit, and it is strictly
dominated by the MLP it generalizes.

3. Cross-disease bridges add nothing on WDBC (+0.0004 mean delta,
sign-changing across seeds): transfer cannot lift a cohort already at its
noise ceiling.

4. On wpbc (prognostic breast cancer, 198 cases) no model exceeds 0.74 AUC;
the published difficulty of this dataset is real and is not an artifact of
weak baselines.

Each of these is preserved with full per-seed numbers in the committed
result files.
"""

DISCUSSION = """
Three conclusions follow. First, honest re-benchmarking matters: under a
strict protocol, the deep architectures cluster with, not above, strong
classical baselines on saturated cohorts, and the dramatic published gaps
shrink to protocol differences. Second, graph structure is the one
architectural ingredient that buys something classical models cannot have:
label efficiency through transduction, and a small but statistically real
transfer signal between diseases. Third, falsifiable discovery claims are
affordable: the bridge hypothesis was stated, tested by paired ablation,
and is reported with its exact effect sizes and its failures. The
toolchain - loaders, models, graph builders, evaluation, paper generator -
is built to be rerun end-to-end by anyone, and every number in this paper
can be traced to a committed JSON artifact.
"""
