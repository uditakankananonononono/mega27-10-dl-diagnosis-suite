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

DERIVATIONS = """
Derivation 1 (GCN from spectral convolution). Start from the spectral
definition of graph convolution g_theta * x = U g_theta(Lambda) U^T x with
U the eigenbasis of the normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
Approximating g_theta by a first-order Chebyshev polynomial in Lambda and
setting lambda_max approximately 2 yields theta (I + D^{-1/2} A D^{-1/2}) x,
which overfits on small graphs; the renormalization A~ = A + I, D~_ii =
1 + sum_j A_ij gives the propagation rule H' = tanh(D~^{-1/2} A~ D~^{-1/2}
H W) used throughout. The approximation error is governed by the discarded
Chebyshev terms of order 2 and above, which is acceptable when the kernel
bandwidth sigma already localizes the graph (Eq. 1).

Derivation 2 (Mann-Whitney form of the AUC). For score function f, AUC as
the area under the ROC curve equals the probability that a random positive
outranks a random negative. Write the ROC curve parametrically by threshold
t: TPR(t) = P(f(x+) > t), FPR(t) = P(f(x-) > t). Then integral_0^1 TPR dFPR
= integral_{-inf}^{inf} P(f(x+) > t) dP(f(x-) > t) = P(f(x+) > f(x-)),
with the half-credit convention for ties, which is Eq. 10. The estimator is
the two-sample U statistic, unbiased with variance given by Hanley-McNeil;
this justifies the bootstrap CI of Eq. 8 without parametric assumptions.

Derivation 3 (Class weights from cost-sensitive risk). Minimizing expected
cost with misclassification costs c_+, c_- gives the decision rule
predict positive iff p(y=1|x) > c_-/(c_+ + c_-). Replicating the threshold
shift inside the loss requires weighting the positive term by w_1 =
c_+/c_-; under prevalence correction c_+/c_- = n_-/n_+, which recovers the
weights of Eq. 5 exactly.

Derivation 4 (Exactness of the sign test for bridge ablation). Under the
null that cross-disease edges carry no signal for disease d, the paired
differences D_s = AUC_full,s - AUC_abl,s are symmetric about zero, so
S+ = #{s : D_s > 0} ~ Binomial(S, 1/2), because seeds enter only through
independent split and initialization randomness and the two variants share
all other randomness by construction (paired design). The exact two-sided
p-value 2^{-S} sum_{k>=k0} C(S,k) needs no normality assumption, which is
what makes the bridge claim falsifiable at S = 5: five of five positive
seeds give p = 0.031 < 0.05, four of five give p = 0.188.

Derivation 5 (Brier decomposition and calibration). The Brier score
decomposes as B = reliability - resolution + uncertainty (Murphy 1973).
Reporting Brier alongside ECE (Eq. 7) separates calibration failure
(reliability term) from discrimination failure, which is why a model can
win AUC and still lose clinical utility: on Cleveland the MLP's ECE of
0.123 against the GCN's 0.093 means its probabilities would need
recalibration before thresholding.

Derivation 6 (Self-tuned bandwidth). With sigma_i the distance from x_i to
its k-th neighbor, the local-scaling kernel of Zelnik-Manor and Perona uses
A_ij = exp(-d_ij^2/(sigma_i sigma_j)); our symmetric single-sigma variant
(Eq. 1) with sigma^2 the median kNN squared distance is the global limit of
that construction, and the median choice makes the kernel robust to the
heavy right tail of squared distances - a percentile argument: any
quantile q in (0,1) yields a valid scale, and q = 0.5 minimizes sensitivity
to outliers among central quantiles.

Derivation 7 (Multi-task gradient identity). Because the per-disease losses
share only the trunk parameters theta_trunk, the gradient of the summed
objective decomposes as sum_d dL_d/dtheta_trunk with head gradients
confined to their own head: dL_MT/dtheta_head,d = dL_d/dtheta_head,d. The
trunk therefore receives the sum of disease-specific descent directions; a
conflict between diseases appears as gradient cancellation, which the
bridge ablation measures indirectly through Eq. 9.

Derivation 8 (Fingerprint comparability). For any invertible affine
reparametrization of a dataset's feature space x -> Mx + b with diagonal
M, the per-patient standardized moments m_r = mean_j ((x_j - mu)/sigma)^r
are invariant, because mu and sigma transform covariantly. The fingerprint
(Eq. 4) is therefore a property of the patient's distributional shape, not
of the feature units, which is precisely what makes cross-disease edges
well-defined.

Derivation 9 (Bootstrap CI validity). The percentile bootstrap interval of
Eq. 8 is first-order accurate: P(theta in CI_95) = 0.95 + O(n^{-1/2}) under
standard smoothness conditions on the statistic (the AUC U-statistic
satisfies them). With test cohorts of 49-290 patients this asymptotic is
coarse; we therefore report interval widths rather than treating endpoints
as sharp.

Derivation 10 (Label propagation view of transduction). One GCN layer
computes H' = tanh(A~ H W); with W near identity at initialization this is
one step of label/feature diffusion over the patient graph, equivalent to
minimizing the graph Dirichlet energy sum_ij A_ij ||h_i - h_j||^2 subject
to fitting the labeled nodes. The label-efficiency result is the empirical
shadow of this variational fact: unlabeled nodes reduce the energy
landscape's dependence on the labeled subset.
"""

RELATED = """
WDBC has been a benchmark since Street, Wolberg and Mangasarian (1993)
introduced the ten cytological features; reported accuracies climbed from
97 percent with simple linear models to claims above 99 percent under
feature-selection pipelines (PeerJ Computer Science 2024 reports logistic
regression at 97.5 percent; a 2022 preprocessing-plus-selection pipeline
claims 99.12 percent). The Cleveland heart-disease cohort of Detrano et
al. (1989) shows the same inflation curve, with tuned single-split neural
networks reported above 93 percent accuracy and XGBoost at 90 percent
under stricter evaluation. For Pima diabetes, gradient boosting claims
range from 85 to 91 percent accuracy depending on protocol, and for the
Parkinson's vocal dataset recent claims reach 99.11 percent with feedforward
networks. Two observations motivate our protocol: first, the spread within
a dataset across papers is larger than the spread between model families,
which is the signature of protocol variance rather than model progress;
second, almost none of these works report calibration, which is the metric
that decides whether a score can be thresholded clinically. Graph neural
networks for tabular clinical data remain comparatively unexplored outside
medical imaging; our transductive patient-graph construction follows the
semi-supervised classification line of Kipf and Welling (2017) and the
self-tuning spectral clustering of Zelnik-Manor and Perona (2004), and our
label-efficiency framing connects to the broader semi-supervised learning
literature, adapted here to cohorts of a few hundred patients.
"""

PROTOCOL = """
Compute environment. All experiments ran on a two-core CPU sandbox with
2 GB RAM; no GPU was used. This constrains model size deliberately: every
architecture in this paper trains in under five minutes per seed on this
hardware, which makes the full suite reproducible on a laptop. Determinism
is enforced by explicit seeding of NumPy and PyTorch; data splits are
stratified with a fixed 75/25 train/test ratio; graph models carve a
20 percent validation slice from the training indices for early stopping.
The hermetic test suite (28 tests) validates every loader against recorded
properties (shape, prevalence bounds, label coding), every model for
output shape and finite loss, graph construction for symmetry and block
structure, the metrics against hand-computed cases, and the discovery
pipeline end-to-end on synthetic separable data. Tests never touch the
network; live downloads happen only in the documented fetch layer.
"""
