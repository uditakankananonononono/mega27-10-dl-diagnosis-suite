# MEGA27 Item 10.4-10.5: CNN+GNN Diagnosis Tools (Malaria, Pneumonia)

Owner: task lane MEGA27-10b. Sibling lane 10a owns items 10.1-10.3 (top level).

- **10.4 Malaria parasite diagnosis** - NIH cell-image dataset (27,558 images, official Lister Hill NCBI release, sha-verified download).
- **10.5 Pneumonia diagnosis** - Kermany et al. 2018 chest X-ray dataset (5,856 images, Mendeley Data rscbjbr9sj v2, sha256-verified).

## Architecture
- CNN core (140k params) + RegionGCN hybrid (159k params): CNN feature map pooled into a 4x4 region graph, 8-neighbour adjacency, two GCN layers, mean readout. From-scratch torch, no geometric DL dependency.
- Discovery: confident-learning label-noise census (Northcutt et al. 2021 estimator, implemented from scratch) - named, content-hashed image IDs of suspected label errors, with measured retraining deltas.

## Reproduce
```
pip install torch torchvision scikit-learn pytest --index-url https://download.pytorch.org/whl/cpu
python -m pytest tests/ -q                     # hermetic, no network
python src/run_malaria.py --phase baseline     # then census, then cleaned
python src/run_pneumonia.py --phase baseline   # then census, then cleaned
```
Results land in `results/<disease>/*.json` (committed).

## Status

- 2026-09-26 13:03 IST (user decision): items 10.4 (malaria) and 10.5 (pneumonia) are CLOSED as AI-judged. No blinded human expert review was performed; verification consisted of the converged ChatGPT judge loops (malaria 6 rounds, pneumonia 2 rounds, verbatim on record in `isef_judge/`). Both papers carry this as an explicit limitation ("Limitations and redirected angles", item 4). Final PDFs: malaria 56pp (v7), pneumonia 57pp (v5), uploaded to the Drive results folder; earlier finals superseded.
