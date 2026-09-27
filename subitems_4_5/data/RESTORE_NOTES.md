# Post-wipe restore record (2026-09-27 ~4:05-4:41 PM IST)
- Sandbox wiped ~3:58 PM. Raw data is gitignored; restored from sources.
- Malaria: NIH Lister Hill cell_images.zip re-download (353,452,851 bytes, zip integrity OK);
  malaria48_linear + malaria48_lanczos rebuilt via src/pretensor_kernel.py. INTEGRITY PROOF:
  linear fold2 re-trained post-wipe produced bit-exact identical metrics to pre-wipe
  (acc 0.9254529254529255, AUC 0.9910031135957058) - rebuilt tensor equals original.
- Pneumonia: HF mirror ahulikal/chest-xray-pneumonia-mirror (PREREG_IMPROVEMENT_LADDER.md-sanctioned),
  chest_xray.zip sha256 42f8a8bc44c252102c02f0f2cf3de10487f948ccdba9e1b86abc6ff52e499053;
  official counts verified (train 5,216 / val 16 / test 624). Restored to ~/work/data/kermany_cxr/.
- L2C phase-3 ckpt restored from Drive science-artifacts parts, md5 e48130d200c7290463b139035756bc8d verified.
