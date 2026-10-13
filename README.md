# Ultrasonic A-scan Defect Detection with TimeDART and Adapters

Deep learning models for binary defect detection in ultrasonic A-scan signals for nondestructive testing.

The primary model is **CAUSAL_ADAPTER**: parallel CNN patch embeddings (kernels 3 and 5), TimeDART-style self-supervised pretraining, and lightweight adapters added during supervised fine-tuning.

[Open the GPU notebook in Colab](https://colab.research.google.com/github/sdho1026/anomaly_detection/blob/main/notebooks/TimeDART_CNN_CausalAdapter_Colab.ipynb)

## Architecture

- Input: one complete 50,001-point voltage waveform per recording.
- Preprocessing: FIR polyphase resampling by a factor of 20, producing 2,501 points; per-record instance normalization.
- Patches: length 32, stride 32, 79 tokens.
- Patch embedding: independent Conv1D branches with kernels 3 and 5, concatenation, and a linear projection to 64 dimensions.
- Backbone: two causal Transformer encoder layers, four attention heads, feed-forward dimension 128.
- Self-supervised pretraining: shifted clean patch context, patch diffusion, and a self-only denoising decoder. No adapters are used in pretraining.
- Fine-tuning: a residual 64 → 8 → 64 adapter follows each encoder layer. The CNN/Transformer backbone is frozen; only the adapters and classification head are updated.
- Classification head: eight ordered time-bin means plus global max pooling, followed by a linear binary classifier.
- CAUSAL_ADAPTER parameters: 102,817 total; 2,769 trainable in fine-tuning.

The original experiment also includes **BI_FULL** and **BI_ADAPTER** comparison arms. They remain in the source to preserve the experiment that was actually executed. All three arms share the same pretraining checkpoint within each fold, and no weights are shared between folds.

## Run on Colab

1. Open the notebook above and select a GPU runtime.
2. Run the cells in order.
3. Upload the original dataset ZIP containing 126 normal and 18 defect DAT recordings.
4. Run pretraining and fine-tuning, then the separate test cell.
5. Read the fold-average metrics and download the result archive.

The dataset is not included. The public repository excludes recording filenames, waveform hashes, and saved fold assignments. Run `prepare_local_metadata.py` on your own dataset ZIP to generate those files locally. Generated folds are not guaranteed to match the earlier saved experimental folds, so this public version does not claim to reproduce an earlier AUROC.

## Run locally

Install dependencies in an environment with the appropriate CPU or CUDA version of PyTorch:

```bash
pip install -r requirements.txt
python timedart_cnn_adapter/prepare_local_metadata.py --data /path/to/data.zip
python timedart_cnn_adapter/checks.py
python timedart_cnn_adapter/cv.py train --data /path/to/data.zip --out /path/to/new_run --device cuda
python timedart_cnn_adapter/cv.py test --data /path/to/data.zip --out /path/to/new_run --device cuda
```

Use a new output directory when the source, configuration, or folds change. Reusing the same unchanged run supports resumption.

## Evaluation protocol

This is record-level stratified **5-fold outer evaluation**, with a separate stratified inner validation holdout in each fold. It is not a single fixed test set plus development-only cross-validation.

- Each recording appears in an outer test fold exactly once.
- Pretraining and fine-tuning use that fold's training recordings.
- Inner validation selects checkpoints and the classification threshold.
- The training and selection stages complete before outer test inference.
- Reported ratios are unweighted means across the five outer folds, with sample standard deviation where provided. Confusion counts can be summed across folds.
- Average Precision is reported as PR-AUC (AP); it is not trapezoidal PR-curve integration.
- No pooled OOF AUROC is reported.

The recordings and earlier test results had already been explored during development. These results are internal resampling results, not an untouched external test or evidence of generalization to new specimens or acquisition sessions. Synthetic demonstration metrics from chat are not included.

## Files

- `notebooks/TimeDART_CNN_CausalAdapter_Colab.ipynb`: clean GPU notebook with embedded source and no saved execution outputs.
- `timedart_cnn_adapter/model.py`: CNN embedding, causal Transformer, diffusion pretrainer, adapters, and classifier.
- `timedart_cnn_adapter/run.py`: pretraining, fine-tuning, checkpointing, and evaluation.
- `timedart_cnn_adapter/cv.py`: original fold handling and aggregate reporting.
- `timedart_cnn_adapter/checks.py`: mask isolation, adapter gradients, frozen-weight invariance, pooling, and checkpoint checks.
- `timedart_cnn_adapter/config.json`: original training settings.
- `timedart_cnn_adapter/data.py`: original recording-level data adapter.
- `timedart_cnn_adapter/prepare_local_metadata.py`: create private manifests and reproducible folds locally; generated files are ignored by Git.
- `timedart_cnn_adapter/README_KO.md`: detailed original Korean experiment documentation.

The original model, data adapter, training, cross-validation, structural checks, and configuration files were extracted from a previously executed notebook and preserved unchanged. Private recording manifests and fold assignments were excluded. The local metadata generator, notebook interface, and documentation were prepared for this public repository. The original archive CRC checks were verified; training and PyTorch checks were not rerun during this upload.

## References

- [TimeDART official repository](https://github.com/ustc-time-series/TimeDART)
- [TimeDART paper](https://arxiv.org/abs/2410.05711)
- [Parameter-Efficient Transfer Learning for NLP (Houlsby et al., 2019)](https://proceedings.mlr.press/v97/houlsby19a.html)

This is a compact NDT adaptation inspired by the papers, not an exact reproduction of the official TimeDART or Houlsby implementations.
