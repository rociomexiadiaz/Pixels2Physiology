# Pixels2Physiology

A cascaded deep learning pipeline for automated digitisation of degraded ECG images into 12-lead time-series signals. Developed as an entry for the **[PhysioNet 2025 ECG Image Digitisation Challenge](https://physionet.org/content/ecg-image-kit/1.0.1/)**.

> **"From Pixels to Physiology: A Two-Stage U-Net Approach for ECG Signal Recovery"**  
> *Rocio Mexia Diaz — r.mexiadiaz@hotmail.com*

---

## Overview

Billions of ECG records exist only as scanned images or physical printouts, making them incompatible with modern machine learning systems that require time-series data. This project addresses that gap with a two-stage deep learning pipeline that can handle severely degraded ECG images — including mould, water damage, rotation, and mobile photography distortions — and recover the underlying 12-lead waveform signal.

### Pipeline

```
Input Image → Preprocessing → UNet-1 (Document Segmentation) → UNet-2 (Trace Extraction) → Time-Series Output
```

1. **Preprocessing** — Pads and resizes all images to 1700×2200 pixels, preserving aspect ratio, then converts to grayscale.
2. **UNet-1: Document Segmentation** — A standard 5-level U-Net detects and crops the ECG document from its background, correcting for rotation and perspective distortion using ORB feature matching for ground-truth mask generation.
3. **UNet-2: ECG Trace Extraction** — A Residual Attention U-Net (ResAttentionUNet) with CBAM modules removes the background grid and isolates the ECG trace as a binary mask. Takes dual-channel input: grayscale + FFT magnitude.
4. **Post-processing & Digitisation** — Morphological operations clean the mask, a column-wise peak-detection algorithm extracts voltage values, and cubic spline interpolation resamples the signal to the target length.

---

## Results

Evaluated across all 12 standard ECG leads on the PhysioNet 2025 challenge dataset:

| Metric | Score |
|---|---|
| Mean RMSE | **0.233 mV** |
| Mean Cross-Correlation | **0.328** |

Performance per lead:

| Lead | RMSE (mV) | Cross-Correlation |
|------|-----------|-------------------|
| I    | 0.210     | 0.361             |
| II   | 0.206     | 0.330             |
| III  | 0.154     | 0.332             |
| aVR  | 0.230     | 0.300             |
| aVL  | 0.160     | 0.360             |
| aVF  | 0.146     | 0.302             |
| V1   | 0.225     | 0.359             |
| V2   | 0.319     | 0.356             |
| V3   | 0.302     | 0.347             |
| V4   | 0.326     | 0.291             |
| V5   | 0.292     | 0.299             |
| V6   | 0.226     | 0.301             |

### Ablation Study — Effect of FFT Input Channel

Removing the FFT magnitude channel from ResAttentionUNet inputs revealed its critical role in temporal alignment:

| Configuration | Mean RMSE | Mean Cross-Correlation |
|---|---|---|
| Grayscale + FFT (full model) | 0.233 mV | 0.328 |
| Grayscale only (ablation) | **0.177 mV** | **−0.0008** |

The FFT channel is essential for temporal coherence; without it, amplitude accuracy improves but phase alignment collapses entirely.

---

## Architecture Details

### UNet-1 — Document Segmentation

- Standard U-Net with 5 encoder-decoder levels
- Input: 512×656 downsampled grayscale images
- Encoder: double 3×3 convolutions + ReLU, max-pooling with stride 2
- Decoder: transposed convolutions + skip connections
- Loss: Binary Cross-Entropy + Dice (0.5/0.5 weighting)
- Optimiser: AdamW, lr=0.001, 10 epochs, batch size 8
- Ground truth generated via **ORB keypoint matching + RANSAC homography** between high-quality and degraded image pairs

### UNet-2 — ResAttentionUNet (Trace Extraction)

- Residual blocks with identity skip connections
- **CBAM (Convolutional Block Attention Module)** at each decoder skip connection — channel attention + spatial attention
- Input: 2-channel (grayscale + FFT magnitude spectrogram), patches of 384×2192
- Loss: DiceFocalLoss (Dice + Focal, 0.5/0.5 weighting)
- Optimiser: Adam, lr=0.0001, 40 epochs, batch size 2

## Data

This project uses the **PhysioNet 2025 ECG Image Digitisation Challenge** dataset, available on Kaggle:

- [physionet-ecg-image-digitization](https://www.kaggle.com/competitions/physionet-ecg-image-digitization)

The dataset contains ECG images paired with ground-truth 12-lead time-series across eight degradation categories.


## Limitations

- Signals with very high amplitude deflections that exceed strip height boundaries are clipped, causing peak detection errors.
- The fixed 4-strip layout assumes standard 12-lead ECG formatting; alternative vendor templates would require adaptation.

---

## Future Work

- Adaptive strip cropping based on detected trace positions
- Multi-task learning combining amplitude and phase objectives
- Extension to variable-duration recordings beyond the 10-second standard
- Clinical validation studies assessing diagnostic equivalence

---

## Citation

If you use this work, please cite:

```
Mexia Diaz, R. (2025). From Pixels to Physiology: A Two-Stage U-Net Approach for ECG Signal Recovery.
PhysioNet 2025 ECG Image Digitisation Challenge.
Contact: r.mexiadiaz@hotmail.com
```

---

## License

This project was developed for the PhysioNet 2025 Challenge. Please refer to the [PhysioNet data use agreement](https://physionet.org/about/data-use/) for data licensing terms.
