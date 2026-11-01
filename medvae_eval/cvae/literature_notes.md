# Image quality metrics: reading notes

Notes on the papers that shaped the quality scores of part B (notebooks 01 to 03).

## Papers on quality metrics

### 1. A study on the adequacy of common IQA measures for medical images (arXiv, 2024)

Image quality assessment (IQA) is central to imaging algorithms, yet almost every quality metric
(PSNR, SSIM, BRISQUE…) was designed and calibrated for natural images (photographs). The paper
argues that applying these formulas blindly to medical images (radiographs, MRI) is risky, since
their visual and statistical properties are fundamentally different. To check whether the usual
metrics are reliable, the authors compare the scores of IQA algorithms with the ratings of medical
experts (radiologists).

Results:
- PSNR and SSIM are disappointing: although they are the industry standards, they correlate rather
  weakly with the experts' opinion.
- No-reference (NR) metrics collapse: BRISQUE, NIQE and PAQ-2-PIQ fail to rate medical images
  consistently with the experts.
- HaarPSI (Haar wavelet-based perceptual similarity) is the clear winner, the most robust metric
  across the medical datasets tested. FSIM, GMSD and MS-SSIM also do well.

Takeaway: no-reference metrics struggle most on X-rays. Two options follow: degrade good ARCADE
images artificially to use full-reference metrics (the paper recommends HaarPSI or MS-SSIM, which
capture the structures of coronary angiograms much better), or compute scores grounded in the
physics of the image (such as `laplacian_var` for sharpness and `rms_contrast` for contrast), since
statistical black boxes fail.

### 2. A no-reference medical image quality assessment method based on automated distortion recognition technology: application to preprocessing in MRI-guided radiotherapy (2024)

The study assesses the quality of MRI used in radiotherapy to track tumours in real time. With no
perfect reference image available, the authors must use no-reference methods (NR-IQA). The main
problem is that classical NR metrics (NIQE, PIQE) were built to judge natural images (landscapes,
faces); they tend to penalise medical images, especially after a preprocessing that enhances the
clinical contrast, which they read as "unnatural".

The authors therefore build their own Quality Index (QI), in several steps:
- **MSCN coefficients**: instead of raw pixels, mean-subtracted contrast-normalised coefficients,
  which normalise contrast and brightness locally and make the image insensitive to global lighting
  variations.
- **Directional analysis**: structures in a medical image (tumour edges, vessels) have directions;
  the algorithm analyses pixel interactions in 4 directions (horizontal, vertical, two diagonals).
- **Modelling and clustering**: the data are fitted with an asymmetric generalised Gaussian
  distribution (AGGD) to extract sharpness and noise parameters, then K-means separates the
  structure, brightness and noise features.
- **Robust aggregation**: the final QI merges these values with the median absolute deviation
  (MAD), which is very robust to outliers.

The tracking algorithm was tested before and after preprocessing. The QI rose sharply after the
enhancement, showing that it captures diagnostic quality. Basic medical metrics (CNR for contrast,
Tenengrad for sharpness) improved slightly but lacked overall sensitivity. NIQE and PIQE dropped:
they judged the enhanced image to be worse, which shows their clinical inadequacy.

Takeaways for ARCADE:
- **Directional analysis**: clinical quality is measured through gradients in specific directions
  (horizontal, vertical, diagonals). This fits coronary angiograms, which are literally a network of
  tubes: directional Sobel or Tenengrad filters, which capture the sharp edges of the arteries, can
  serve as the main quality indicator.
- **Local normalisation (MSCN)**: cardiac X-rays often have strong illumination problems (dark
  borders, light grey centre), which bias a contrast computed over the whole image. Applying the MSCN
  normalisation before computing sharpness or noise metrics (such as `laplacian_var`) flattens the
  illumination and leaves only the sharpness and the true noise.
- **A composite quality index**: the paper does not choose between contrast and noise but merges
  them into a single formula. The VAE needs a single score (the inductive bias), which can be built
  the same way: a contrast score (`rms_contrast`) and a sharpness score (`tenengrad`), combined with
  weights into an "ARCADE quality score".

### 3. Deep learning-based no-reference image quality assessment framework for Cryptosporidium spp. and Giardia spp. (2026)

Instead of signal-processing formulas (like the MSCN method above), the authors use neural networks:
- **Dataset**: 23 good images of parasites, artificially degraded with 4 distortions (Gaussian noise,
  salt-and-pepper, speckle, JPEG compression) at 9 intensity levels.
- **Reference score (MOS)**: 20 people rated these images from 1 to 5.
- **Feature extraction**: the images go through 9 well-known networks pre-trained on ImageNet
  (ResNet-101, EfficientNet-B0, DarkNet…), without retraining; the feature vectors of the last
  convolutional layers are kept.
- **Regression**: classical algorithms (support vector machines) learn the mapping from these vectors
  to the human rating.

Results: ResNet-101 followed by a cubic SVM (PRIQA) gives the best results and predicts quality
almost exactly like a human rater. The 10 classical methods compared (NIQE, BRISQUE, PIQE…) fare
very badly (their Table 11): on speckle noise or JPEG compression, BRISQUE and NIQE have correlations
close to zero, i.e. essentially random ratings.

Takeaways for ARCADE:
- **A frozen feature extractor** makes deep learning usable without training a large network from
  scratch, which would be too long or impossible on the unannotated ARCADE images: pass the
  radiographs through a basic pre-trained PyTorch model (ResNet-18 or ResNet-50) and keep the vector
  of the `AdaptiveAvgPool2d` layer, before the classifier.
- **The inductive bias** could even be this feature vector itself rather than a single 0-100 score:
  a very rich representation of the texture and quality of the image.
- **Validation by artificial degradation** is confirmed as a sound and widely used methodology. To
  rate ARCADE images: select about 50 clean images by hand, apply typical X-ray noise (Poisson or
  Gaussian noise, motion blur), train a small model (SVM or random forest, as in the paper) to
  recover the applied noise level from ResNet features, then use it to rate the rest of the dataset
  automatically. This is approach C of notebook 03.

## Papers on digital subtraction angiography (DSA)

### 4. Reduction of patient motion artifacts in digital subtraction angiography: evaluation of a fast and fully automatic technique

In DSA, the slightest patient motion creates severe artefacts that hamper diagnosis. The classical
correction (manual pixel shifting by the radiologist) was slow (about 12 seconds per image) and,
above all, ineffective against complex motion (rotations, torsions), since it could only shift the
whole image uniformly (translations).

Takeaways:
- **Quality is a local problem**: motion does not degrade the whole image equally; a rotation blurs
  one corner and leaves another intact. The quality score should therefore be computed on small
  regions (patches) rather than averaged over the whole image.
- **Motivation for the report**: current clinical methods struggle with the complex motion
  artefacts of cardiac X-rays; guiding a generative model (MedVAE) with a quality metric is one way
  to move beyond manual deformation algorithms.

### 5. PatchDSA: improving digital subtraction angiography with patch-based phase-matching in natural breathing scenarios

Abdominal DSA suffers from severe motion artefacts when the patient breathes or the organs move
(peristalsis, heartbeat). Existing methods matched the global respiratory phase (by looking at the
diaphragm) to pick the right mask image, but failed because the organs do not all move in sync.

The PatchDSA method works locally:
- **Patch grid**: the mask and contrast images are split into small patches.
- **Best-patch search**: for each patch of the contrast image, the algorithm searches all available
  mask images for the patch with the most similar structure (ZNCC metric).
- **Fine adjustment**: a sub-pixel alignment, then a brightness correction to avoid tiling
  artefacts, before the image is reassembled.

Takeaways:
- **Patch-based quality**: motion and defects on a radiograph (especially cardiac or abdominal) are
  local. Instead of a contrast or sharpness score over the whole ARCADE image, the image can be split
  into a grid (e.g. 4×4 or 8×8 patches) and each patch rated. The encoder could then receive a
  quality map (a small 2D tensor) showing which regions are noisy and which are sharp, rather than a
  single number.
- **Entropy as an additional metric**: the authors use entropy to show that their images have fewer
  artefacts. Entropy measures the disorder of the pixels: a perfect DSA image has only a uniform grey
  background and dark vessels, hence a low entropy; motion artefacts or noise make it rise. Global or
  local entropy (`skimage.measure.shannon_entropy`) is a no-reference metric validated for this type
  of image.
- **Tiling artefacts**: processing images by patches and adjusting their brightness can create
  checkerboard artefacts (their Figure 5). If quality is assessed or images generated patch by patch,
  the patch borders must blend smoothly.
