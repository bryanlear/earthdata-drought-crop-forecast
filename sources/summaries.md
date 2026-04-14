* **Remote sensing**: Employs a wide range of sensor technologies to acquire data for earth observation and monitoring through satellites and aircrafts. Data may differ in terms of Ground Sample Distance (GSD) nased on sensor technology and altitude. 
[Noman et al., 2024](https://arxiv.org/html/2403.05419v1)


8 **Ground sample distance**: Real-world size on the ground represented by one pixel in an image

Authors argue grouping alone of channels by resolution not equivalent to exploting multi scale info:
  * SatMAE framework resizese input imagery to fixed resolution and performs reconstruction only at that given scale.

| Feature | SatMAE Approach | SatMAE++ Approach |
| :--- | :--- | :--- |
| **Scale Handling** | **Input-level**: Groups channels by their native GSD to allow simultaneous processing. | **Objective-level**: Reconstructs the image at multiple upsampled scale levels. |
| **Positional Encodings** | Uses spectral encodings to identify band groups. | Uses standard encodings but relies on multi-scale reconstruction to learn scale-robust features. |
| **Reconstruction** | Reconstructs masked patches at a single, fixed input resolution. | Uses convolution-based upsample blocks to reconstruct images at two or three different scale levels

Each attention block computes interactions between token susing *queries, keys, and values*.

* **Query**: What this patch is looking for
* **Key**: What another patch offers
* **Value**: Information that patch contributes

After attention, each token becomes a **weighted combination** of information from the whole image
Dynamically decides what matters based on input image.

In remote sensing/geospatial images, attention is useful because relevant patterns may span large regions:
  * Irrigation structure
  * Drought patterns
  * Field boundaries
  * Flood extent
  * Multi scale land cover organization

### MAE architecture

1. Patchification and High-ratio masking:
   1. Input image is divided into grid of non-operlappping, fixed patches
   2. ~75% randomly removed/masked $\rightarrow$ elimination localized redundancy
2. Deep Encoder:
   1. ~25% remaining visible patches are processed by encoder
      1. ViT scales quadratically with sequence length so discarding ~75% of data at entry yields 3x-4x compute and memory reduction during training
3. Lightweight decoder:
   1. Reconstruction of full list of tokens from latent representation of visible patches
   2. Encoded visible patches are merged with shared, learnable mask tokens that represent missing patches.
   3. Positional embeddings are added to all tokens to retain spatial awareness. Full list is passed though s smaller, shallower transformer decoder to predict pixel values.
4. Reconstruction loss:
   1. Output layer projects decoder's results to match original pixel countrs
   2. Loss function is computed only on the masked patches. **Model's success** is evaluated by capacity to fill in the blanks, not recreate what it was already shown

---

BERT maskls 10-15% of image but model can cheat by looking at immediate adjacent pixel to guess what's missing.

MAE: By masking 75% model is forced to:
* Learn global semantics
* Identify high-level latents
* Improvel generalization

$$\mathcal{L} = \frac{1}{N} \sum_{i \in \text{Masked}} || \hat{x}_i - x_i ||^2$$

Loss is calculated only on the 75% of patches that were hidden. THere is no mathematical reward for just remembering the patches the model could aready see.

**SatMAE**: Framework created by combining several distinct mathematical methods into a pipeline to optimize learning from termporal and multi-spectral satellite imagery.

