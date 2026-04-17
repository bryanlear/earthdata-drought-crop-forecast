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

---

### SatMAE

1. SatMAE pretraining $\rightarrow$ Enconder becomes specalized tot he geometry, noise structure, and spatial statistics of soil moisture imagery

$$x \in \mathbf{R}^{C*H*W}$$

where:
* $C =$ Number of channels
* $H, W =$ spatial dimensions

image is split into non-overlapping patches of size $PxP$:

$$N=\frac{H}{P} \frac{W}{P}$$

Each patch is flattened:

$$p_i \in \mathbb{R}^{P^{2}C}$$
$$i = 1,...,N$$

Mapped intop embedding space:
$$e_i = W_p P_i +b+ u_i$$

where:

* $W_p=$ Path embedding matrix
* $b = bias$
* $u_i=$Positional encoding

During masked autoencoding, large subset of patches is hidden:

* $V =$ Visible path indices
* $M=$ masked path indices
  
with $V \cup M = \{1,...N\}$

Ecoder see visible patches: $z_V = f_{\theta}(e_V)$

where $f_{\theta} =$ transformer encoder with parameters $\theta$

Decoder then tries to reconstruct missing patches:

$$\hat{p}_M=d_{\psi}(z_V,mask\_token)$$

$\psi=$ decoder parameters

Reconstruction loss standard:

$$\mathcal{L}_{MAE}=\frac{1}{|M|}\sum_{i \in M} ||\hat{p}_i-p_i||_2^2$$

Learned features:
- Spatial smoothness
- Local texture
- Gradients
- Mesoscale moisture structure
- Channel relationships
- Arrangements in SMAP imagery
- Kinds of missingness/noise/statistical regularities that occur in the domain

Therefore: $f_{}\theta: x \mapsto h$
Map raw remote-sensing inputs into internal features $h$

2. Supervised fine-tuning $\rightarrow$ Attach task-specific head and train model on labeled examples
- Labeled data:

$$\mathcal{D}=\{(x_{r,t},y_{r,t})\}^N_{n=1}$$

where:

* $x_n=$ SMAP-based input sensor
* $y_n=$ Target label
* Encoder producer latent tokens: $Z=f_{\theta}(x)$

Thus, 
$$x_{r,t} \in R^{C*H_r*W_r}$$
$$y_{r,t} = SPI3_{r,t}$$