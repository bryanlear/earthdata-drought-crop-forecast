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

