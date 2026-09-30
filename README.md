# Image annotation with Segment Anything

A Python tool for generating candidate bounding boxes with Meta's SAM, selecting them interactively and assigning reusable labels. The current script exports **Pascal VOC XML**.

[Ángel's profile](https://github.com/Ahgarzon) · [Robotics and vision background](https://github.com/Ahgarzon/Ahgarzon/blob/main/projects/robotics-vision.md)

## What the code does

1. Reads `.jpg` images from `images/`.
2. Loads the SAM ViT-B checkpoint and generates masks.
3. Keeps up to six of the largest masks and displays their bounding boxes.
4. Lets the user select boxes and assign or reuse labels through terminal prompts.
5. Saves annotations to `voc_output/<image-name>.xml`.
6. Opens LabelImg when no masks are found or all candidates are skipped.

This is an academic annotation tool, with a human reviewing the proposed regions. It does not automatically identify semantic classes.

## Setup

Clone the actual repository:

```bash
git clone https://github.com/Ahgarzon/Sistema-de-Etiquetado-de-Im-genes-con-SAM.git
cd Sistema-de-Etiquetado-de-Im-genes-con-SAM
python -m venv .venv
```

Activate the environment for your shell. Install PyTorch and TorchVision using the [official PyTorch instructions](https://pytorch.org/get-started/locally/) for your operating system, then install the script's supporting packages:

```bash
python -m pip install numpy pillow matplotlib opencv-python labelImg
python -m pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download the **ViT-B** checkpoint from the [official SAM checkpoint list](https://github.com/facebookresearch/segment-anything#model-checkpoints), and place `sam_vit_b_01ec64.pth` beside the script. Create an `images` directory and add your own `.jpg` files. Use a desktop session that can display Matplotlib and LabelImg windows.

```text
.
├── clasificador_SAM_v4.py
├── sam_vit_b_01ec64.pth    # download separately
├── images/               # your input .jpg files
└── voc_output/           # generated XML annotations
```

## Run

```bash
python clasificador_SAM_v4.py
```

Close the plot to continue to the terminal questions. Enter only displayed box numbers, for example `1,3`, then choose an existing label or type a new one. Check the resulting XML and boxes before using them to train a model.

## Current limits

- The implemented export is Pascal VOC XML. YOLO export is not implemented in this script.
- Model weights, sample images and a locked dependency environment are not bundled.
- LabelImg must be available on the command path for the fallback to open.
- The script contains an older, unused manual-selection helper using Matplotlib's `drawtype` argument; that helper may need adaptation if called with newer Matplotlib versions.
- This documentation was checked against the source. A fresh end-to-end model run and a productivity benchmark have not been performed for this documentation update.

## Implementation

[`clasificador_SAM_v4.py`](clasificador_SAM_v4.py) contains mask generation, candidate selection, label reuse and XML export. SAM is provided by [Meta's Segment Anything project](https://github.com/facebookresearch/segment-anything).
