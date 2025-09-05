# 🖼️ Image Annotation System with SAM + YOLO  

This project implements an **automatic image segmentation and annotation pipeline** using **Meta AI's Segment Anything Model (SAM)** combined with manual fine-tuning.  
The system allows exporting annotations in **Pascal VOC** and **YOLO** formats, ensuring compatibility with widely used computer vision models.  

It also integrates **LabelImg** for validation and manual adjustments, making the annotation process faster and more efficient.  

## 🚀 Features  
- Automatic segmentation of objects using **SAM**.  
- Manual refinement of bounding boxes and masks.  
- Export annotations in **Pascal VOC** and **YOLO** formats.  
- Integration with **LabelImg** for validation and corrections.  
- Reduced labeling time by ~60% compared to manual annotation.  

## 🛠️ Tech Stack  
- **Language:** Python  
- **Libraries:** PyTorch, OpenCV, NumPy, SAM  
- **Annotation Tool:** LabelImg  

## 📂 Project Structure  
- `clasificador_SAM_v4.py` → Main script for annotation.  
- `README.md` → Documentation.  

## ⚙️ Installation  

1. Clone this repository:  
   ```bash
   git clone https://github.com/Ahgarzon/Sistema-de-Etiquetado-de-Imagenes-con-SAM
   cd Sistema-de-Etiquetado-de-Imagenes-con-SAM
