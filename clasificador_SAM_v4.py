import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from matplotlib.widgets import RectangleSelector
import subprocess

############################################
# Función para guardar en formato Pascal VOC
############################################
def save_pascal_voc_xml(folder, filename, path, width, height, depth, boxes, output_path):
    lines = ["<annotation>"]
    lines.append(f"  <folder>{folder}</folder>")
    lines.append(f"  <filename>{filename}</filename>")
    lines.append(f"  <path>{path}</path>")
    lines.append("  <source><database>Unknown</database></source>")
    lines.append(f"  <size><width>{width}</width><height>{height}</height><depth>{depth}</depth></size>")
    lines.append("  <segmented>0</segmented>")
    for (label, xmin, ymin, xmax, ymax) in boxes:
        lines.append("  <object>")
        lines.append(f"    <name>{label}</name>")
        lines.append("    <pose>Unspecified</pose>")
        lines.append("    <truncated>0</truncated>")
        lines.append("    <difficult>0</difficult>")
        lines.append(f"    <bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox>")
        lines.append("  </object>")
    lines.append("</annotation>")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[*] XML guardado en: {output_path}")

############################################
# Selección manual de cajas con el mouse
############################################
def manual_bbox_selection(image_np):
    global bbox_coords
    bbox_coords = []

    def onselect(eclick, erelease):
        global bbox_coords
        x_min, y_min = int(eclick.xdata), int(eclick.ydata)
        x_max, y_max = int(erelease.xdata), int(erelease.ydata)
        bbox_coords = [x_min, y_min, x_max, y_max]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image_np)
    ax.set_title("Selecciona manualmente una región")
    rect_selector = RectangleSelector(
        ax, onselect, drawtype='box', useblit=True,
        button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True
    )
    plt.show()

    if bbox_coords:
        print(f"[+] Caja seleccionada manualmente: {bbox_coords}")
        return bbox_coords
    else:
        print("[!] No se seleccionó ninguna caja.")
        return None

############################################
# Función principal con reutilización interactiva de etiquetas
############################################
def label_bboxes_with_sam_and_reusable_labels():
    images_dir = "./images"
    output_dir = "./voc_output"
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
    if not image_files:
        print(f"[!] No se encontraron imágenes en {images_dir}.")
        return

    ckpt_path = "./sam_vit_b_01ec64.pth"
    if not os.path.exists(ckpt_path):
        print(f"[!] Pesos del modelo no encontrados en {ckpt_path}. Descárgalos y colócalos en esa ubicación.")
        return

    print("[*] Cargando modelo SAM vit_b...")
    sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.88,
        min_mask_region_area=800
    )

    reusable_labels = []  # Lista para almacenar etiquetas reutilizables

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        print(f"\n=== Procesando: {img_path} ===")

        try:
            image_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[!] No se pudo abrir {img_path}. Error: {e}")
            continue

        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]

        print("[*] Generando máscaras...")
        with torch.no_grad():
            masks_info = mask_generator.generate(image_np)

        if not masks_info:
            print("[!] No se detectaron máscaras con la configuración actual. Abriendo LabelImg para selección manual.")
            subprocess.run(["labelImg", img_path])
            continue

        masks_info.sort(key=lambda m: m["area"], reverse=True)
        masks_info = masks_info[:6]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image_np)
        boxes = []
        for i, mask_info in enumerate(masks_info):
            seg = mask_info["segmentation"]
            ys, xs = np.where(seg)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            boxes.append((x_min, y_min, x_max, y_max))
            ax.text(x_min, max(0, y_min - 10), f"#{i + 1}", color='yellow', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
        ax.set_title("Cajas generadas por SAM. Selecciona las que deseas conservar.")
        ax.axis("off")
        plt.show()

        chosen_str = input(f"¿Cuáles bounding boxes conservar? (ej: 1,3) o Enter para omitir todas: ").strip()
        if chosen_str == "":
            print("[!] Omitiendo todas. Abriendo LabelImg para selección manual.")
            subprocess.run(["labelImg", img_path])
            continue

        chosen_indices = [int(x) - 1 for x in chosen_str.split(",") if x.isdigit()]
        if not chosen_indices:
            print("[!] No se seleccionaron cajas.")
            continue

        chosen_boxes = [boxes[i] for i in chosen_indices]
        labeled_boxes = []
        for box in chosen_boxes:
            print(f"\nCaja: {box}")
            print("Etiquetas disponibles:")
            for idx, label in enumerate(reusable_labels, 1):
                print(f"{idx}. {label}")
            print(f"{len(reusable_labels) + 1}. [Crear nueva etiqueta]")

            label_choice = input("Selecciona una etiqueta (número) o escribe una nueva: ").strip()
            if label_choice.isdigit():
                choice_idx = int(label_choice) - 1
                if 0 <= choice_idx < len(reusable_labels):
                    label = reusable_labels[choice_idx]
                else:
                    label = input("Nueva etiqueta: ").strip()
                    reusable_labels.append(label)
            else:
                label = label_choice
                reusable_labels.append(label)

            labeled_boxes.append((label, *box))

        save_pascal_voc_xml(
            folder="images",
            filename=img_name,
            path=img_path,
            width=w,
            height=h,
            depth=3,
            boxes=labeled_boxes,
            output_path=os.path.join(output_dir, os.path.splitext(img_name)[0] + ".xml")
        )

    print("\n=== ¡Proceso completado! ===")

# Ejecutar
label_bboxes_with_sam_and_reusable_labels()
