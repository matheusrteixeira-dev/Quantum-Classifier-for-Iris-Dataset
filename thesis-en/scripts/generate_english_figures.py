"""Generate English-labeled versions of the thesis result figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


IMAGE_DIR = Path("imagens")


def save_confusion_matrix(source_name, output_name, values):
    source_size = Image.open(IMAGE_DIR / source_name).size
    figure = plt.figure(
        figsize=(source_size[0] / 100, source_size[1] / 100), dpi=100
    )
    axis = sns.heatmap(
        np.asarray(values),
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=True,
        xticklabels=["setosa", "versicolor", "virginica"],
        yticklabels=["setosa", "versicolor", "virginica"],
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(IMAGE_DIR / output_name, dpi=100, transparent=True)
    plt.close(figure)


def draw_rotated_label(image, text, box, font):
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, fill="white")
    left, top, right, bottom = font.getbbox(text)
    label = Image.new(
        "RGBA", (right - left + 8, bottom - top + 8), (255, 255, 255, 0)
    )
    ImageDraw.Draw(label).text((4 - left, 4 - top), text, font=font, fill="black")
    label = label.rotate(90, expand=True)
    x = box[0] + (box[2] - box[0] - label.width) // 2
    y = box[1] + (box[3] - box[1] - label.height) // 2
    image.alpha_composite(label, (x, y))


def translate_roc_plot(
    source_name,
    output_name,
    legend_box,
    legend_x,
    lines,
    ylabel_box,
    ylabel_font_size,
):
    image = Image.open(IMAGE_DIR / source_name).convert("RGBA")
    font_path = font_manager.findfont("DejaVu Sans")
    legend_font = ImageFont.truetype(font_path, 15)
    axis_font = ImageFont.truetype(font_path, 16)
    ylabel_font = ImageFont.truetype(font_path, ylabel_font_size)
    draw = ImageDraw.Draw(image)

    # Replace only the Portuguese raster text; all curve pixels remain unchanged.
    draw.rectangle(legend_box, fill="white")
    for index, line in enumerate(lines):
        draw.text(
            (legend_x, 385 + index * 24), line, font=legend_font, fill="black"
        )

    xlabel_box = (345, 603, 785, 631)
    draw.rectangle(xlabel_box, fill="white")
    xlabel = "False Positive Rate"
    xlabel_width = axis_font.getlength(xlabel)
    draw.text(
        ((image.width - xlabel_width) / 2, 604),
        xlabel,
        font=axis_font,
        fill="black",
    )
    draw_rotated_label(image, "True Positive Rate", ylabel_box, ylabel_font)
    image.save(IMAGE_DIR / output_name)


save_confusion_matrix(
    "confusion_classic_train.PNG",
    "confusion_classic_train_en.png",
    [[34, 0, 0], [0, 29, 13], [0, 0, 36]],
)
save_confusion_matrix(
    "confusion_classic_test.png",
    "confusion_classic_test_en.png",
    [[16, 0, 0], [0, 7, 1], [0, 1, 13]],
)
save_confusion_matrix(
    "confusion_quantum_train.png",
    "confusion_quantum_train_en.png",
    [[34, 0, 0], [2, 36, 4], [0, 15, 21]],
)
save_confusion_matrix(
    "confusion_quantum_test.png",
    "confusion_quantum_test_en.png",
    [[16, 0, 0], [1, 7, 0], [0, 6, 8]],
)

translate_roc_plot(
    "roc_train.png",
    "roc_train_en.png",
    (630, 382, 1076, 571),
    639,
    [
        "Classical training ROC (micro-average area = 0.91)",
        "Quantum training ROC (micro-average area = 0.86)",
        "Setosa class ROC (classical, area = 1.00)",
        "Versicolor class ROC (classical, area = 0.85)",
        "Virginica class ROC (classical, area = 0.91)",
        "Setosa class ROC (quantum, area = 0.99)",
        "Versicolor class ROC (quantum, area = 0.82)",
        "Virginica class ROC (quantum, area = 0.77)",
    ],
    (0, 145, 34, 445),
    16,
)
translate_roc_plot(
    "roc_test.png",
    "roc_test_en.png",
    (615, 382, 1058, 571),
    623,
    [
        "Classical test ROC (micro-average area = 0.96)",
        "Quantum test ROC (micro-average area = 0.86)",
        "Setosa class ROC (classical, area = 1.00)",
        "Versicolor class ROC (classical, area = 0.92)",
        "Virginica class ROC (classical, area = 0.94)",
        "Setosa class ROC (quantum, area = 0.98)",
        "Versicolor class ROC (quantum, area = 0.84)",
        "Virginica class ROC (quantum, area = 0.79)",
    ],
    (0, 145, 17, 445),
    13,
)