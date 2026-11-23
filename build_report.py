"""Build the Assignment 02 research report from recorded, checked artifacts."""

from __future__ import annotations

import csv
import json
import os
from html import escape
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
PDF = Path(os.environ.get("REPORT_OUTPUT", ROOT / "report.pdf"))
ROWS = list(csv.DictReader((ROOT / "results.csv").open(encoding="utf-8", newline="")))
BY_NAME = {row["architecture_variant"]: row for row in ROWS}
TEST = json.loads((ART / "final_test_result.json").read_text(encoding="utf-8"))
WINNER = json.loads((ART / "selected_winner.json").read_text(encoding="utf-8"))
VGG = json.loads((ART / "vgg_head_analysis.json").read_text(encoding="utf-8"))

assert len(ROWS) == 11
assert WINNER["run_id"] == "A4-T_s42"
assert TEST["checkpoint"] == WINNER["checkpoint"]
assert sum(map(sum, TEST["confusion_matrix"])) == 10000

INK = colors.HexColor("#16233B")
BLUE = colors.HexColor("#16498A")
TEAL = colors.HexColor("#0D7776")
MUTED = colors.HexColor("#52627A")
PALE = colors.HexColor("#EDF3F9")
LINE = colors.HexColor("#CAD5E2")
WHITE = colors.white

ST = {
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.5,
                           leading=15.5, textColor=INK, spaceAfter=0),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=9.1,
                            leading=13.0, textColor=INK),
    "tiny": ParagraphStyle("tiny", fontName="Helvetica", fontSize=7.2,
                           leading=9.3, textColor=INK),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique",
                              fontSize=8.7, leading=12, textColor=MUTED),
    "table": ParagraphStyle("table", fontName="Helvetica", fontSize=7.3,
                            leading=9.4, textColor=INK),
    "tablehead": ParagraphStyle("tablehead", fontName="Helvetica-Bold",
                                fontSize=7.1, leading=8.7, textColor=WHITE),
}

C = canvas.Canvas(str(PDF), pagesize=A4, pageCompression=1)
C.setTitle("Assignment 02 - CNN Architectures, Training and Ablation Studies")
C.setAuthor("Deep Learning Assignment 02")
C.setSubject("CIFAR-10 architecture comparison and controlled ablations")
PAGE = 0
W, H = A4
M = 43


def next_page(title: str, subtitle: str = "", *, wide: bool = False) -> float:
    global PAGE, W, H
    if PAGE:
        C.showPage()
    PAGE += 1
    W, H = landscape(A4) if wide else A4
    C.setPageSize((W, H))
    C.setFillColor(BLUE)
    C.rect(0, H - 8, W, 8, fill=1, stroke=0)
    C.setFont("Helvetica-Bold", 13.2 if not wide else 14)
    C.setFillColor(INK)
    C.drawString(M, H - 43, title)
    if subtitle:
        C.setFont("Helvetica", 8.3)
        C.setFillColor(MUTED)
        C.drawString(M, H - 60, subtitle)
    C.setStrokeColor(LINE)
    C.line(M, 38, W - M, 38)
    C.setFont("Helvetica", 7.5)
    C.setFillColor(MUTED)
    C.drawString(M, 25, "AIR UNIVERSITY  |  DEEP LEARNING  |  ASSIGNMENT 02")
    C.drawRightString(W - M, 25, f"{PAGE} / 8")
    return H - (82 if subtitle else 67)


def para(text: str, y: float, *, style: str = "body", x: float | None = None,
         width: float | None = None, gap: float = 9) -> float:
    x = M if x is None else x
    width = W - 2 * M if width is None else width
    p = Paragraph(text, ST[style])
    _, height = p.wrap(width, 1000)
    p.drawOn(C, x, y - height)
    y -= height + gap
    if y < 44:
        raise RuntimeError(f"Page {PAGE} overflow after: {text[:80]}")
    return y


def heading(text: str, y: float) -> float:
    C.setFillColor(TEAL)
    C.rect(M, y - 3, 3, 13, fill=1, stroke=0)
    C.setFont("Helvetica-Bold", 10.8)
    C.setFillColor(INK)
    C.drawString(M + 11, y, text)
    return y - 19


def bullet(text: str, y: float, *, style: str = "body") -> float:
    return para("<font color='#0D7776'><b>-</b></font>  " + text,
                y, style=style, x=M + 5, width=W - 2 * M - 5, gap=5)


def table(headers: list[str], rows: list[list[str]], widths: list[float],
          y: float, *, head: bool = True, font_size: float = 7.5,
          row_bg: bool = True) -> float:
    data = []
    if head:
        data.append([Paragraph(h, ST["tablehead"]) for h in headers])
    for row in rows:
        data.append([Paragraph(str(v), ST["table"]) for v in row])
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1 if head else 0)
    styles = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE),
    ]
    if head:
        styles += [("BACKGROUND", (0, 0), (-1, 0), BLUE),
                   ("TOPPADDING", (0, 0), (-1, 0), 7),
                   ("BOTTOMPADDING", (0, 0), (-1, 0), 7)]
    if row_bg:
        start = 1 if head else 0
        for i in range(start, len(data)):
            if (i - start) % 2 == 0:
                styles.append(("BACKGROUND", (0, i), (-1, i), PALE))
    t.setStyle(TableStyle(styles))
    _, height = t.wrap(sum(widths), 1000)
    t.drawOn(C, M, y - height)
    if y - height < 44:
        raise RuntimeError(f"Page {PAGE} table overflow: {height}")
    return y - height - 10


def figure(path: Path, y: float, *, max_width: float, max_height: float,
           x: float | None = None) -> float:
    x = M if x is None else x
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(max_width / iw, max_height / ih)
    dw, dh = iw * scale, ih * scale
    C.drawImage(ImageReader(str(path)), x + (max_width - dw) / 2, y - dh,
                width=dw, height=dh, preserveAspectRatio=True)
    return y - dh


def pct(value: str | float, decimals: int = 2) -> str:
    return f"{100 * float(value):.{decimals}f}%"


def pp(a: str | float, b: str | float) -> str:
    return f"{100 * (float(a) - float(b)):+.2f} pp"


# Page 1: Part I Q1/Q2.
y = next_page("Part I - numerical reasoning (Q1-Q2)",
              "All calculations use the stated shapes, padding, strides and bias assumptions.")
y = para(
    "This report checks the four required CNN designs and compares their recorded "
    "CIFAR-10 results under the assignment protocol. The ablations test whether "
    "normalization, augmentation and residual shortcuts change validation "
    "performance; the ImageNet-pretrained branch is evaluated separately. "
    "All findings are limited to the recorded five-epoch, single-seed runs.",
    y, style="small")
y = heading("Q1. CNN shape, parameters and first-convolution MACs", y)
y = para(
    "For same padding, output height/width is ceil(input/stride). A convolution "
    "with bias has (k_h x k_w x C_in + 1) x C_out trainable parameters; a dense "
    "layer has (N_in + 1) x N_out. A MAC is one multiply-accumulate for one "
    "kernel weight at one output position. Pooling and GAP have no trainable "
    "parameters.", y)
y = table(
    ["Operation", "Output shape", "Trainable parameters", "Calculation"],
    [
        ["Input", "96 x 96 x 3", "0", "-"],
        ["Conv 5 x 5, 32, stride 2", "48 x 48 x 32", "2,432", "(5 x 5 x 3 + 1) x 32"],
        ["MaxPool 2 x 2, stride 2", "24 x 24 x 32", "0", "-"],
        ["Conv 3 x 3, 64, stride 1", "24 x 24 x 64", "18,496", "(3 x 3 x 32 + 1) x 64"],
        ["Global average pool", "64", "0", "-"],
        ["Dense, 10 logits", "10", "650", "(64 + 1) x 10"],
    ], [142, 112, 112, W - 2 * M - 366], y)
y = para(
    "<b>Total trainable parameters:</b> 2,432 + 18,496 + 650 = 21,578. "
    "<b>First-convolution MACs:</b> 48 x 48 x 32 x 5 x 5 x 3 = 5,529,600. "
    "Bias additions are excluded from MACs; boundary zeros are still counted "
    "under this dense-kernel convention.", y)
y = heading("Q2. Why stacked 3 x 3 kernels can replace 7 x 7", y - 4)
y = para(
    "With stride one and same padding, one 7 x 7 convolution has a theoretical "
    "7 x 7 receptive field. Three stacked 3 x 3 layers expand the theoretical "
    "field from 3 to 5 to 7 in each spatial dimension. With 64 input and "
    "output channels at every layer and biases ignored:", y)
y = table(
    ["Design", "Weights", "Receptive field", "Difference"],
    [
        ["One 7 x 7", "7 x 7 x 64 x 64 = 200,704", "7 x 7", "Reference"],
        ["Three 3 x 3", "3 x (3 x 3 x 64 x 64) = 110,592", "7 x 7", "90,112 fewer (44.9%)"],
    ], [104, 198, 86, W - 2 * M - 388], y)
y = para(
    "The three-layer option inserts two additional ReLU transitions and a "
    "deeper computation graph, allowing more nonlinear feature composition. "
    "Its theoretical field matches 7 x 7; actual effective influence can "
    "differ after training. This is the parameter-efficient small-kernel idea "
    "associated with VGG [2].", y)


# Page 3: Part I Q3/Q4.
y = next_page("Part I - numerical reasoning (Q3-Q4)",
              "Inception counts ignore biases; the projection shortcut uses no bias.")
y = heading("Q3. Inception branch arithmetic", y)
y = para(
    "The input is 28 x 28 x 192. Every branch keeps the 28 x 28 spatial size "
    "through stride one and same padding. Convolution weights equal "
    "k_h x k_w x C_in x C_out.", y)
y = table(
    ["Branch", "Output", "Convolution weights"],
    [
        ["1 x 1 -> 64", "28 x 28 x 64", "192 x 64 = 12,288"],
        ["1 x 1 -> 32; 3 x 3 -> 128", "28 x 28 x 128",
         "192 x 32 + 9 x 32 x 128 = 43,008"],
        ["1 x 1 -> 16; 5 x 5 -> 32", "28 x 28 x 32",
         "192 x 16 + 25 x 16 x 32 = 15,872"],
        ["3 x 3 pool; 1 x 1 -> 32", "28 x 28 x 32",
         "192 x 32 = 6,144"],
        ["Concatenation", "28 x 28 x 256", "Total = 77,312"],
    ], [185, 105, W - 2 * M - 290], y)
y = para(
    "Without the 3 x 3 and 5 x 5 reductions, the two wide convolutions cost "
    "9 x 192 x 128 = 221,184 and 25 x 192 x 32 = 153,600 weights. "
    "Including the unchanged 1 x 1 and pool-projection branches gives "
    "12,288 + 221,184 + 153,600 + 6,144 = 393,216 weights. "
    "The reductions save 315,904 weights (80.34%) for this module while "
    "keeping its output width 256 [3].", y)
y = heading("Q4. Projection shortcut and BatchNorm", y - 2)
y = para(
    "Here x is 32 x 32 x 64 while F(x) is 16 x 16 x 128. Use a 1 x 1 "
    "projection convolution with 128 filters, stride 2, no bias and same "
    "padding. It produces 16 x 16 x 128, so elementwise addition is valid. "
    "Its weight count is 1 x 1 x 64 x 128 = <b>8,192</b>.", y)
y = para(
    "The residual output is y = F(x) + W_s(x), followed by the block "
    "activation in this implementation. The shortcut contributes a direct "
    "gradient route through W_s; a shape-preserving identity shortcut "
    "contributes an exact additive identity route. A projection is necessary "
    "here because x and F(x) differ in both spatial size and channels [4].", y)
y = para(
    "For NHWC BatchNorm, each channel's training mean and variance are "
    "estimated over the minibatch and its H x W positions; learned gamma and "
    "beta scale and shift normalized activations. During inference, moving "
    "averages replace the current batch statistics. Small batches can make "
    "training estimates noisy.", y)


# Page 4: methods and fidelity.
y = next_page("Methods and implementation fidelity",
              "Structural evidence is available in artifacts/architectures/ and the model source.")
y = heading("Data and optimization protocol", y)
y = para(
    "CIFAR-10 supplies ten classes. One stratified split drawn from its original "
    "training partition fixed 12,000 training and 3,000 validation images "
    "(seed 42); the original 10,000 test images were reserved. Resizing occurs "
    "in tf.data: 96 x 96 for the architecture benchmark and A4, 64 x 64 for "
    "A0-A3. All comparable scratch runs use batch 16, Adam at 3e-4, sparse "
    "categorical cross-entropy from float32 logits, five epochs and the same "
    "best-validation-accuracy checkpoint rule. The learning-rate sanity check "
    "tested 1e-4 and 3e-4 briefly before the main runs. A4-T used 1e-5 after "
    "loading A4-F weights and recompiling. Mixed precision was enabled on the "
    "T4; datasets prefetch and models are cleared between runs.", y)
y = heading("Architecture fidelity and declared adaptations", y)
y = table(
    ["Model", "Preserved design and historical role", "CIFAR-10 adaptation / departure"],
    [
        ["AlexNet [1]",
         "Five convolutions (96/256/384/384/256), 11 x 11 stride-4 first layer, "
         "placed pooling, ReLU, two 4096-unit FC layers and dropout. The 2012 "
         "design demonstrated large-scale GPU CNN classification.",
         "96 x 96 input and 10 logits. LRN and grouped convolutions from the "
         "original two-GPU design are omitted; same padding is used."],
        ["VGG16 [2]",
         "Five blocks with 2/2/3/3/3 small-kernel convolutions, 13 conv layers "
         "and two 4096-unit FC layers. It demonstrated the value of depth with "
         "uniform 3 x 3 filters.",
         "Manual implementation, 96 x 96 input and 10 logits. The classifier "
         "is retained; a GAP alternative is analyzed but not trained."],
        ["GoogLeNet [3]",
         "Stem plus nine four-branch Inception modules with 1 x 1 reductions "
         "and GAP. It combined multiscale features with controlled cost.",
         "96 x 96 input and 10 logits. Auxiliary classifiers are described "
         "but not trained."],
        ["ResNet-18 [4]",
         "Four stages of 2/2/2/2 two-convolution basic blocks with identity or "
         "projection shortcuts. Residual learning eased deep optimization.",
         "96 x 96 input and 10 logits. Projection paths use 1 x 1 convolutions "
         "when shape changes; Add shapes are asserted."],
    ], [78, 219, W - 2 * M - 297], y)
y = para(
    "Each model passed a 2 x 96 x 96 x 3 forward check with 2 x 10 logits. "
    "Manual first-convolution counts matched Keras: AlexNet 34,944; VGG16 "
    "1,792; GoogLeNet 9,472; ResNet-18 9,408. The diagrams, summaries and "
    "JSON checks are retained in the repository. These are CIFAR-10 "
    "adaptations, not pixel-perfect ImageNet reproductions.", y, style="small")
y = para(
    f"<b>VGG head analysis.</b> Manual VGG16 has {VGG['canonical_total_params']:,} "
    f"parameters. A GAP-head variant would have {VGG['gap_total_params']:,}, "
    f"saving {VGG['saved_parameters']:,}; this is analytic only, not an "
    "additional benchmark run.", y, style="small")
y = figure(ART / "figures" / "architecture_comparison.png", y - 2,
           max_width=W - 2 * M, max_height=175) - 4
y = para(
    "Figure 1. Recorded best validation accuracy for the four scratch "
    "architecture adaptations; complete histories and fit times appear on "
    "pages 4-5.",
    y, style="caption")


# Page 5: full comparison table on landscape page.
y = next_page("All recorded runs - required comparison table",
              "Best validation metrics use the selected checkpoint; gap is final train accuracy minus final validation accuracy.",
              wide=True)
headers = ["Run ID", "Architecture / variant", "Input", "Seed", "Trainable / total params",
           "Epochs", "Best val acc.", "Val macro-F1", "Final train acc.",
           "Gen. gap", "Train min", "Peak GPU MB", "Key change / caveat"]
caveat = {
    "AlexNet": "Scratch; 96 px; single seed",
    "VGG16": "Scratch; 96 px; chance-level result",
    "GoogLeNet": "Scratch; 96 px; chance-level result",
    "ResNet18": "Scratch; 96 px; large final gap",
    "A0": "64 px baseline",
    "A1": "A0 + BN",
    "A2": "A1 + augmentation",
    "A3-P": "A2 + deeper plain blocks",
    "A3-R": "A3-P + residual paths; small param difference",
    "A4-F": "Separate pretrained frozen trunk",
    "A4-T": "Separate fine-tuned trunk; starts from A4-F",
}
body = []
for row in ROWS:
    body.append([
        escape(row["run_id"]), escape(row["architecture_variant"]),
        escape(row["input_size"]), escape(row["seed"]),
        f"{int(row['trainable_params']):,} / {int(row['total_params']):,}",
        escape(row["epochs"]), pct(row["best_val_accuracy"]),
        pct(row["macro_f1"]), pct(row["final_train_accuracy"]),
        f"{100 * float(row['generalization_gap']):+.2f} pp",
        f"{float(row['training_minutes']):.2f}",
        f"{float(row['peak_gpu_memory_mb']):.0f}",
        escape(caveat[row["architecture_variant"]]),
    ])
usable = W - 2 * M
widths = [79, 73, 38, 27, 91, 29, 46, 47, 48, 45, 39, 45,
          usable - sum([79, 73, 38, 27, 91, 29, 46, 47, 48, 45, 39, 45])]
y = table(headers, body, widths, y)
y = para(
    "Training minutes time model.fit only. Peak GPU memory is the per-run "
    "TensorFlow measurement where available, not a whole-session maximum. "
    "A4 uses ImageNet initialization and 96 x 96 inputs, so it is shown "
    "separately from the controlled 64 x 64 ablation chain. All rows are "
    "seed 42; no seed-123 repetitions were run.",
    y, style="small", width=usable)
y = para(
    "Source: recorded artifacts/results.csv (11 rows), per-run histories and "
    "checkpoint files. The reserved test result is intentionally excluded "
    "from model selection and appears on page 7.",
    y, style="caption", width=usable)


# Page 6: architecture results and 6-sentence selection analysis.
y = next_page("Four-architecture benchmark",
              "Same 96 x 96 subset, batch, optimizer, seed and five-epoch budget.")
y = figure(ART / "figures" / "four_architecture_curves.png", y,
           max_width=W - 2 * M, max_height=190) - 5
y = para(
    "Figure 2. Training and validation accuracy/loss for the four scratch "
    "architectures. Values are the recorded five-epoch histories.",
    y, style="caption")
y = para(
    "ResNet-18 had the best scratch validation accuracy (52.10%), closely "
    "followed by AlexNet (50.90%). VGG16 and GoogLeNet remained at 10.00%, "
    "with losses near log(10) = 2.303; this run did not establish their "
    "intrinsic capability, only failure to learn under the short common "
    "configuration. ResNet-18's final training accuracy reached 77.19% while "
    "its final validation accuracy fell to 46.70%, a 30.49-point final gap "
    "that warns against selecting by training accuracy alone.", y)
y = heading("Why fastest and best validation can differ", y)
for sentence in [
    "The fastest scratch fit was AlexNet at 1.44 minutes, yet ResNet-18 achieved the higher best validation accuracy.",
    "A fixed five-epoch budget gives the models equal data exposure but does not equalize parameters, kernels or arithmetic cost.",
    "Optimization and regularization can change how much validation progress a model makes per minute.",
    "The same model can peak on validation before its final epoch, as the ResNet-18 curves illustrate.",
    "Validation performance estimates generalization under this split, whereas fit time measures resource expenditure.",
    "Therefore the chosen model should follow the predeclared validation rule, with speed and memory reported as separate practical constraints.",
]:
    y = para(sentence, y, style="small", gap=4)
y = para(
    "The architecture benchmark does not isolate any single architectural "
    "feature because depth, width, parameter count, normalization and "
    "classifier heads differ simultaneously. These runs are descriptive "
    "comparisons of adapted designs [1-4].",
    y, style="small")


# Page 7: ablations.
y = next_page("Controlled ablations and transfer branch",
              "A0-A3 share the fixed 64 x 64 data split and optimizer controls.")
y = figure(ART / "figures" / "ablation_comparison.png", y,
           max_width=W - 2 * M, max_height=215) - 5
y = para(
    "Figure 3. Best validation accuracy of the ablation chain and separate "
    "pretrained branch. A4 should not be interpreted as a one-factor "
    "continuation of A3.",
    y, style="caption")
y = table(
    ["Comparison", "Best-val change", "Interpretation"],
    [
        ["A1 vs A0", pp(BY_NAME["A1"]["best_val_accuracy"], BY_NAME["A0"]["best_val_accuracy"]),
         "BN improved this run; convergence and gap also changed."],
        ["A2 vs A1", pp(BY_NAME["A2"]["best_val_accuracy"], BY_NAME["A1"]["best_val_accuracy"]),
         "Augmentation gave a small improvement under the fixed protocol."],
        ["A3-R vs A3-P", pp(BY_NAME["A3-R"]["best_val_accuracy"], BY_NAME["A3-P"]["best_val_accuracy"]),
         "Shortcuts did not improve this short run; projections add parameters."],
        ["A4-T vs A4-F", pp(BY_NAME["A4-T"]["best_val_accuracy"], BY_NAME["A4-F"]["best_val_accuracy"]),
         "Limited fine-tuning improved the frozen transfer branch."],
    ], [105, 97, W - 2 * M - 202], y)
y = para(
    "A0 -> A1 isolates BN in the same lightweight layout. A1 -> A2 adds "
    "label-preserving flip, translation and contrast augmentation while "
    "retaining BN. A3-P adds depth and capacity, so its gain over A2 is not "
    "a pure depth effect. A3-R matches A3-P's main blocks and widths but "
    "introduces identity/projection shortcuts; the projections increase "
    "parameters from 66,794 to 69,322, so the comparison is close but not "
    "perfectly parameter-matched. The observed -1.83-point A3-R difference "
    "does not demonstrate that residual learning is harmful in general. "
    "All effects are provisional because only seed 42 was recorded.", y)
y = para(
    "A4-F uses ImageNet-pretrained ResNet50 features and original 0-255-scale "
    "inputs passed through resnet50.preprocess_input. The frozen backbone "
    "is called with training=False to preserve BN inference behavior. "
    "A4-T starts from A4-F, unfreezes selected high layers, recompiles "
    "and uses Adam at 1e-5. Its 2.33-point validation gain is a within-branch "
    "finding; comparison with A3 also changes architecture, initialization, "
    "input resolution and external training data.", y, style="small")


# Page 8: error analysis.
y = next_page("Final selection and error analysis",
              "The winner was recorded before the single test evaluation.")
y = para(
    f"A4-T had the largest validation accuracy ({pct(BY_NAME['A4-T']['best_val_accuracy'])}) "
    f"and validation macro-F1 ({pct(BY_NAME['A4-T']['macro_f1'])}) among the "
    "recorded runs. The selected_winner.json file names its checkpoint before "
    "final_test_result.json was written. On the original 10,000-image CIFAR-10 "
    f"test split, accuracy was {pct(TEST['test_accuracy'])} and macro-F1 was "
    f"{pct(TEST['test_macro_f1'])}; test accuracy was "
    f"{pp(TEST['test_accuracy'], BY_NAME['A4-T']['best_val_accuracy'])} "
    "relative to best validation accuracy.", y)
img_y = y
figure(ART / "figures" / "final_confusion_matrix.png", img_y,
       max_width=310, max_height=285, x=M)
right_x = M + 322
right_w = W - M - right_x
ry = img_y - 4
ry = para("<b>Largest off-diagonal errors</b>", ry, style="small",
          x=right_x, width=right_w, gap=5)
for item in TEST["top_confusions"]:
    ry = para(
        f"{escape(item['true'])} -> {escape(item['predicted'])}: "
        f"{item['count']}",
        ry, style="small", x=right_x, width=right_w, gap=4)
ry = para(
    "Cat/dog and deer/horse confusions suggest visually related classes "
    "remain difficult at CIFAR-10 scale. This is an interpretation of "
    "the observed matrix, not a claim about individual images.",
    ry, style="small", x=right_x, width=right_w)
y = img_y - 305
y = para(
    "Figure 4. Confusion matrix for the selected A4-T checkpoint. Each true "
    "class contributes 1,000 test images; diagonal total is 8,497.",
    y, style="caption")
y = heading("Compute budget and limitations", y - 14)
y = para(
    f"The 11 recorded model.fit times sum to "
    f"{sum(float(r['training_minutes']) for r in ROWS):.2f} minutes; "
    "this excludes setup, download, diagrams, report generation and the "
    "later CPU final test. Training used a Colab Tesla T4; the final test "
    "used a separate Colab runtime with no GPU visible, both documented "
    "in the environment JSON files. Peak GPU memory was recorded per run "
    "in the comparison table. No failed_runs.jsonl is present in the "
    "archived evidence, so no failed/OOM attempt or alternate batch size "
    "is claimed. Single-seed, five-epoch findings have substantial "
    "uncertainty; the requested seed-123 repeats were not run because "
    "they were quota-conditional. The test set was evaluated only for "
    "the preselected A4-T model.", y, style="small")


# Page 9: reproducibility and references.
y = next_page("Reproducibility, conclusions and references",
              "Repository evidence is organized for independent inspection.")
y = heading("Reproduction record", y)
for item in [
    "Source: cnn_assignment.ipynb writes src/models.py, src/data.py and src/experiments.py; requirements.txt lists the Python dependencies.",
    "Split: artifacts/split_indices.npz records the stratified 12,000/3,000 indices; test uses the original 10,000-image partition.",
    "Run settings: seed 42, batch 16, five epochs, Adam 3e-4 for comparable scratch runs, 1e-5 for A4-T; best validation accuracy selects checkpoints.",
    "Evidence: results.csv, per-run histories, model summaries, diagrams, parameter/forward checks, plots and final-test JSON are in the repository; large weights are retained in the local archive.",
    "Environment: TensorFlow 2.21.0, Keras 3.13.2, Tesla T4 for training. The recorded nvidia-smi output and CPU final-test environment are separate JSON files.",
    "Code snapshot: Git commit 4bc414d contains the source and experiment results; this commit was created after the runs and is not claimed as a runtime-generated hash.",
]:
    y = bullet(item, y, style="small")
y = heading("Conclusions", y - 3)
y = para(
    "The structural checks reproduced the intended stage patterns and "
    "10-logit interfaces. Under the short common scratch budget, ResNet-18 "
    "led the four adapted architectures, but its large final train-validation "
    "gap cautions against reading training accuracy as generalization. "
    "BN improved the lightweight baseline; augmentation added a small "
    "increment; the residual variant did not beat the deeper plain control "
    "in this seed. ImageNet-pretrained features produced the strongest "
    "validation result, and limited fine-tuning improved the frozen branch. "
    "Future work should repeat the controlled runs at seed 123, extend "
    "training with a tuned schedule, and report uncertainty across seeds "
    "without touching the reserved test set again.", y)
y = heading("References", y - 2)
refs = [
    "[1] A. Krizhevsky, I. Sutskever and G. E. Hinton, "
    "<i>ImageNet Classification with Deep Convolutional Neural Networks</i>, "
    "NeurIPS 25, 2012. papers.nips.cc/paper_files/paper/2012/hash/"
    "c399862d3b9d6b76c8436e924a68c45b-Abstract.html",
    "[2] K. Simonyan and A. Zisserman, "
    "<i>Very Deep Convolutional Networks for Large-Scale Image Recognition</i>, "
    "arXiv:1409.1556, 2014. arxiv.org/abs/1409.1556",
    "[3] C. Szegedy et al., <i>Going Deeper with Convolutions</i>, "
    "arXiv:1409.4842, 2014. arxiv.org/abs/1409.4842",
    "[4] K. He, X. Zhang, S. Ren and J. Sun, "
    "<i>Deep Residual Learning for Image Recognition</i>, "
    "arXiv:1512.03385, 2015. arxiv.org/abs/1512.03385",
    "[5] Air University Deep Learning course, Lecture 05, "
    "<i>Image Classification with CNNs</i>, course slides.",
    "[6] Air University Deep Learning course, Lecture 06, "
    "<i>CNN Architectures &amp; Training</i>, course slides.",
]
for ref in refs:
    y = para(ref, y, style="small", gap=5)

assert PAGE == 8
C.save()
print(PDF)
