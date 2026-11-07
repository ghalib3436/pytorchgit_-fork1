from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageTk
from torchvision import models


weights = models.MobileNet_V3_Small_Weights.DEFAULT
if getattr(sys, "frozen", False):
    bundled_weights = Path(sys._MEIPASS) / "model" / "mobilenet_v3_small-047dcff4.pth"
    model = models.mobilenet_v3_small(weights=None)
    model.load_state_dict(
        torch.load(bundled_weights, map_location="cpu", weights_only=True)
    )
else:
    model = models.mobilenet_v3_small(weights=weights)
model = model.eval()
preprocess = weights.transforms()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

camera = None
frame_lock = threading.Lock()
latest_frame = None
analysis_overlay = None
events = queue.Queue()
stop_camera = threading.Event()
analysis_running = False
camera_available = False


def camera_loop():
    global latest_frame

    if not camera.isOpened():
        events.put(
            (
                "camera_error",
                "Camera unavailable. Check that it is connected and not in use.",
            )
        )
        camera.release()
        return

    events.put(("camera_ready", None))
    try:
        while not stop_camera.is_set():
            ok, captured_frame = camera.read()
            if not ok:
                events.put(
                    ("camera_error", "Camera unavailable. Could not read a frame.")
                )
                break
            with frame_lock:
                latest_frame = captured_frame
    finally:
        camera.release()


def class_activation_map(input_tensor, class_index, image_size):
    with torch.enable_grad():
        input_tensor = input_tensor.detach().requires_grad_(True)
        features = model.features(input_tensor)
        pooled = model.avgpool(features)
        logits = model.classifier(torch.flatten(pooled, 1))
        gradients, = torch.autograd.grad(logits[0, class_index], features)

    channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
    activation = torch.relu((channel_weights * features).sum(dim=1, keepdim=True))
    activation = functional.interpolate(
        activation,
        size=(image_size[1], image_size[0]),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    activation = activation.detach().cpu().numpy()
    minimum = float(activation.min())
    maximum = float(activation.max())
    if maximum > minimum:
        activation = (activation - minimum) / (maximum - minimum)
    else:
        activation = np.zeros_like(activation)
    return activation


def make_overlay(frame, activation):
    heatmap = cv2.applyColorMap(
        np.uint8(np.clip(activation, 0, 1) * 255),
        cv2.COLORMAP_JET,
    )
    blended = cv2.addWeighted(frame, 0.55, heatmap, 0.45, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def analyze_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = model(input_tensor).softmax(dim=1)[0]
        scores, indices = probabilities.topk(3)

    candidates = []
    for score, index in zip(scores.tolist(), indices.tolist()):
        activation = class_activation_map(input_tensor, index, image.size)
        candidates.append(
            (weights.meta["categories"][index], score, index, activation)
        )

    winner_name, _, _, winner_activation = candidates[0]
    report = [
        "1. Detection: MobileNetV3 proposes three possible objects.",
        "",
        "2. Visual features: Grad-CAM highlights the image regions that "
        "contribute to each classification. The preview shows the selected "
        "hypothesis.",
        "",
        "3. Comparison: the model ranks the three options by class score. "
        "Percentages are relative probabilities, not guarantees.",
        "",
        "4. Ranked results:",
    ]
    report.extend(
        f"   {rank}. {name} — {score:.1%}"
        for rank, (name, score, _, _) in enumerate(candidates, start=1)
    )
    report.extend(
        [
            "",
            "5. Analysis path: captured image → preprocessing → top-three "
            "classification → score comparison → visual explanation for "
            f"{winner_name}.",
            "",
            f"6. Best match: {winner_name} ({candidates[0][1]:.1%}).",
            "",
            "Colored regions show visual influence, not the exact outline "
            "of the object.",
        ]
    )
    return make_overlay(frame, winner_activation), "\n".join(report)


def begin_analysis():
    global analysis_running, analysis_overlay

    if analysis_running:
        return
    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    if frame is None:
        if camera_available:
            status_var.set("Waiting for camera image...")
            show_report("Waiting for an image from the camera.")
        else:
            status_var.set("Camera unavailable")
            show_report("Camera unavailable. Check that it is connected and not in use.")
        return

    analysis_overlay = None
    analysis_running = True
    status_var.set("Analyzing...")
    analyze_button.configure(state="disabled", text="Analyzing...")
    live_button.configure(state="disabled")
    threading.Thread(target=analysis_worker, args=(frame,), daemon=True).start()


def return_to_live_preview():
    global analysis_overlay

    if analysis_running:
        return
    analysis_overlay = None
    show_report(
        "Live camera preview. Select Start Analysis to run another recognition."
    )
    status_var.set("Ready" if camera_available else "Camera unavailable")
    analyze_button.configure(
        state="normal" if camera_available else "disabled",
        text="Start Analysis",
    )


def analysis_worker(frame):
    try:
        events.put(("analysis", analyze_frame(frame)))
    except Exception as error:
        events.put(("analysis_error", f"Analysis failed: {error}"))


def show_report(text):
    report_text.configure(state="normal")
    report_text.delete("1.0", tk.END)
    report_text.insert(tk.END, text)
    report_text.configure(state="disabled")


def process_events():
    global analysis_running, analysis_overlay, camera_available

    while True:
        try:
            event_type, payload = events.get_nowait()
        except queue.Empty:
            break
        if event_type == "analysis":
            analysis_overlay, report = payload
            show_report(report)
            analysis_running = False
            status_var.set("Analysis completed")
            analyze_button.configure(
                state="normal" if camera_available else "disabled",
                text="Start Analysis",
            )
            live_button.configure(
                state="normal" if camera_available else "disabled"
            )
        elif event_type == "camera_ready":
            camera_available = True
            status_var.set("Ready")
            live_button.configure(state="normal")
            if not analysis_running:
                analyze_button.configure(state="normal", text="Start Analysis")
        elif event_type == "camera_error":
            camera_available = False
            status_var.set("Camera unavailable")
            show_report(payload)
            analyze_button.configure(state="disabled", text="Start Analysis")
            live_button.configure(state="disabled")
        elif event_type == "analysis_error":
            show_report(payload)
            analysis_running = False
            status_var.set("Analysis failed")
            analyze_button.configure(
                state="normal" if camera_available else "disabled",
                text="Start Analysis",
            )
            live_button.configure(
                state="normal" if camera_available else "disabled"
            )

    root.after(100, process_events)


def draw_frame():
    if analysis_overlay is not None:
        image_source = analysis_overlay
    else:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        image_source = None if frame is None else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if image_source is not None:
        canvas_width = max(canvas.winfo_width(), 1)
        canvas_height = max(canvas.winfo_height(), 1)
        image = Image.fromarray(image_source)
        image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=photo,
            anchor=tk.CENTER,
        )
        canvas.image = photo
    else:
        canvas_width = max(canvas.winfo_width(), 1)
        canvas_height = max(canvas.winfo_height(), 1)
        canvas.delete("all")
        canvas.create_text(
            canvas_width // 2,
            canvas_height // 2,
            text=status_var.get(),
            fill="#cbd5e1",
            font=("Segoe UI", 11),
            width=max(canvas_width - 40, 1),
        )

    root.after(30, draw_frame)


def close_app():
    stop_camera.set()
    root.destroy()


def main():
    global analyze_button, camera, canvas, live_button, report_text, root, status_var

    root = tk.Tk()
    root.title("AI Vision – Object Recognition")
    root.geometry("1180x760")
    root.minsize(760, 520)

    background = "#f3f6fb"
    surface = "#ffffff"
    foreground = "#172033"
    muted = "#65738a"
    accent = "#2563eb"
    root.configure(background=background)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("App.TFrame", background=background)
    style.configure("Card.TFrame", background=surface)
    style.configure(
        "Title.TLabel",
        background=background,
        foreground=foreground,
        font=("Segoe UI", 20, "bold"),
    )
    style.configure(
        "Subtitle.TLabel",
        background=background,
        foreground=muted,
        font=("Segoe UI", 10),
    )
    style.configure(
        "Section.TLabel",
        background=surface,
        foreground=foreground,
        font=("Segoe UI", 13, "bold"),
    )
    style.configure(
        "Hint.TLabel",
        background=surface,
        foreground=muted,
        font=("Segoe UI", 9),
    )
    style.configure(
        "Status.TLabel",
        background=background,
        foreground=accent,
        font=("Segoe UI", 10, "bold"),
    )
    style.configure(
        "Accent.TButton",
        font=("Segoe UI", 10, "bold"),
        padding=(14, 10),
    )
    style.map(
        "Accent.TButton",
        foreground=[("disabled", "#ffffff"), ("!disabled", "#ffffff")],
        background=[("disabled", "#93a9d0"), ("!disabled", accent)],
    )
    style.configure("Exit.TButton", font=("Segoe UI", 10), padding=(14, 10))

    header = ttk.Frame(root, style="App.TFrame", padding=(22, 18, 22, 12))
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)
    ttk.Label(
        header,
        text="AI Vision – Object Recognition",
        style="Title.TLabel",
    ).grid(row=0, column=0, sticky="w")
    ttk.Label(
        header,
        text="Live camera preview and explainable image classification",
        style="Subtitle.TLabel",
    ).grid(row=1, column=0, sticky="w", pady=(3, 0))
    status_var = tk.StringVar(value="Connecting to camera...")
    ttk.Label(header, textvariable=status_var, style="Status.TLabel").grid(
        row=0, column=1, rowspan=2, sticky="e", padx=(12, 0)
    )

    content = ttk.Frame(root, style="App.TFrame", padding=(22, 8, 22, 12))
    content.grid(row=1, column=0, sticky="nsew")
    content.columnconfigure(0, weight=3, uniform="panels")
    content.columnconfigure(1, weight=2, uniform="panels")
    content.rowconfigure(0, weight=1)

    preview_card = ttk.Frame(content, style="Card.TFrame", padding=14)
    preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    preview_card.columnconfigure(0, weight=1)
    preview_card.rowconfigure(2, weight=1)
    ttk.Label(
        preview_card,
        text="Camera Preview",
        style="Section.TLabel",
    ).grid(row=0, column=0, sticky="w")
    ttk.Label(
        preview_card,
        text="Your live feed and analysis heatmap appear here",
        style="Hint.TLabel",
    ).grid(row=1, column=0, sticky="w", pady=(3, 10))
    canvas = tk.Canvas(
        preview_card,
        background="#111827",
        highlightthickness=0,
        borderwidth=0,
    )
    canvas.grid(row=2, column=0, sticky="nsew")

    results_card = ttk.Frame(content, style="Card.TFrame", padding=14)
    results_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
    results_card.columnconfigure(0, weight=1)
    results_card.rowconfigure(2, weight=1)
    ttk.Label(
        results_card,
        text="Recognition Results",
        style="Section.TLabel",
    ).grid(row=0, column=0, sticky="w")
    ttk.Label(
        results_card,
        text="Top predictions and visual explanation",
        style="Hint.TLabel",
    ).grid(row=1, column=0, sticky="w", pady=(3, 10))

    report_text = tk.Text(
        results_card,
        wrap="word",
        font=("Segoe UI", 10),
        relief="flat",
        padx=12,
        pady=12,
        background="#f7f9fc",
        foreground=foreground,
        selectbackground="#bfdbfe",
        borderwidth=0,
        spacing1=2,
        spacing3=3,
    )
    report_text.grid(row=2, column=0, sticky="nsew")
    report_scrollbar = ttk.Scrollbar(
        results_card,
        orient="vertical",
        command=report_text.yview,
    )
    report_scrollbar.grid(row=2, column=1, sticky="ns")
    report_text.configure(yscrollcommand=report_scrollbar.set)
    report_text.insert(
        tk.END,
        "The camera feed will appear in the preview. Select Start Analysis "
        "to rank the top three predictions and visualize the regions that "
        "influenced the best match.",
    )
    report_text.configure(state="disabled")

    actions = ttk.Frame(root, style="App.TFrame", padding=(22, 0, 22, 20))
    actions.grid(row=2, column=0, sticky="ew")
    actions.columnconfigure(0, weight=1)
    analyze_button = ttk.Button(
        actions,
        text="Start Analysis",
        command=begin_analysis,
        style="Accent.TButton",
        state="disabled",
    )
    live_button = ttk.Button(
        actions,
        text="Back to Live Preview",
        command=return_to_live_preview,
        style="Exit.TButton",
        state="disabled",
    )
    live_button.grid(row=0, column=1, sticky="e", padx=(0, 10))
    analyze_button.grid(row=0, column=2, sticky="e", padx=(0, 10))
    ttk.Button(
        actions,
        text="Exit",
        command=close_app,
        style="Exit.TButton",
    ).grid(row=0, column=3, sticky="e")

    camera = cv2.VideoCapture(0)
    threading.Thread(target=camera_loop, daemon=True).start()
    root.protocol("WM_DELETE_WINDOW", close_app)
    draw_frame()
    process_events()
    root.mainloop()


if __name__ == "__main__":
    main()
