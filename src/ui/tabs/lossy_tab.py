import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinterdnd2 import DND_FILES
from PIL import Image, ImageTk
import numpy as np
import io
import os

from ...config import ModernStyle
from ...algorithms import lossy


class LossyTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.lossy_original_filepath = None
        self.lossy_image = None
        self.lossy_quantized_image = None
        self.setup_ui()

    def setup_ui(self):
        container = tk.Frame(self, bg=ModernStyle.BG_PRIMARY)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # Controls
        top_card = tk.Frame(container, bg=ModernStyle.BG_SECONDARY, relief="flat", bd=1)
        top_card.pack(fill="x", pady=(0, 15))
        self._build_controls(top_card)

        # Stats
        stats_card = tk.Frame(
            container, bg=ModernStyle.BG_SECONDARY, relief="flat", bd=1
        )
        stats_card.pack(fill="x", pady=(0, 15))
        self._build_stats(stats_card)

        # Images
        images_card = tk.Frame(
            container, bg=ModernStyle.BG_SECONDARY, relief="flat", bd=1
        )
        images_card.pack(fill="both", expand=True)
        self._build_image_display(images_card)

    def _build_controls(self, parent):
        controls = tk.Frame(parent, bg=ModernStyle.BG_SECONDARY)
        controls.pack(fill="x", padx=20, pady=15)

        # Upload
        upload_section = tk.Frame(controls, bg=ModernStyle.BG_SECONDARY)
        upload_section.pack(side="left", fill="x", expand=True)
        self.dnd_img_lbl = tk.Label(
            upload_section,
            text="📁 Drag & Drop Image Here",
            bg="#f8f9fa",
            fg=ModernStyle.TEXT_SECONDARY,
            relief="solid",
            bd=1,
            cursor="hand2",
        )
        self.dnd_img_lbl.pack(fill="x", ipady=15)
        self.dnd_img_lbl.drop_target_register(DND_FILES)
        self.dnd_img_lbl.dnd_bind("<<Drop>>", self.on_drop_image)
        self.dnd_img_lbl.bind("<Button-1>", self.on_browse_image)

        # Settings
        settings = tk.Frame(controls, bg=ModernStyle.BG_SECONDARY)
        settings.pack(side="left", padx=20)
        tk.Label(settings, text="Levels:", bg=ModernStyle.BG_SECONDARY).pack(
            side="left", padx=5
        )
        self.quant_levels = tk.IntVar(value=4)
        tk.Spinbox(
            settings, from_=2, to=256, textvariable=self.quant_levels, width=6
        ).pack(side="left")

        # Buttons
        btns = tk.Frame(controls, bg=ModernStyle.BG_SECONDARY)
        btns.pack(side="left", padx=10)
        ttk.Button(btns, text="▶ Apply", command=self.perform_quantization).pack(
            side="left", padx=5
        )
        self.btn_save_img = ttk.Button(
            btns, text="💾 Save", command=self.save_image, state="disabled"
        )
        self.btn_save_img.pack(side="left", padx=5)

    def _build_stats(self, parent):
        inner = tk.Frame(parent, bg=ModernStyle.BG_SECONDARY)
        inner.pack(fill="x", padx=20, pady=10)
        self.lbl_lossy_orig_size = tk.Label(
            inner, text="Original Size: - KB", bg=ModernStyle.BG_SECONDARY
        )
        self.lbl_lossy_orig_size.pack(side="left", padx=10)
        self.lbl_lossy_comp_size = tk.Label(
            inner, text="Compressed Size: - KB", bg=ModernStyle.BG_SECONDARY
        )
        self.lbl_lossy_comp_size.pack(side="left", padx=10)
        self.lbl_lossy_stats = tk.Label(
            inner,
            text="Ratio: - | Efficiency: -",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.SUCCESS_COLOR,
            font=(ModernStyle.FONT_FAMILY, 10, "bold"),
        )
        self.lbl_lossy_stats.pack(side="left", padx=10)
        # Mean Squared Error label (added)
        self.lbl_lossy_mse = tk.Label(
            inner,
            text="MSE: -",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_SECONDARY,
            font=(ModernStyle.FONT_FAMILY, 10),
        )
        self.lbl_lossy_mse.pack(side="left", padx=10)

    def _build_image_display(self, parent):
        frame = tk.Frame(parent, bg=ModernStyle.BG_SECONDARY)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Grid layout
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        for idx, title in enumerate(["Original Image", "Quantized Image"]):
            sub_frame = tk.Frame(frame, bg=ModernStyle.BG_SECONDARY)
            sub_frame.grid(row=0, column=idx, padx=10, sticky="nsew")
            tk.Label(
                sub_frame,
                text=title,
                bg=ModernStyle.BG_SECONDARY,
                font=ModernStyle.FONT_HEADING,
            ).pack(pady=(0, 10))
            canvas = tk.Canvas(
                sub_frame,
                bg="#f8f9fa",
                width=500,
                height=500,
                highlightthickness=1,
                highlightbackground=ModernStyle.BORDER_COLOR,
            )
            canvas.pack()
            if idx == 0:
                self.canvas_orig = canvas
            else:
                self.canvas_quant = canvas

    # --- Logic ---
    def load_image(self, filepath):
        try:
            self.lossy_original_filepath = filepath
            img = Image.open(filepath)
            self.lossy_image = img
            file_size_bytes = os.path.getsize(filepath)

            self.lbl_lossy_orig_size.config(
                text=f"Original Size: {file_size_bytes/1024:.2f} KB"
            )
            self.lbl_lossy_comp_size.config(text="Compressed Size: -")
            self.lbl_lossy_stats.config(text="Ratio: - | Efficiency: -")
            self.btn_save_img.config(state="disabled")

            self._display_on_canvas(img, self.canvas_orig)
            self.canvas_quant.delete("all")
            self.dnd_img_lbl.config(
                text=f"✓ Loaded: {os.path.basename(filepath)}",
                bg="#d4edda",
                fg=ModernStyle.SUCCESS_COLOR,
            )
        except Exception as e:
            messagebox.showerror("Error", f"Invalid Image: {e}")

    def _display_on_canvas(self, pil_image, canvas):
        display_img = pil_image.copy()
        display_img.thumbnail((500, 500))
        tk_img = ImageTk.PhotoImage(display_img)
        canvas.create_image(250, 250, image=tk_img, anchor="center")
        canvas.image = tk_img  # Keep reference

    def on_drop_image(self, event):
        filepath = event.data.strip("{}")
        self.load_image(filepath)

    def on_browse_image(self, event):
        filepath = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if filepath:
            self.load_image(filepath)

    def perform_quantization(self):
        if self.lossy_image is None:
            messagebox.showwarning("Warning", "Upload an image first.")
            return
        levels = self.quant_levels.get()
        img_array = np.array(self.lossy_image)
        quantized_arr, _ = lossy.uniform_quantization(img_array, levels)

        self.lossy_quantized_image = Image.fromarray(quantized_arr)
        self._display_on_canvas(self.lossy_quantized_image, self.canvas_quant)

        self.btn_save_img.config(state="normal")
        self.calculate_stats()

    def calculate_stats(self):
        if not self.lossy_original_filepath or not self.lossy_quantized_image:
            return
        orig_bytes = os.path.getsize(self.lossy_original_filepath)

        buffer = io.BytesIO()
        fmt = self.lossy_image.format if self.lossy_image.format else "PNG"
        self.lossy_quantized_image.save(buffer, format=fmt)
        comp_bytes = buffer.tell()
        if comp_bytes == 0:
            comp_bytes = 1

        ratio = orig_bytes / comp_bytes
        efficiency = (1 - (comp_bytes / orig_bytes)) * 100
        diff = orig_bytes - comp_bytes

        self.lbl_lossy_comp_size.config(text=f"Compressed: {comp_bytes/1024:.2f} KB")
        self.lbl_lossy_stats.config(
            text=f"Ratio: {ratio:.2f} | Efficiency: {efficiency:.2f}% (Saved {diff/1024:.2f} KB)"
        )
        # Calculate MSE between original and quantized images
        try:
            orig_arr = np.array(self.lossy_image).astype("float32")
            quant_arr = np.array(self.lossy_quantized_image).astype("float32")
            # Resize quantized array to match original if shapes differ
            if orig_arr.shape != quant_arr.shape:
                # Attempt to broadcast or reshape channels: convert to same shape via simple resize
                from PIL import Image

                q_img_resized = Image.fromarray(self.lossy_quantized_image).resize(
                    (orig_arr.shape[1], orig_arr.shape[0])
                )
                quant_arr = np.array(q_img_resized).astype("float32")

            mse = np.mean((orig_arr - quant_arr) ** 2)
        except Exception:
            mse = float("nan")

        self.lbl_lossy_mse.config(text=f"MSE: {mse:.2f}")

    def save_image(self):
        if self.lossy_quantized_image is None:
            return
        file_types = [("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")]
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=file_types
        )
        if save_path:
            try:
                self.lossy_quantized_image.save(save_path)
                messagebox.showinfo("Success", f"Image saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")
