import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinterdnd2 import DND_FILES
import os
from collections import Counter

from ...config import ModernStyle
from ...algorithms import lossless, utils
from ...ui.components import UIHelpers


class LosslessTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.lossless_file_content = ""
        self.lossless_encoded_data = None
        self.lossless_extra_data = None
        self.setup_ui()

    def setup_ui(self):
        container = tk.Frame(self, bg=ModernStyle.BG_PRIMARY)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # Split Left/Right
        left_panel = tk.Frame(
            container, bg=ModernStyle.BG_SECONDARY, relief="flat", bd=1
        )
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.configure(width=320)
        left_panel.pack_propagate(False)

        right_panel = tk.Frame(
            container, bg=ModernStyle.BG_SECONDARY, relief="flat", bd=1
        )
        right_panel.pack(side="right", fill="both", expand=True)

        self._build_left_panel(left_panel)
        self._build_right_panel(right_panel)

    def _build_left_panel(self, parent):
        # Algorithm Selection
        algo_card = tk.Frame(parent, bg=ModernStyle.BG_SECONDARY)
        algo_card.pack(fill="x", padx=15, pady=15)

        tk.Label(
            algo_card,
            text="Select Algorithm",
            bg=ModernStyle.BG_SECONDARY,
            font=ModernStyle.FONT_HEADING,
        ).pack(anchor="w", pady=(0, 10))

        self.algo_var = tk.StringVar(value="RLE")
        algos = [
            ("🔄 Run-Length Encoding", "RLE"),
            ("🌳 Huffman Coding", "Huffman"),
            ("📊 Golomb Coding", "Golomb"),
            ("📚 LZW Coding", "LZW"),
        ]
        for text, val in algos:
            ttk.Radiobutton(
                algo_card, text=text, variable=self.algo_var, value=val
            ).pack(anchor="w", pady=3)

        tk.Frame(parent, bg=ModernStyle.BORDER_COLOR, height=1).pack(fill="x", pady=15)

        # Upload
        self._setup_upload_area(parent)

        # Buttons
        tk.Frame(parent, bg=ModernStyle.BORDER_COLOR, height=1).pack(fill="x", pady=15)
        btn_frame = tk.Frame(parent, bg=ModernStyle.BG_SECONDARY)
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))

        ttk.Button(
            btn_frame, text="▶ Compress Data", command=self.perform_compression
        ).pack(fill="x", pady=5)
        self.btn_decompress = ttk.Button(
            btn_frame,
            text="◀ Decompress Result",
            command=self.perform_decompression,
            state="disabled",
        )
        self.btn_decompress.pack(fill="x", pady=5)

        # Stats
        tk.Frame(parent, bg=ModernStyle.BORDER_COLOR, height=1).pack(fill="x", pady=15)
        self._setup_stats_area(parent)

    def _setup_upload_area(self, parent):
        upload_frame = tk.Frame(parent, bg=ModernStyle.BG_SECONDARY)
        upload_frame.pack(fill="x", padx=15)
        tk.Label(
            upload_frame,
            text="Input File",
            bg=ModernStyle.BG_SECONDARY,
            font=ModernStyle.FONT_HEADING,
        ).pack(anchor="w", pady=(0, 10))

        self.dnd_lbl = tk.Label(
            upload_frame,
            text="📁\n\nDrag & Drop Text File\nor click to browse",
            bg="#f8f9fa",
            fg=ModernStyle.TEXT_SECONDARY,
            relief="solid",
            bd=1,
            cursor="hand2",
            justify="center",
        )
        self.dnd_lbl.pack(fill="x", ipady=30)
        self.dnd_lbl.drop_target_register(DND_FILES)
        self.dnd_lbl.dnd_bind("<<Drop>>", self.on_drop_text_file)
        self.dnd_lbl.bind("<Button-1>", self.on_browse_text)

    def _setup_stats_area(self, parent):
        stats_frame = tk.Frame(parent, bg=ModernStyle.BG_SECONDARY)
        stats_frame.pack(fill="x", padx=15, pady=(0, 15))
        tk.Label(
            stats_frame,
            text="Statistics",
            bg=ModernStyle.BG_SECONDARY,
            font=ModernStyle.FONT_HEADING,
        ).pack(anchor="w", pady=(0, 10))

        self.lbl_file_before = UIHelpers.create_stat_label(
            stats_frame, "Size Before:", "- bytes (- bits, - KB)"
        )
        self.lbl_file_after = UIHelpers.create_stat_label(
            stats_frame, "Size After:", "- bytes (- bits, - KB)"
        )
        self.lbl_ratio = UIHelpers.create_stat_label(
            stats_frame, "Compression Ratio:", "-"
        )
        tk.Frame(stats_frame, bg=ModernStyle.BORDER_COLOR, height=1).pack(
            fill="x", pady=8
        )
        self.lbl_entropy = UIHelpers.create_stat_label(
            stats_frame, "Entropy:", "-", ModernStyle.TEXT_ACCENT
        )
        self.lbl_avg_len = UIHelpers.create_stat_label(
            stats_frame, "Avg Length:", "- bits/sym"
        )
        self.lbl_eff = UIHelpers.create_stat_label(
            stats_frame, "Efficiency:", "-", ModernStyle.SUCCESS_COLOR
        )

    def _build_right_panel(self, parent):
        right_inner = tk.Frame(parent, bg=ModernStyle.BG_SECONDARY)
        right_inner.pack(fill="both", expand=True, padx=20, pady=20)
        self.text_input = UIHelpers.create_text_section(
            right_inner, "Original Text Input"
        )
        self.text_output = UIHelpers.create_text_section(
            right_inner, "Compressed Output", bg="#f0f8ff"
        )
        self.text_check = UIHelpers.create_text_section(
            right_inner, "Decompressed Verification", bg="#f0fff0"
        )

    # --- Logic ---
    def load_text_content(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            self.text_input.delete(1.0, tk.END)
            self.text_input.insert(tk.END, content)
            self.text_output.delete(1.0, tk.END)
            self.text_check.delete(1.0, tk.END)
            self.btn_decompress.config(state="disabled")
            self.dnd_lbl.config(
                text=f"✓ Loaded\n{os.path.basename(filepath)}",
                bg="#d4edda",
                fg=ModernStyle.SUCCESS_COLOR,
            )
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {e}")

    def on_drop_text_file(self, event):
        filepath = event.data.strip("{}")
        self.load_text_content(filepath)

    def on_browse_text(self, event):
        filepath = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filepath:
            self.load_text_content(filepath)

    def update_stats(self, text, compressed_bits, avg_len_override=None):
        total_chars = len(text)
        if total_chars == 0:
            return
        original_bits = total_chars * 8
        entropy = utils.calculate_entropy(text)
        avg_length = (
            avg_len_override if avg_len_override else (compressed_bits / total_chars)
        )
        efficiency = ((entropy / avg_length) * 100) if avg_length > 0 else 0.0
        ratio = original_bits / compressed_bits if compressed_bits > 0 else 0

        # show bits and also provide byte and KB representation
        orig_bytes = original_bits // 8
        comp_bytes = compressed_bits // 8
        orig_kb = orig_bytes / 1024.0
        comp_kb = comp_bytes / 1024.0
        self.lbl_file_before.config(
            text=f"{orig_bytes:,} bytes ({original_bits:,} bits, {orig_kb:.2f} KB)"
        )
        self.lbl_file_after.config(
            text=f"{comp_bytes:,} bytes ({compressed_bits:,} bits, {comp_kb:.2f} KB)"
        )
        self.lbl_ratio.config(text=f"{ratio:.2f}")
        self.lbl_entropy.config(text=f"{entropy:.4f}")
        self.lbl_avg_len.config(text=f"{avg_length:.4f}")
        self.lbl_eff.config(text=f"{efficiency:.2f}%")

    def perform_compression(self):
        algo = self.algo_var.get()
        text = self.text_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No text to compress.")
            return

        try:
            compressed_bits = 0
            avg_len_formula = None
            display_text = ""

            if algo == "RLE":
                self.lossless_encoded_data = lossless.run_length_encode(text)
                display_text = self.lossless_encoded_data
                compressed_bits = len(self.lossless_encoded_data) * 8

            elif algo == "Huffman":
                freq = utils.build_frequency(text)
                heap = lossless.build_heap(freq)
                codes = lossless.build_codes(heap)
                encoded = lossless.huffman_encode(text, codes)
                self.lossless_encoded_data = encoded
                self.lossless_extra_data = codes
                display_text = f"Bits: {encoded}\n\nCodes: {codes}"
                compressed_bits = len(encoded)
                bit_lengths = {char: len(code) for char, code in codes.items()}
                avg_len_formula = utils.calculate_avg_length_formula(text, bit_lengths)

            elif algo == "Golomb":
                freqs = Counter(text)
                best_m = 2
                min_total_bits = float("inf")
                # Simple optimization to find best M
                for m_candidate in range(2, 256):
                    current_bits = 0
                    for char, freq in freqs.items():
                        current_bits += (
                            lossless.get_golomb_bits_len(ord(char), m_candidate) * freq
                        )
                    if current_bits < min_total_bits:
                        min_total_bits = current_bits
                        best_m = m_candidate

                encoded_stream = ""
                bit_lengths = {}
                for char in text:
                    code = lossless.golomb_encode(ord(char), best_m)
                    encoded_stream += code
                    if char not in bit_lengths:
                        bit_lengths[char] = len(code)

                self.lossless_encoded_data = encoded_stream
                self.lossless_extra_data = best_m
                display_text = f"Best M found: {best_m}\nBitstream:\n{encoded_stream}"
                compressed_bits = len(encoded_stream)
                avg_len_formula = utils.calculate_avg_length_formula(text, bit_lengths)

            elif algo == "LZW":
                compressed, dictionary = lossless.lzw_encode(text)
                self.lossless_encoded_data = compressed
                display_text = f"Indices: {compressed}"
                compressed_bits = len(compressed) * 12

            self.text_output.delete(1.0, tk.END)
            self.text_output.insert(tk.END, display_text)
            self.btn_decompress.config(state="normal")
            self.update_stats(text, compressed_bits, avg_len_formula)

        except Exception as e:
            messagebox.showerror("Compression Error", str(e))

    def perform_decompression(self):
        algo = self.algo_var.get()
        try:
            result = ""
            if algo == "RLE":
                result = lossless.run_length_decode(self.lossless_encoded_data)
            elif algo == "Huffman":
                result = lossless.huffman_decode(
                    self.lossless_encoded_data, self.lossless_extra_data
                )
            elif algo == "Golomb":
                result = lossless.golomb_decode_stream(
                    self.lossless_encoded_data, self.lossless_extra_data
                )
            elif algo == "LZW":
                result = lossless.lzw_decode(self.lossless_encoded_data)

            self.text_check.delete(1.0, tk.END)
            self.text_check.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Decompression Error", str(e))
