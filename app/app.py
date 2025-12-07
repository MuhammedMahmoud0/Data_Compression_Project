import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import numpy as np
import math
from collections import Counter
import heapq
import os
import io

# ==============================================================================
#  MODERN UI STYLING
# ==============================================================================


class ModernStyle:
    """Modern color scheme and styling constants"""

    BG_PRIMARY = "#f5f6fa"
    BG_SECONDARY = "#ffffff"
    BG_ACCENT = "#4a90e2"
    BG_ACCENT_HOVER = "#357abd"
    TEXT_PRIMARY = "#2c3e50"
    TEXT_SECONDARY = "#7f8c8d"
    TEXT_ACCENT = "#4a90e2"
    BORDER_COLOR = "#dfe4ea"
    SUCCESS_COLOR = "#27ae60"
    WARNING_COLOR = "#f39c12"
    ERROR_COLOR = "#e74c3c"
    FONT_FAMILY = "Segoe UI"
    FONT_TITLE = (FONT_FAMILY, 16, "bold")
    FONT_HEADING = (FONT_FAMILY, 11, "bold")
    FONT_NORMAL = (FONT_FAMILY, 10)
    FONT_SMALL = (FONT_FAMILY, 9)


# ==============================================================================
#  ALGORITHMS & HELPERS
# ==============================================================================


def calculate_entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    total_chars = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total_chars
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_avg_length_formula(text, codes_map):
    if not text:
        return 0.0
    counts = Counter(text)
    total_chars = len(text)
    avg_len = 0.0
    for char, count in counts.items():
        p = count / total_chars
        bit_len = codes_map.get(char, 0)
        avg_len += p * bit_len
    return avg_len


def run_length_encode(data: str) -> str:
    if not data:
        return ""
    encoded = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded.append(f"{count}|{data[i - 1]}")
            count = 1
    encoded.append(f"{count}|{data[-1]}")
    return "".join(encoded)


def run_length_decode(encoded: str) -> str:
    decoded = []
    i = 0
    n = len(encoded)
    while i < n:
        count_str = ""
        while i < n and encoded[i] != "|":
            count_str += encoded[i]
            i += 1
        i += 1
        if i < n:
            char = encoded[i]
            if count_str.isdigit():
                decoded.append(char * int(count_str))
            i += 1
    return "".join(decoded)


def build_frequency(text):
    return Counter(text)


def build_heap(freq):
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)
    return heap


def build_codes(heap):
    local_heap = heap[:]
    while len(local_heap) > 1:
        smallest = heapq.heappop(local_heap)
        secsmallest = heapq.heappop(local_heap)
        for pair in smallest[1:]:
            pair[1] = "0" + pair[1]
        for pair in secsmallest[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(
            local_heap, [smallest[0] + secsmallest[0]] + smallest[1:] + secsmallest[1:]
        )
    if not local_heap:
        return {}
    return dict(local_heap[0][1:])


def huffman_encode(text, codes):
    return "".join(codes[ch] for ch in text)


def huffman_decode(encoded_text, codes):
    reverse_codes = {v: k for k, v in codes.items()}
    current_code = ""
    decoded_text = ""
    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_codes:
            decoded_text += reverse_codes[current_code]
            current_code = ""
    return decoded_text


def unary_encode(q: int) -> str:
    return "1" * q + "0"


def golomb_encode(n: int, m: int) -> str:
    q = n // m
    r = n % m
    quotient_code = unary_encode(q)
    if (m & (m - 1)) == 0:
        k = int(math.log2(m))
        remainder_code = format(r, f"0{k}b")
    else:
        b = math.ceil(math.log2(m))
        T = 2**b - m
        if r < T:
            remainder_code = format(r, f"0{b-1}b")
        else:
            remainder_code = format(r + T, f"0{b}b")
    return quotient_code + remainder_code


def golomb_decode_stream(bitstream, m):
    decoded_text = ""
    idx = 0
    n_len = len(bitstream)
    if (m & (m - 1)) == 0:
        k = int(math.log2(m))
        is_power_2 = True
    else:
        b = math.ceil(math.log2(m))
        T = 2**b - m
        is_power_2 = False
    while idx < n_len:
        q = 0
        while idx < n_len and bitstream[idx] == "1":
            q += 1
            idx += 1
        idx += 1
        r = 0
        if is_power_2:
            if idx + k > n_len:
                break
            r_bin = bitstream[idx : idx + k]
            r = int(r_bin, 2)
            idx += k
        else:
            if idx + (b - 1) > n_len:
                break
            temp_r = int(bitstream[idx : idx + (b - 1)], 2)
            if temp_r < T:
                r = temp_r
                idx += b - 1
            else:
                if idx + b > n_len:
                    break
                temp_r = int(bitstream[idx : idx + b], 2)
                r = temp_r - T
                idx += b
        val = q * m + r
        try:
            decoded_text += chr(val)
        except:
            decoded_text += "?"
    return decoded_text


def get_golomb_bits_len(val, m):
    q = val // m
    r = val % m
    len_unary = q + 1
    if (m & (m - 1)) == 0:
        k = int(math.log2(m))
        len_rem = k
    else:
        b = math.ceil(math.log2(m))
        T = 2**b - m
        if r < T:
            len_rem = b - 1
        else:
            len_rem = b
    return len_unary + len_rem


def lzw_encode(text):
    dictionary = {chr(i): i for i in range(256)}
    next_code = 256
    current_c = ""
    result = []
    for next_n in text:
        combined = current_c + next_n
        if combined in dictionary:
            current_c = combined
        else:
            result.append(dictionary[current_c])
            dictionary[combined] = next_code
            next_code += 1
            current_c = next_n
    if current_c != "":
        result.append(dictionary[current_c])
    return result, dictionary


def lzw_decode(codes):
    dictionary = {i: chr(i) for i in range(256)}
    next_code = 256
    if not codes:
        return ""
    prev_entry = dictionary[codes[0]]
    result = prev_entry
    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        else:
            entry = prev_entry + prev_entry[0]
        result += entry
        dictionary[next_code] = prev_entry + entry[0]
        next_code += 1
        prev_entry = entry
    return result


def uniform_quantization(data, num_levels):
    data = np.array(data)
    data_min, data_max = np.min(data), np.max(data)
    if data_max == data_min:
        return data, 1.0
    step_size = (data_max - data_min) / num_levels
    quantized_data = np.floor((data - data_min) / step_size) * step_size + data_min
    return quantized_data.astype(np.uint8), step_size


# ==============================================================================
#  MAIN APPLICATION
# ==============================================================================


class CompressionApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Compression Algorithms Suite")
        self.geometry("1300x900")
        self.configure(bg=ModernStyle.BG_PRIMARY)

        self.lossless_file_content = ""
        self.lossless_encoded_data = None
        self.lossless_extra_data = None
        self.lossy_original_filepath = None
        self.lossy_image = None
        self.lossy_quantized_image = None

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TNotebook", background=ModernStyle.BG_PRIMARY, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            padding=[20, 10],
            background=ModernStyle.BG_SECONDARY,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_HEADING,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", ModernStyle.BG_ACCENT)],
            foreground=[("selected", "white")],
        )

        style.configure("Card.TFrame", background=ModernStyle.BG_SECONDARY)
        style.configure("TFrame", background=ModernStyle.BG_PRIMARY)
        style.configure(
            "TLabel",
            background=ModernStyle.BG_PRIMARY,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_NORMAL,
        )
        style.configure("Card.TLabel", background=ModernStyle.BG_SECONDARY)
        style.configure(
            "TButton",
            font=ModernStyle.FONT_NORMAL,
            borderwidth=0,
            relief="flat",
            padding=[20, 8],
        )
        style.configure(
            "TRadiobutton",
            background=ModernStyle.BG_SECONDARY,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_NORMAL,
        )

    def create_widgets(self):
        header = tk.Frame(self, bg=ModernStyle.BG_ACCENT, height=60)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        title_label = tk.Label(
            header,
            text="🗜️ Compression Algorithms Suite",
            bg=ModernStyle.BG_ACCENT,
            fg="white",
            font=ModernStyle.FONT_TITLE,
        )
        title_label.pack(pady=15, padx=20, side="left")

        content = tk.Frame(self, bg=ModernStyle.BG_PRIMARY)
        content.pack(fill="both", expand=True, padx=10, pady=10)

        main_tab_control = ttk.Notebook(content)
        self.tab_lossless = ttk.Frame(main_tab_control)
        self.tab_lossy = ttk.Frame(main_tab_control)

        main_tab_control.add(self.tab_lossless, text="📝 Lossless Compression")
        main_tab_control.add(self.tab_lossy, text="🖼️ Lossy Compression")
        main_tab_control.pack(expand=1, fill="both")

        self.setup_lossless_tab()
        self.setup_lossy_tab()

    def setup_lossless_tab(self):
        frame = self.tab_lossless
        container = tk.Frame(frame, bg=ModernStyle.BG_PRIMARY)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # Left Panel
        left_panel = tk.Frame(
            container, bg=ModernStyle.BG_SECONDARY, relief="flat", bd=1
        )
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.configure(width=320)
        left_panel.pack_propagate(False)

        algo_card = tk.Frame(left_panel, bg=ModernStyle.BG_SECONDARY)
        algo_card.pack(fill="x", padx=15, pady=15)

        tk.Label(
            algo_card,
            text="Select Algorithm",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
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
                algo_card,
                text=text,
                variable=self.algo_var,
                value=val,
                style="TRadiobutton",
            ).pack(anchor="w", pady=3)

        tk.Frame(left_panel, bg=ModernStyle.BORDER_COLOR, height=1).pack(
            fill="x", pady=15
        )

        upload_frame = tk.Frame(left_panel, bg=ModernStyle.BG_SECONDARY)
        upload_frame.pack(fill="x", padx=15)

        tk.Label(
            upload_frame,
            text="Input File",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_HEADING,
        ).pack(anchor="w", pady=(0, 10))

        self.dnd_lbl = tk.Label(
            upload_frame,
            text="📁\n\nDrag & Drop Text File\nor click to browse",
            bg="#f8f9fa",
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_NORMAL,
            relief="solid",
            bd=1,
            cursor="hand2",
            justify="center",
        )
        self.dnd_lbl.pack(fill="x", ipady=30)
        self.dnd_lbl.drop_target_register(DND_FILES)
        self.dnd_lbl.dnd_bind("<<Drop>>", self.on_drop_text_file)
        self.dnd_lbl.bind("<Button-1>", self.on_browse_text)

        tk.Frame(left_panel, bg=ModernStyle.BORDER_COLOR, height=1).pack(
            fill="x", pady=15
        )

        btn_frame = tk.Frame(left_panel, bg=ModernStyle.BG_SECONDARY)
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.btn_compress = ttk.Button(
            btn_frame, text="▶ Compress Data", command=self.perform_compression
        )
        self.btn_compress.pack(fill="x", pady=5)

        self.btn_decompress = ttk.Button(
            btn_frame,
            text="◀ Decompress Result",
            command=self.perform_decompression,
            state="disabled",
        )
        self.btn_decompress.pack(fill="x", pady=5)

        tk.Frame(left_panel, bg=ModernStyle.BORDER_COLOR, height=1).pack(
            fill="x", pady=15
        )

        stats_frame = tk.Frame(left_panel, bg=ModernStyle.BG_SECONDARY)
        stats_frame.pack(fill="x", padx=15, pady=(0, 15))

        tk.Label(
            stats_frame,
            text="Statistics",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_HEADING,
        ).pack(anchor="w", pady=(0, 10))

        stats_grid = tk.Frame(stats_frame, bg=ModernStyle.BG_SECONDARY)
        stats_grid.pack(fill="x")

        self.lbl_file_before = self.create_stat_label(
            stats_grid, "Size Before:", "- bits"
        )
        self.lbl_file_after = self.create_stat_label(
            stats_grid, "Size After:", "- bits"
        )
        self.lbl_ratio = self.create_stat_label(stats_grid, "Compression Ratio:", "-")

        tk.Frame(stats_grid, bg=ModernStyle.BORDER_COLOR, height=1).pack(
            fill="x", pady=8
        )

        self.lbl_entropy = self.create_stat_label(
            stats_grid, "Entropy:", "-", ModernStyle.TEXT_ACCENT
        )
        self.lbl_avg_len = self.create_stat_label(
            stats_grid, "Avg Length:", "- bits/sym"
        )
        self.lbl_eff = self.create_stat_label(
            stats_grid, "Efficiency:", "-", ModernStyle.SUCCESS_COLOR
        )

        # Right Panel
        right_panel = tk.Frame(
            container, bg=ModernStyle.BG_SECONDARY, relief="flat", bd=1
        )
        right_panel.pack(side="right", fill="both", expand=True)

        right_inner = tk.Frame(right_panel, bg=ModernStyle.BG_SECONDARY)
        right_inner.pack(fill="both", expand=True, padx=20, pady=20)

        self.create_text_section(right_inner, "Original Text Input", "text_input")
        self.create_text_section(
            right_inner, "Compressed Output", "text_output", bg="#f0f8ff"
        )
        self.create_text_section(
            right_inner, "Decompressed Verification", "text_check", bg="#f0fff0"
        )

    def create_stat_label(self, parent, label, value, color=None):
        frame = tk.Frame(parent, bg=ModernStyle.BG_SECONDARY)
        frame.pack(fill="x", pady=3)

        tk.Label(
            frame,
            text=label,
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SMALL,
            anchor="w",
        ).pack(side="left")

        val_label = tk.Label(
            frame,
            text=value,
            bg=ModernStyle.BG_SECONDARY,
            fg=color or ModernStyle.TEXT_PRIMARY,
            font=(ModernStyle.FONT_FAMILY, 10, "bold"),
            anchor="e",
        )
        val_label.pack(side="right")
        return val_label

    def create_text_section(self, parent, title, attr_name, bg="#ffffff"):
        section = tk.Frame(parent, bg=ModernStyle.BG_SECONDARY)
        section.pack(fill="both", expand=True, pady=(0, 15))

        tk.Label(
            section,
            text=title,
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_HEADING,
        ).pack(anchor="w", pady=(0, 5))

        text_widget = tk.Text(
            section,
            height=8,
            width=50,
            font=ModernStyle.FONT_NORMAL,
            bg=bg,
            relief="solid",
            bd=1,
            padx=10,
            pady=10,
        )
        text_widget.pack(fill="both", expand=True)
        setattr(self, attr_name, text_widget)

    def load_text_content(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            self.lossless_file_content = content
            self.text_input.delete(1.0, tk.END)
            self.text_input.insert(tk.END, content)
            self.text_output.delete(1.0, tk.END)
            self.text_check.delete(1.0, tk.END)
            self.btn_decompress.config(state="disabled")
            self.reset_lossless_stats()
            self.dnd_lbl.config(
                text=f"✓ Loaded\n{os.path.basename(filepath)}",
                bg="#d4edda",
                fg=ModernStyle.SUCCESS_COLOR,
            )
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {e}")

    def on_drop_text_file(self, event):
        filepath = event.data
        if filepath.startswith("{") and filepath.endswith("}"):
            filepath = filepath[1:-1]
        self.load_text_content(filepath)

    def on_browse_text(self, event):
        filepath = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filepath:
            self.load_text_content(filepath)

    def reset_lossless_stats(self):
        self.lbl_file_before.config(text="- bits")
        self.lbl_file_after.config(text="- bits")
        self.lbl_ratio.config(text="-")
        self.lbl_entropy.config(text="-")
        self.lbl_avg_len.config(text="-")
        self.lbl_eff.config(text="-")

    def update_lossless_stats(self, text, compressed_bits, avg_len_override=None):
        total_chars = len(text)
        if total_chars == 0:
            return
        original_bits = total_chars * 8
        entropy = calculate_entropy(text)
        if avg_len_override:
            avg_length = avg_len_override
        else:
            avg_length = compressed_bits / total_chars
        efficiency = ((entropy / avg_length) * 100) if avg_length > 0 else 0.0
        ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        self.lbl_file_before.config(text=f"{original_bits:,}")
        self.lbl_file_after.config(text=f"{compressed_bits:,}")
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
        compressed_bits = 0
        avg_len_formula = None
        try:
            if algo == "RLE":
                self.lossless_encoded_data = run_length_encode(text)
                display_text = self.lossless_encoded_data
                self.lossless_extra_data = None
                compressed_bits = len(self.lossless_encoded_data) * 8
            elif algo == "Huffman":
                freq = build_frequency(text)
                heap = build_heap(freq)
                codes = build_codes(heap)
                encoded = huffman_encode(text, codes)
                self.lossless_encoded_data = encoded
                self.lossless_extra_data = codes
                display_text = f"Bits: {encoded}\n\nCodes: {codes}"
                compressed_bits = len(encoded)
                bit_lengths = {char: len(code) for char, code in codes.items()}
                avg_len_formula = calculate_avg_length_formula(text, bit_lengths)
            elif algo == "Golomb":
                freqs = Counter(text)
                best_m = 2
                min_total_bits = float("inf")
                for m_candidate in range(2, 256):
                    current_bits = 0
                    for char, freq in freqs.items():
                        bit_len = get_golomb_bits_len(ord(char), m_candidate)
                        current_bits += bit_len * freq
                    if current_bits < min_total_bits:
                        min_total_bits = current_bits
                        best_m = m_candidate
                encoded_stream = ""
                bit_lengths = {}
                for char in text:
                    code = golomb_encode(ord(char), best_m)
                    encoded_stream += code
                    if char not in bit_lengths:
                        bit_lengths[char] = len(code)
                self.lossless_encoded_data = encoded_stream
                self.lossless_extra_data = best_m
                display_text = f"Best M found: {best_m}\nBitstream:\n{encoded_stream}"
                compressed_bits = len(encoded_stream)
                avg_len_formula = calculate_avg_length_formula(text, bit_lengths)
            elif algo == "LZW":
                compressed, dictionary = lzw_encode(text)
                self.lossless_encoded_data = compressed
                self.lossless_extra_data = None
                display_text = f"Indices: {compressed}"
                compressed_bits = len(compressed) * 12
            self.text_output.delete(1.0, tk.END)
            self.text_output.insert(tk.END, display_text)
            self.btn_decompress.config(state="normal")
            self.update_lossless_stats(text, compressed_bits, avg_len_formula)
        except Exception as e:
            messagebox.showerror("Compression Error", str(e))

    def perform_decompression(self):
        algo = self.algo_var.get()
        try:
            result = ""
            if algo == "RLE":
                result = run_length_decode(self.lossless_encoded_data)
            elif algo == "Huffman":
                result = huffman_decode(
                    self.lossless_encoded_data, self.lossless_extra_data
                )
            elif algo == "Golomb":
                result = golomb_decode_stream(
                    self.lossless_encoded_data, self.lossless_extra_data
                )
            elif algo == "LZW":
                result = lzw_decode(self.lossless_encoded_data)
            self.text_check.delete(1.0, tk.END)
            self.text_check.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Decompression Error", str(e))

    def setup_lossy_tab(self):
        frame = self.tab_lossy
        container = tk.Frame(frame, bg=ModernStyle.BG_PRIMARY)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        top_card = tk.Frame(container, bg=ModernStyle.BG_SECONDARY, relief="flat", bd=1)
        top_card.pack(fill="x", pady=(0, 15))

        controls = tk.Frame(top_card, bg=ModernStyle.BG_SECONDARY)
        controls.pack(fill="x", padx=20, pady=15)

        upload_section = tk.Frame(controls, bg=ModernStyle.BG_SECONDARY)
        upload_section.pack(side="left", fill="x", expand=True)

        self.dnd_img_lbl = tk.Label(
            upload_section,
            text="📁 Drag & Drop Image Here or Click to Browse",
            bg="#f8f9fa",
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_NORMAL,
            relief="solid",
            bd=1,
            cursor="hand2",
        )
        self.dnd_img_lbl.pack(fill="x", ipady=15)
        self.dnd_img_lbl.drop_target_register(DND_FILES)
        self.dnd_img_lbl.dnd_bind("<<Drop>>", self.on_drop_image)
        self.dnd_img_lbl.bind("<Button-1>", self.on_browse_image)

        settings_section = tk.Frame(controls, bg=ModernStyle.BG_SECONDARY)
        settings_section.pack(side="left", padx=20)

        tk.Label(
            settings_section,
            text="Quantization Levels:",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_NORMAL,
        ).pack(side="left", padx=(0, 10))

        self.quant_levels = tk.IntVar(value=4)
        tk.Spinbox(
            settings_section,
            from_=2,
            to=256,
            textvariable=self.quant_levels,
            width=6,
            font=ModernStyle.FONT_NORMAL,
        ).pack(side="left")

        btn_section = tk.Frame(controls, bg=ModernStyle.BG_SECONDARY)
        btn_section.pack(side="left", padx=10)

        ttk.Button(
            btn_section, text="▶ Apply Quantization", command=self.perform_quantization
        ).pack(side="left", padx=5)

        self.btn_save_img = ttk.Button(
            btn_section, text="💾 Save Image", command=self.save_image, state="disabled"
        )
        self.btn_save_img.pack(side="left", padx=5)

        stats_card = tk.Frame(
            container, bg=ModernStyle.BG_SECONDARY, relief="flat", bd=1
        )
        stats_card.pack(fill="x", pady=(0, 15))

        stats_inner = tk.Frame(stats_card, bg=ModernStyle.BG_SECONDARY)
        stats_inner.pack(fill="x", padx=20, pady=10)

        self.lbl_lossy_orig_size = tk.Label(
            stats_inner,
            text="Original Size: - KB",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_NORMAL,
        )
        self.lbl_lossy_orig_size.pack(side="left", padx=10)

        self.lbl_lossy_comp_size = tk.Label(
            stats_inner,
            text="Compressed Size: - KB",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_NORMAL,
        )
        self.lbl_lossy_comp_size.pack(side="left", padx=10)

        self.lbl_lossy_stats = tk.Label(
            stats_inner,
            text="Ratio: - | Efficiency: -",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.SUCCESS_COLOR,
            font=(ModernStyle.FONT_FAMILY, 10, "bold"),
        )
        self.lbl_lossy_stats.pack(side="left", padx=10)

        images_card = tk.Frame(
            container, bg=ModernStyle.BG_SECONDARY, relief="flat", bd=1
        )
        images_card.pack(fill="both", expand=True)

        self.img_display_frame = tk.Frame(images_card, bg=ModernStyle.BG_SECONDARY)
        self.img_display_frame.pack(fill="both", expand=True, padx=20, pady=20)

        left_img = tk.Frame(self.img_display_frame, bg=ModernStyle.BG_SECONDARY)
        left_img.grid(row=0, column=0, padx=10, sticky="nsew")

        tk.Label(
            left_img,
            text="Original Image",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_HEADING,
        ).pack(pady=(0, 10))

        self.canvas_orig = tk.Canvas(
            left_img,
            bg="#f8f9fa",
            width=500,
            height=500,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER_COLOR,
        )
        self.canvas_orig.pack()

        right_img = tk.Frame(self.img_display_frame, bg=ModernStyle.BG_SECONDARY)
        right_img.grid(row=0, column=1, padx=10, sticky="nsew")

        tk.Label(
            right_img,
            text="Quantized Image",
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_HEADING,
        ).pack(pady=(0, 10))

        self.canvas_quant = tk.Canvas(
            right_img,
            bg="#f8f9fa",
            width=500,
            height=500,
            highlightthickness=1,
            highlightbackground=ModernStyle.BORDER_COLOR,
        )
        self.canvas_quant.pack()

        self.img_display_frame.columnconfigure(0, weight=1)
        self.img_display_frame.columnconfigure(1, weight=1)

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
            display_img = img.copy()
            display_img.thumbnail((500, 500))
            tk_img = ImageTk.PhotoImage(display_img)
            self.canvas_orig.create_image(250, 250, image=tk_img, anchor="center")
            self.canvas_orig.image = tk_img
            self.canvas_quant.delete("all")
            self.dnd_img_lbl.config(
                text=f"✓ Loaded: {os.path.basename(filepath)}",
                bg="#d4edda",
                fg=ModernStyle.SUCCESS_COLOR,
            )
        except Exception as e:
            messagebox.showerror("Error", f"Invalid Image: {e}")

    def on_drop_image(self, event):
        filepath = event.data
        if filepath.startswith("{") and filepath.endswith("}"):
            filepath = filepath[1:-1]
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
        quantized_arr, _ = uniform_quantization(img_array, levels)
        self.lossy_quantized_image = Image.fromarray(quantized_arr)
        display_img = self.lossy_quantized_image.copy()
        display_img.thumbnail((500, 500))
        tk_img = ImageTk.PhotoImage(display_img)
        self.canvas_quant.create_image(250, 250, image=tk_img, anchor="center")
        self.canvas_quant.image = tk_img
        self.btn_save_img.config(state="normal")
        self.calculate_lossy_stats()

    def calculate_lossy_stats(self):
        if not self.lossy_original_filepath or not self.lossy_quantized_image:
            return
        orig_bytes = os.path.getsize(self.lossy_original_filepath)
        buffer = io.BytesIO()
        try:
            fmt = self.lossy_image.format if self.lossy_image.format else "PNG"
            self.lossy_quantized_image.save(buffer, format=fmt)
            comp_bytes = buffer.tell()
        except:
            self.lossy_quantized_image.save(buffer, format="PNG")
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


if __name__ == "__main__":
    app = CompressionApp()
    app.mainloop()
