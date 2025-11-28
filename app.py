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
    """
    Calculates Avg Length = Sum(P(c) * Bits(c))
    codes_map: dict {char: bit_length}
    """
    if not text:
        return 0.0

    counts = Counter(text)
    total_chars = len(text)
    avg_len = 0.0

    for char, count in counts.items():
        p = count / total_chars
        # Use provided bit length, or 0 if missing (shouldn't happen in valid encoding)
        bit_len = codes_map.get(char, 0)
        avg_len += p * bit_len

    return avg_len


# ---------------------------- RLE (Fixed with Delimiter) ----------------------------
def run_length_encode(data: str) -> str:
    if not data:
        return ""

    encoded = []
    count = 1
    # Use '|' as delimiter to separate count and character safely
    # Format: "count|char"

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
    # Loop through the string, parsing "number|char"
    i = 0
    n = len(encoded)
    while i < n:
        # 1. Parse Count (read until '|')
        count_str = ""
        while i < n and encoded[i] != "|":
            count_str += encoded[i]
            i += 1

        # Skip the '|'
        i += 1

        # 2. Parse Character (next char)
        if i < n:
            char = encoded[i]
            if count_str.isdigit():
                decoded.append(char * int(count_str))
            i += 1

    return "".join(decoded)


# ---------------------------- Huffman ----------------------------
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


# ---------------------------- Golomb ----------------------------
def unary_encode(q: int) -> str:
    return "1" * q + "0"


def golomb_encode(n: int, m: int) -> str:
    q = n // m
    r = n % m
    quotient_code = unary_encode(q)

    if (m & (m - 1)) == 0:  # Power of 2
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
    """Helper to get length of Golomb code for a value without building string"""
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


# ---------------------------- LZW ----------------------------
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


# ---------------------------- Uniform Quantization ----------------------------
def uniform_quantization(data, num_levels):
    data = np.array(data)
    data_min, data_max = np.min(data), np.max(data)
    if data_max == data_min:
        return data, 1.0
    step_size = (data_max - data_min) / num_levels
    quantized_data = np.floor((data - data_min) / step_size) * step_size + data_min
    return quantized_data.astype(np.uint8), step_size


# ==============================================================================
#  GUI IMPLEMENTATION
# ==============================================================================


class CompressionApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Compression Algorithms Project")
        self.geometry("1200x850")

        self.lossless_file_content = ""
        self.lossless_encoded_data = None
        self.lossless_extra_data = None

        self.lossy_original_filepath = None
        self.lossy_image = None
        self.lossy_quantized_image = None

        self.create_widgets()

    def create_widgets(self):
        main_tab_control = ttk.Notebook(self)
        self.tab_lossless = ttk.Frame(main_tab_control)
        self.tab_lossy = ttk.Frame(main_tab_control)

        main_tab_control.add(self.tab_lossless, text="Lossless Compression")
        main_tab_control.add(self.tab_lossy, text="Lossy Compression")
        main_tab_control.pack(expand=1, fill="both")

        self.setup_lossless_tab()
        self.setup_lossy_tab()

    # ================= LOSSLESS TAB =================
    def setup_lossless_tab(self):
        frame = self.tab_lossless

        # --- Left Panel ---
        left_panel = ttk.Frame(frame, padding=10)
        left_panel.pack(side="left", fill="y")

        ttk.Label(
            left_panel, text="1. Select Algorithm:", font=("Arial", 10, "bold")
        ).pack(anchor="w", pady=5)
        self.algo_var = tk.StringVar(value="RLE")
        algos = [
            ("Run-Length Encoding (RLE)", "RLE"),
            ("Huffman Coding", "Huffman"),
            ("Golomb Coding", "Golomb"),
            ("LZW Coding", "LZW"),
        ]
        for text, val in algos:
            ttk.Radiobutton(
                left_panel, text=text, variable=self.algo_var, value=val
            ).pack(anchor="w")

        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", pady=10)

        self.dnd_lbl = ttk.Label(
            left_panel,
            text="Drag & Drop Text File Here\n(or click to browse)",
            relief="sunken",
            anchor="center",
            background="#e1e1e1",
            padding=20,
        )
        self.dnd_lbl.pack(fill="x", pady=10, ipady=10)
        self.dnd_lbl.drop_target_register(DND_FILES)
        self.dnd_lbl.dnd_bind("<<Drop>>", self.on_drop_text_file)
        self.dnd_lbl.bind("<Button-1>", self.on_browse_text)

        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", pady=10)

        self.btn_compress = ttk.Button(
            left_panel, text="Compress Data", command=self.perform_compression
        )
        self.btn_compress.pack(fill="x", pady=5)

        self.btn_decompress = ttk.Button(
            left_panel,
            text="Decompress Result",
            command=self.perform_decompression,
            state="disabled",
        )
        self.btn_decompress.pack(fill="x", pady=5)

        # --- Statistics Panel (Lossless) ---
        stats_frame = ttk.LabelFrame(
            left_panel, text="Statistics Dashboard", padding=10
        )
        stats_frame.pack(fill="x", pady=20)

        # Grid layout for stats
        self.lbl_file_before = ttk.Label(stats_frame, text="Size Before: - bits")
        self.lbl_file_before.grid(row=0, column=0, sticky="w", pady=2)

        self.lbl_file_after = ttk.Label(stats_frame, text="Size After: - bits")
        self.lbl_file_after.grid(row=1, column=0, sticky="w", pady=2)

        self.lbl_ratio = ttk.Label(stats_frame, text="Compression Ratio: -")
        self.lbl_ratio.grid(row=2, column=0, sticky="w", pady=2)

        ttk.Separator(stats_frame, orient="horizontal").grid(
            row=3, column=0, sticky="ew", pady=5
        )

        self.lbl_entropy = ttk.Label(
            stats_frame, text="Entropy: -", foreground="purple"
        )
        self.lbl_entropy.grid(row=4, column=0, sticky="w", pady=2)

        self.lbl_avg_len = ttk.Label(stats_frame, text="Avg Length: - bits/sym")
        self.lbl_avg_len.grid(row=5, column=0, sticky="w", pady=2)

        self.lbl_eff = ttk.Label(
            stats_frame,
            text="Efficiency: -",
            foreground="green",
            font=("Arial", 9, "bold"),
        )
        self.lbl_eff.grid(row=6, column=0, sticky="w", pady=(5, 0))

        ttk.Label(
            stats_frame, text="(Entropy / Avg Length) * 100", font=("Arial", 7)
        ).grid(row=7, column=0, sticky="w")

        # --- Right Panel ---
        right_panel = ttk.Frame(frame, padding=10)
        right_panel.pack(side="right", fill="both", expand=True)

        ttk.Label(
            right_panel, text="Original Text Input:", font=("Arial", 9, "bold")
        ).pack(anchor="w")
        self.text_input = tk.Text(right_panel, height=8, width=50)
        self.text_input.pack(fill="x", pady=(0, 10))

        ttk.Label(
            right_panel, text="Compressed Output:", font=("Arial", 9, "bold")
        ).pack(anchor="w")
        self.text_output = tk.Text(right_panel, height=8, width=50, bg="#f0f8ff")
        self.text_output.pack(fill="x", pady=(0, 10))

        ttk.Label(
            right_panel, text="Decompressed Verification:", font=("Arial", 9, "bold")
        ).pack(anchor="w")
        self.text_check = tk.Text(right_panel, height=8, width=50, bg="#f0fff0")
        self.text_check.pack(fill="x")

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
        self.lbl_file_before.config(text="Size Before: - bits")
        self.lbl_file_after.config(text="Size After: - bits")
        self.lbl_ratio.config(text="Compression Ratio: -")
        self.lbl_entropy.config(text="Entropy: -")
        self.lbl_avg_len.config(text="Avg Length: -")
        self.lbl_eff.config(text="Efficiency: -")

    def update_lossless_stats(self, text, compressed_bits, avg_len_override=None):
        total_chars = len(text)
        if total_chars == 0:
            return

        original_bits = total_chars * 8

        # 1. Entropy
        entropy = calculate_entropy(text)

        # 2. Avg Length
        # If an override is provided (from Sum(P*Bits)), use it.
        # Otherwise, calculate effective average (Total Bits / Count).
        if avg_len_override:
            avg_length = avg_len_override
        else:
            avg_length = compressed_bits / total_chars

        # 3. Efficiency = (Entropy / Avg Length) * 100
        efficiency = ((entropy / avg_length) * 100) if avg_length > 0 else 0.0

        # 4. Compression Ratio
        ratio = original_bits / compressed_bits if compressed_bits > 0 else 0

        # Update Labels
        self.lbl_file_before.config(text=f"Size Before: {original_bits:,} bits")
        self.lbl_file_after.config(text=f"Size After: {compressed_bits:,} bits")
        self.lbl_ratio.config(text=f"Compression Ratio: {ratio:.2f}")
        self.lbl_entropy.config(text=f"Entropy: {entropy:.4f}")
        self.lbl_avg_len.config(text=f"Avg Length: {avg_length:.4f} bits/sym")
        self.lbl_eff.config(text=f"Efficiency: {efficiency:.2f}%")

    def perform_compression(self):
        algo = self.algo_var.get()
        text = self.text_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No text to compress.")
            return

        compressed_bits = 0
        avg_len_formula = None  # Will hold result of Sum(P*Bits) if applicable

        try:
            if algo == "RLE":
                # RLE encodes text -> "3|A2|B" -> calculate bits as chars * 8
                self.lossless_encoded_data = run_length_encode(text)
                display_text = self.lossless_encoded_data
                self.lossless_extra_data = None

                # Compressed size in bits (assuming ASCII output)
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

                # Calculate Avg Length using Sum(P * Bits)
                # codes map is {char: '010'} -> length is len('010')
                bit_lengths = {char: len(code) for char, code in codes.items()}
                avg_len_formula = calculate_avg_length_formula(text, bit_lengths)

            elif algo == "Golomb":
                # --- AUTOMATIC GRID SEARCH ---
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

                # Compress with Best M
                encoded_stream = ""
                # We can construct the bitstream or just calculate size.
                # Let's construct for display.
                # Map for Avg Length Calculation
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

                # Calculate Avg Length using Sum(P * Bits)
                avg_len_formula = calculate_avg_length_formula(text, bit_lengths)

            elif algo == "LZW":
                compressed, dictionary = lzw_encode(text)
                self.lossless_encoded_data = compressed
                self.lossless_extra_data = None
                display_text = f"Indices: {compressed}"
                # Estimate: Each index 12 bits
                compressed_bits = len(compressed) * 12

            self.text_output.delete(1.0, tk.END)
            self.text_output.insert(tk.END, display_text)
            self.btn_decompress.config(state="normal")

            # Update Statistics
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

    # ================= LOSSY TAB (Unchanged Logic) =================
    def setup_lossy_tab(self):
        frame = self.tab_lossy
        controls = ttk.Frame(frame, padding=10)
        controls.pack(side="top", fill="x")

        self.dnd_img_lbl = ttk.Label(
            controls,
            text="Drag & Drop Image Here",
            relief="sunken",
            anchor="center",
            background="#e1e1e1",
            padding=10,
        )
        self.dnd_img_lbl.pack(side="left", fill="x", expand=True, padx=5)
        self.dnd_img_lbl.drop_target_register(DND_FILES)
        self.dnd_img_lbl.dnd_bind("<<Drop>>", self.on_drop_image)
        self.dnd_img_lbl.bind("<Button-1>", self.on_browse_image)

        setting_frame = ttk.Frame(controls)
        setting_frame.pack(side="left", padx=20)
        ttk.Label(setting_frame, text="Quantization Levels:").pack(side="left")
        self.quant_levels = tk.IntVar(value=4)
        tk.Spinbox(
            setting_frame, from_=2, to=256, textvariable=self.quant_levels, width=5
        ).pack(side="left", padx=5)

        ttk.Button(
            controls, text="Apply Quantization", command=self.perform_quantization
        ).pack(side="left", padx=10)
        self.btn_save_img = ttk.Button(
            controls,
            text="Save Compressed Image",
            command=self.save_image,
            state="disabled",
        )
        self.btn_save_img.pack(side="left", padx=10)

        stats_frame = ttk.Frame(frame, padding=5)
        stats_frame.pack(side="top", fill="x", padx=10)
        self.lbl_lossy_orig_size = ttk.Label(
            stats_frame, text="Orig Size: - KB", font=("Arial", 9)
        )
        self.lbl_lossy_orig_size.pack(side="left", padx=10)
        self.lbl_lossy_comp_size = ttk.Label(
            stats_frame, text="Compressed Size: - KB", font=("Arial", 9)
        )
        self.lbl_lossy_comp_size.pack(side="left", padx=10)
        self.lbl_lossy_stats = ttk.Label(
            stats_frame,
            text="Ratio: - | Efficiency: -",
            font=("Arial", 9, "bold"),
            foreground="blue",
        )
        self.lbl_lossy_stats.pack(side="left", padx=10)

        self.img_display_frame = ttk.Frame(frame)
        self.img_display_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.lbl_orig_img = ttk.Label(self.img_display_frame, text="Original Image")
        self.lbl_orig_img.grid(row=0, column=0, padx=10)
        self.lbl_quant_img = ttk.Label(self.img_display_frame, text="Quantized Image")
        self.lbl_quant_img.grid(row=0, column=1, padx=10)
        self.canvas_orig = tk.Canvas(
            self.img_display_frame, bg="gray", width=500, height=500
        )
        self.canvas_orig.grid(row=1, column=0, sticky="nsew")
        self.canvas_quant = tk.Canvas(
            self.img_display_frame, bg="gray", width=500, height=500
        )
        self.canvas_quant.grid(row=1, column=1, sticky="nsew")
        self.img_display_frame.columnconfigure(0, weight=1)
        self.img_display_frame.columnconfigure(1, weight=1)

    def load_image(self, filepath):
        try:
            self.lossy_original_filepath = filepath
            img = Image.open(filepath)
            self.lossy_image = img
            file_size_bytes = os.path.getsize(filepath)
            self.lbl_lossy_orig_size.config(
                text=f"Orig Size: {file_size_bytes/1024:.2f} KB"
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
        self.lbl_lossy_comp_size.config(
            text=f"Est. Compressed: {comp_bytes/1024:.2f} KB"
        )
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
