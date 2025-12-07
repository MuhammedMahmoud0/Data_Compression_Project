import tkinter as tk
from tkinter import ttk
from tkinterdnd2 import TkinterDnD

from ..config import ModernStyle
from .tabs.lossless_tab import LosslessTab
from .tabs.lossy_tab import LossyTab


class CompressionApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Compression Algorithms Suite")
        self.geometry("1300x900")
        self.configure(bg=ModernStyle.BG_PRIMARY)

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Tabs
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

        # Buttons
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
        style.configure("TFrame", background=ModernStyle.BG_PRIMARY)
        style.configure(
            "TLabel",
            background=ModernStyle.BG_PRIMARY,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_NORMAL,
        )
        style.configure("Card.TLabel", background=ModernStyle.BG_SECONDARY)

    def create_widgets(self):
        # Header
        header = tk.Frame(self, bg=ModernStyle.BG_ACCENT, height=60)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(
            header,
            text="🗜️ Compression Algorithms Suite",
            bg=ModernStyle.BG_ACCENT,
            fg="white",
            font=ModernStyle.FONT_TITLE,
        ).pack(pady=15, padx=20, side="left")

        # Tabs Container
        content = tk.Frame(self, bg=ModernStyle.BG_PRIMARY)
        content.pack(fill="both", expand=True, padx=10, pady=10)

        main_tab_control = ttk.Notebook(content)

        # Instantiate Tab Classes
        self.tab_lossless = LosslessTab(main_tab_control)
        self.tab_lossy = LossyTab(main_tab_control)

        main_tab_control.add(self.tab_lossless, text="📝 Lossless Compression")
        main_tab_control.add(self.tab_lossy, text="🖼️ Lossy Compression")
        main_tab_control.pack(expand=1, fill="both")
