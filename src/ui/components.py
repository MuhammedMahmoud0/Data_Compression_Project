import tkinter as tk
from ..config import ModernStyle


class UIHelpers:
    @staticmethod
    def create_stat_label(parent, label, value, color=None):
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

    @staticmethod
    def create_text_section(parent, title, bg="#ffffff"):
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
        return text_widget
