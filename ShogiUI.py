import customtkinter as ctk

class ShogiControlGUI(ctk.CTk):
    def __init__(self, key_callback):
        super().__init__()
        
        # Window Setup
        self.title("Shogi AI Control Center")
        self.geometry("420x600")
        self.key_callback = key_callback
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Move Output Section
        self.move_frame = ctk.CTkFrame(self)
        self.move_frame.pack(pady=15, padx=20, fill="both")

        self.label_side = ctk.CTkLabel(self.move_frame, text="SENTE TO MOVE", font=("Roboto", 12, "bold"), text_color="gold")
        self.label_side.pack(pady=5)

        self.move_val = ctk.CTkLabel(self.move_frame, text="--", font=("Roboto", 36, "bold"))
        self.move_val.pack(pady=5)

        self.explanation_val = ctk.CTkLabel(self.move_frame, text="Awaiting board detection...", wraplength=350)
        self.explanation_val.pack(pady=10)

        # Stats Area
        self.stats_label = ctk.CTkLabel(self, text="Value: 0.000 | Policy: 0.000", font=("Roboto", 11))
        self.stats_label.pack(pady=5)

        # Controls
        self.ctrl_frame = ctk.CTkFrame(self)
        self.ctrl_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # Labels for the button section
        ctk.CTkLabel(self.ctrl_frame, text="CONTROLS", font=("Roboto", 10, "bold")).pack(pady=5)

        btn_grid = [
            ("Toggle Side (T)", "t"), ("Re-detect Board (R)", "r"),
            ("AI Model (M)", "m"), ("Basic Engine (B)", "b"),
            ("Show Grid (A)", "a"), ("Save Debug (S)", "s"),
        ]

        for text, key in btn_grid:
            btn = ctk.CTkButton(self.ctrl_frame, text=text, command=lambda k=key: self.key_callback(k))
            btn.pack(pady=4, padx=15, fill="x")

        self.btn_quit = ctk.CTkButton(self.ctrl_frame, text="QUIT (Q)", fg_color="#721c24", hover_color="#af233a",
                                      command=lambda: self.key_callback("q"))
        self.btn_quit.pack(pady=15, padx=15, fill="x")

    def update_display(self, side, move, explanation, value=0.0, prob=0.0):
        # Update the UI elements from the main loop
        from shogiEngine import BLACK # Local import to avoid circular issues
        
        side_text = "SENTE (BLACK) TO MOVE" if side == BLACK else "GOTE (WHITE) TO MOVE"
        color = "gold" if side == BLACK else "#3b8ed0"
        
        self.label_side.configure(text=side_text, text_color=color)
        self.move_val.configure(text=move)
        self.explanation_val.configure(text=explanation)
        self.stats_label.configure(text=f"Value: {value:.3f} | Legal-Policy: {prob:.3f}")