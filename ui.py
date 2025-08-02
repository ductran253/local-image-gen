import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import os
import sys


if getattr(sys, 'frozen', False):
    backend_path = os.path.join(sys._MEIPASS, 'backend')
else:
    backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
sys.path.insert(0, backend_path)

from backend.model import generate_image


class DiffusionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local Image Generator")
        self.geometry("800x600")

        # UI Layout
        self.prompt_entry = ttk.Entry(self, font=("Arial", 14), width=40)
        self.prompt_entry.place(x=550, y=200)
        
        self.generate_btn = ttk.Button(self, text="Generate", command=self.on_generate)
        self.generate_btn.place(x=650, y=250)

        self.canvas = tk.Canvas(self, width=512, height=512, bg="white")
        self.canvas.place(x=10, y=40)

        self.status = ttk.Label(self, text="Ready", anchor="w")
        self.status.place(x=10, y=10)

    def on_generate(self):
        prompt = self.prompt_entry.get()
        if not prompt:
            messagebox.showwarning("No Prompt", "Please enter a prompt.")
            return
        self.status.config(text="Generating...")
        self.generate_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_generation, args=(prompt,)).start()

    def _run_generation(self, prompt):
        try:
            path = generate_image(prompt)
            img = Image.open(path)
            img.thumbnail((512, 512))
            tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=tk_img)
            self.canvas.image = tk_img
            self.status.config(text="Done")
        except Exception as e:
            self.status.config(text="Error")
            messagebox.showerror("Generation Failed", str(e))
        finally:
            self.generate_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = DiffusionApp()
    app.mainloop()