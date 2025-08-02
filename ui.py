import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import os
import sys
import transformers


if getattr(sys, 'frozen', False):
    backend_path = os.path.join(sys._MEIPASS, 'backend')
else:
    backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, backend_path)

from backend.model import generate_image, is_pipeline_loaded


class DiffusionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local Image Generator")
        self.geometry("800x600")
        self.model_loaded = False

        # UI Layout
        self.prompt_entry = ttk.Entry(self, font=("Arial", 14), width=40)
        self.prompt_entry.place(x=550, y=200)
        
        self.generate_btn = ttk.Button(self, text="Generate", command=self.on_generate)
        self.generate_btn.place(x=650, y=250)

        self.canvas = tk.Canvas(self, width=512, height=512, bg="white")
        self.canvas.place(x=10, y=40)

        self.status = ttk.Label(self, text="Ready - Model will load on first generation", anchor="w")
        self.status.place(x=10, y=10)

        # Progress bar for model loading
        self.progress = ttk.Progressbar(self, mode='indeterminate')
        self.progress.place(x=550, y=300, width=200)
        self.progress.place_forget()  # Hide initially

    def on_generate(self):
        prompt = self.prompt_entry.get()
        if not prompt:
            messagebox.showwarning("No Prompt", "Please enter a prompt.")
            return
        
        # Check if pipeline is loaded
        if not is_pipeline_loaded():
            self.status.config(text="Loading model... (this may take a while)")
            self.progress.place(x=550, y=300, width=200)
            self.progress.start()
        else:
            self.status.config(text="Generating...")
            
        self.generate_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_generation, args=(prompt,)).start()

    def _run_generation(self, prompt):
        try:
            was_loaded = is_pipeline_loaded()
            
            if not was_loaded:
                # Update UI to show model is loading
                self.after(0, lambda: self.progress.stop())
                self.after(0, lambda: self.progress.place_forget())
                self.after(0, lambda: self.status.config(text="Model loaded! Generating..."))
            
            path = generate_image(prompt)
            
            # Load and display image on main thread
            def update_image():
                img = Image.open(path)
                img.thumbnail((512, 512))
                tk_img = ImageTk.PhotoImage(img)
                self.canvas.delete("all")  # Clear previous image
                self.canvas.create_image(0, 0, anchor="nw", image=tk_img)
                self.canvas.image = tk_img  # Keep a reference
                self.status.config(text="Done")
                self.generate_btn.config(state=tk.NORMAL)
            
            self.after(0, update_image)
            
        except Exception as e:
            error_message = str(e)  # Capture the error message
            def update_error():
                self.progress.stop()
                self.progress.place_forget()
                self.status.config(text="Error")
                self.generate_btn.config(state=tk.NORMAL)
                messagebox.showerror("Generation Failed", error_message)
            
            self.after(0, update_error)

if __name__ == "__main__":
    app = DiffusionApp()
    app.mainloop()