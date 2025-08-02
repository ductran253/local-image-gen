import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from backend.model import generate_image

def run_generation():
    prompt = prompt_entry.get()
    if not prompt.strip():
        status_label.config(text="Please enter a prompt!")
        return
    status_label.config(text="Generating image...")
    generate_button.config(state="disabled")

    def task():
        img = generate_image(prompt)
        img.save("output.png")
        img.thumbnail((512, 512))
        photo = ImageTk.PhotoImage(img)

        def update_ui():
            image_label.config(image=photo)
            image_label.image = photo
            status_label.config(text="Done!")
            generate_button.config(state="normal")

        root.after(0, update_ui)

    threading.Thread(target=task).start()

root = tk.Tk()
root.title("Stable Diffusion UI")

frame = ttk.Frame(root, padding=10)
frame.pack(fill="both", expand=True)

prompt_entry = ttk.Entry(frame, width=60)
prompt_entry.pack(pady=5)
prompt_entry.insert(0, "Enter your prompt here...")

generate_button = ttk.Button(frame, text="Generate", command=run_generation)
generate_button.pack(pady=5)

status_label = ttk.Label(frame, text="")
status_label.pack()

image_label = ttk.Label(frame)
image_label.pack(pady=10)

root.mainloop()