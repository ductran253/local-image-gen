import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import os
import sys
from io import StringIO

if getattr(sys, 'frozen', False):
    backend_path = os.path.join(sys._MEIPASS, 'backend')
else:
    backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, backend_path)



class InstallationWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local Image Generator - Setup")
        self.geometry("500x300")
        self.resizable(False, False)
        
        # Center the window
        self.center_window()
        
        # Make it look like a splash screen
        self.configure(bg='#2c3e50')
        
        # Title
        title_label = tk.Label(self, text="Local Image Generator", 
                              font=("Arial", 20, "bold"), 
                              fg="white", bg='#2c3e50')
        title_label.pack(pady=30)
        
        # Status label
        self.status_label = tk.Label(self, text="Initializing...", 
                                   font=("Arial", 12), 
                                   fg="white", bg='#2c3e50')
        self.status_label.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self, mode='indeterminate', length=300)
        self.progress.pack(pady=20)
        self.progress.start()
        
        # Details label for package installation
        self.details_label = tk.Label(self, text="", 
                                    font=("Arial", 10), 
                                    fg="#bdc3c7", bg='#2c3e50',
                                    wraplength=450)
        self.details_label.pack(pady=10)
        
        # Start installation in background
        self.after(100, self.start_installation)
    
    def center_window(self):
        """Center the window on the screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
        
    def update_status(self, message, details=""):
        """Update the status message and details"""
        self.status_label.config(text=message)
        self.details_label.config(text=details)
        self.update()
    
    def start_installation(self):
        """Start the installation process in a separate thread"""
        threading.Thread(target=self.run_installation, daemon=True).start()
        
    def run_installation(self):
        """Run the installation process"""
        try:
            self.after(0, lambda: self.update_status("Checking dependencies...", 
                                                   "Verifying required packages"))
            
            # Import and run installation
            from backend.model import ensure_ml_packages
            
            # Custom installation with UI updates
            self.after(0, lambda: self.update_status("Installing packages...", 
                                                   "Downloading and installing ML libraries"))
            
            # Monkey patch print to capture installation messages
            original_print = print
            def custom_print(*args, **kwargs):
                message = ' '.join(str(arg) for arg in args)
                self.after(0, lambda: self.update_status("Installing packages...", message))
                original_print(*args, **kwargs)
            
            # Temporarily replace print
            import builtins
            builtins.print = custom_print
            
            try:
                ensure_ml_packages()
            finally:
                # Restore original print
                builtins.print = original_print
            
            # Installation complete
            self.after(0, lambda: self.update_status("Installation complete!", 
                                                   "Starting application..."))
            
            # Wait a moment then start main app
            self.after(2000, self.start_main_app)
            
        except Exception as e:
            error_msg = f"Installation failed: {str(e)}"
            self.after(0, lambda: self.show_error(error_msg))
    
    def show_error(self, error_msg):
        """Show error message"""
        self.progress.stop()
        self.update_status("Installation Failed", error_msg)
        messagebox.showerror("Installation Error", error_msg)
        self.quit()
    
    def start_main_app(self):
        """Close installation window and start main app"""
        self.destroy()
        app = DiffusionApp()
        app.mainloop()

class DiffusionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local Image Generator")
        self.geometry("800x600")
        self.model_loaded = False
        
        # Center the main window too
        self.center_window()
        
        # UI Layout
        self.prompt_entry = ttk.Entry(self, font=("Arial", 14), width=40)
        self.prompt_entry.place(x=550, y=200)
       
        self.generate_btn = ttk.Button(self, text="Generate", command=self.on_generate)
        self.generate_btn.place(x=650, y=250)
        
        self.canvas = tk.Canvas(self, width=512, height=512, bg="white")
        self.canvas.place(x=10, y=60)
        
        self.status = ttk.Label(self, text="Ready - Model will load on first generation", anchor="w")
        self.status.place(x=10, y=30)
        
        # Progress bar for model loading
        self.progress = ttk.Progressbar(self, mode='indeterminate')
        self.progress.place(x=550, y=300, width=200)
        self.progress.place_forget()  # Hide initially
        
        # Add some helpful labels
        prompt_label = ttk.Label(self, text="Enter your prompt:")
        prompt_label.place(x=550, y=175)
        
        canvas_label = ttk.Label(self, text="Generated Image:")
        canvas_label.place(x=10, y=10)
    
    def center_window(self):
        """Center the window on the screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def on_generate(self):
        prompt = self.prompt_entry.get()
        if not prompt:
            messagebox.showwarning("No Prompt", "Please enter a prompt.")
            return
       
        # Import here since packages are now guaranteed to be installed
        from backend.model import is_pipeline_loaded
        
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
            # Import here since packages are now guaranteed to be installed
            from backend.model import is_pipeline_loaded, generate_image
            
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
    # Start with the installation window instead of main app
    installer = InstallationWindow()
    installer.mainloop()