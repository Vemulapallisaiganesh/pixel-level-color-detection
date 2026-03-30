import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from main import process_image
# Desktop Tkinter interface (alternative to the Flask web app).


class PixelFrontend:
    def __init__(self, root):
        # Initialize main window and shared state for image navigation + output preview.
        self.root = root
        self.root.title("Pixel - Pixel-Level Color Detection & Image Recognition")
        self.root.geometry("1250x760")
        self.root.minsize(980, 620)
        self.root.configure(bg="#0f172a")

        self.image_list = []
        self.current_index = 0
        self.current_output = None

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#0f172a")
        self.style.configure("Card.TFrame", background="#111827")
        self.style.configure("Header.TLabel", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 18, "bold"))
        self.style.configure("Sub.TLabel", background="#0f172a", foreground="#cbd5e1", font=("Segoe UI", 10))
        self.style.configure("PanelTitle.TLabel", background="#111827", foreground="#e2e8f0", font=("Segoe UI", 11, "bold"))
        self.style.configure("Dashboard.TLabel", background="#111827", foreground="#f8fafc", font=("Segoe UI", 11))
        self.style.configure("DashboardTitle.TLabel", background="#111827", foreground="#60a5fa", font=("Segoe UI", 16, "bold"))

        self.confidence_var = tk.DoubleVar(value=0.30)
        self.status_var = tk.StringVar(value="Ready. Upload images to begin.")
        self.position_var = tk.StringVar(value="No images loaded")

        self.input_panel_image = None
        self.output_panel_image = None

        self._build_ui()
        self._bind_keys()

    def _build_ui(self):
        # Build two-tab layout: informational dashboard + interactive processor.
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=14, pady=(12, 6))

        ttk.Label(header, text="Pixel", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Pixel-Level Color Detection & Advanced Image Recognition",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=8)

        self.home_tab = ttk.Frame(self.notebook, style="TFrame")
        self.processor_tab = ttk.Frame(self.notebook, style="TFrame")

        self.notebook.add(self.home_tab, text="Home Dashboard")
        self.notebook.add(self.processor_tab, text="Image Processor")

        self._build_home_tab()
        self._build_processor_tab()

    def _build_home_tab(self):
        # Static product overview content shown in the Home tab.
        container = ttk.Frame(self.home_tab, style="TFrame")
        container.pack(fill="both", expand=True, padx=8, pady=8)

        dashboard = ttk.Frame(container, style="Card.TFrame", padding=16)
        dashboard.pack(fill="both", expand=True)

        ttk.Label(dashboard, text="Pixel - Project Overview", style="DashboardTitle.TLabel").pack(anchor="w", pady=(0, 12))

        sections = [
            ("About", "Pixel is an advanced image recognition and scene understanding platform that combines pixel-level color detection with deep learning-based object segmentation using YOLOv8."),
            ("Core Technology", "YOLOv8 Segmentation: State-of-the-art real-time instance and semantic segmentation\nOpenCV: High-performance image processing and color manipulation\nUltralytics: Modern deep learning framework with pre-trained models"),
            ("Key Features", "• Pixel-level color detection and mapping\n• Real-time object segmentation\n• Customizable confidence thresholds\n• Batch image processing\n• COCO dataset training support\n• Multi-format export (JPG, PNG, BMP, WebP)"),
            ("Capabilities", "• Detect and segment objects in images\n• Apply color regions to detected objects\n• Label objects with class names and colors\n• Process full image batches\n• Save annotated outputs\n• Train on custom datasets (COCO)"),
            ("Workflow", "1. Upload images → 2. Adjust confidence threshold → 3. Process images → 4. Review results → 5. Save outputs\n\nNavigate to 'Image Processor' tab to start working with images."),
        ]

        for title, content in sections:
            section = ttk.Frame(dashboard, style="Card.TFrame", padding=12)
            section.pack(fill="x", pady=8)

            ttk.Label(section, text=title, style="DashboardTitle.TLabel").pack(anchor="w", pady=(0, 6))
            ttk.Label(
                section,
                text=content,
                style="Dashboard.TLabel",
                wraplength=600,
                justify="left"
            ).pack(anchor="w", fill="x")

    def _build_processor_tab(self):
        # Processing controls, side-by-side input/output viewers, and footer status.
        controls = ttk.Frame(self.processor_tab, style="Card.TFrame", padding=10)
        controls.pack(fill="x", padx=8, pady=8)

        tk.Button(
            controls,
            text="Upload Images",
            command=self.upload_images,
            bg="#22c55e",
            fg="white",
            activebackground="#16a34a",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=7,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        ).grid(row=0, column=0, padx=(0, 8), pady=4)

        tk.Button(
            controls,
            text="Process Current",
            command=self.process_current,
            bg="#0ea5e9",
            fg="white",
            activebackground="#0284c7",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=7,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        ).grid(row=0, column=1, padx=8, pady=4)

        tk.Button(
            controls,
            text="Save Output",
            command=self.save_output,
            bg="#f59e0b",
            fg="white",
            activebackground="#d97706",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=7,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        ).grid(row=0, column=2, padx=8, pady=4)

        tk.Button(
            controls,
            text="Previous",
            command=self.prev_image,
            bg="#334155",
            fg="white",
            activebackground="#1e293b",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=7,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        ).grid(row=0, column=3, padx=8, pady=4)

        tk.Button(
            controls,
            text="Next",
            command=self.next_image,
            bg="#334155",
            fg="white",
            activebackground="#1e293b",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=7,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        ).grid(row=0, column=4, padx=8, pady=4)

        ttk.Label(controls, text="Confidence", style="PanelTitle.TLabel").grid(row=0, column=5, padx=(16, 6))

        conf_scale = tk.Scale(
            controls,
            from_=0.10,
            to=0.95,
            resolution=0.05,
            orient="horizontal",
            variable=self.confidence_var,
            bg="#111827",
            fg="#e2e8f0",
            troughcolor="#1e293b",
            highlightthickness=0,
            length=180,
            font=("Segoe UI", 9),
        )
        conf_scale.grid(row=0, column=6, padx=(4, 0), sticky="w")

        viewer = ttk.Frame(self.processor_tab)
        viewer.pack(fill="both", expand=True, padx=8, pady=8)

        left_card = ttk.Frame(viewer, style="Card.TFrame", padding=8)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 6))

        right_card = ttk.Frame(viewer, style="Card.TFrame", padding=8)
        right_card.pack(side="left", fill="both", expand=True, padx=(6, 0))

        ttk.Label(left_card, text="Input", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 6))
        self.input_panel = tk.Label(left_card, bg="#0b1220")
        self.input_panel.pack(fill="both", expand=True)

        ttk.Label(right_card, text="Output", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 6))
        self.output_panel = tk.Label(right_card, bg="#0b1220")
        self.output_panel.pack(fill="both", expand=True)

        footer = ttk.Frame(self.processor_tab, style="Card.TFrame", padding=8)
        footer.pack(fill="x", padx=8, pady=(0, 8))

        tk.Label(
            footer,
            textvariable=self.position_var,
            bg="#111827",
            fg="#f8fafc",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")

        tk.Label(
            footer,
            textvariable=self.status_var,
            bg="#111827",
            fg="#cbd5e1",
            font=("Segoe UI", 9),
        ).pack(side="right")

    def _bind_keys(self):
        # Keyboard shortcuts for quick image browsing.
        self.root.bind("<Left>", lambda _event: self.prev_image())
        self.root.bind("<Right>", lambda _event: self.next_image())

    def upload_images(self):
        # Load selected images into memory list and reset processor state.
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp *.webp")]
        )

        if not file_paths:
            return

        self.image_list = list(file_paths)
        self.current_index = 0
        self.current_output = None
        self.status_var.set("Images loaded. Click Process Current to run segmentation.")

        self.show_current_input()
        self.clear_output_panel()
        self._update_position_text()

    def _fit_preview(self, bgr_image, max_w=560, max_h=430):
        # Resize image proportionally to fit panel bounds without upscaling.
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        w, h = pil_image.size
        ratio = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        new_size = (int(w * ratio), int(h * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_image)

    def show_current_input(self):
        # Render currently selected source image in the input panel.
        if not self.image_list:
            return

        path = self.image_list[self.current_index]
        image = cv2.imread(path)

        if image is None:
            self.status_var.set("Failed to load selected image.")
            return

        self.input_panel_image = self._fit_preview(image)
        self.input_panel.config(image=self.input_panel_image)

    def process_current(self):
        # Run segmentation on the active image and refresh output preview.
        if not self.image_list:
            messagebox.showinfo("No Image", "Please upload at least one image first.")
            return

        path = self.image_list[self.current_index]
        conf = float(self.confidence_var.get())

        filename = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join("output", f"{filename}_result.jpg")

        try:
            self.status_var.set("Processing...")
            self.root.update_idletasks()

            output = process_image(path, output_path=output_path, conf_threshold=conf)
            self.current_output = output

            self.output_panel_image = self._fit_preview(output)
            self.output_panel.config(image=self.output_panel_image)

            self.status_var.set(f"Processed at confidence {conf:.2f}. Saved to output folder.")
        except Exception as err:
            messagebox.showerror("Processing Error", str(err))
            self.status_var.set("Processing failed.")

    def save_output(self):
        # Persist the latest processed frame to a user-selected location.
        if self.current_output is None:
            messagebox.showinfo("No Output", "Process an image before saving output.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All Files", "*.*")],
            title="Save Processed Output",
        )

        if not save_path:
            return

        try:
            cv2.imwrite(save_path, self.current_output)
            self.status_var.set(f"Saved output to {save_path}")
        except Exception as err:
            messagebox.showerror("Save Error", str(err))
            self.status_var.set("Failed to save output.")

    def _update_position_text(self):
        # Keep footer text synced with image index.
        if not self.image_list:
            self.position_var.set("No images loaded")
            return
        self.position_var.set(f"Image {self.current_index + 1} of {len(self.image_list)}")

    def clear_output_panel(self):
        # Remove output preview when switching images.
        self.output_panel.config(image="")
        self.output_panel_image = None

    def next_image(self):
        # Move to next image and reset current output.
        if not self.image_list:
            return
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.current_output = None
            self.show_current_input()
            self.clear_output_panel()
            self._update_position_text()
            self.status_var.set("Moved to next image.")

    def prev_image(self):
        # Move to previous image and reset current output.
        if not self.image_list:
            return
        if self.current_index > 0:
            self.current_index -= 1
            self.current_output = None
            self.show_current_input()
            self.clear_output_panel()
            self._update_position_text()
            self.status_var.set("Moved to previous image.")


def main():
    # Entry point for launching the desktop application.
    root = tk.Tk()
    PixelFrontend(root)
    root.mainloop()


if __name__ == "__main__":
    main()