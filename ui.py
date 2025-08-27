import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import joblib
from feature_extraction import extract_single_image_features
from pca import apply_pca

class EmotionClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Expression Classifier")
        self.root.geometry("800x600")
        
        # Configure styles
        self.root.configure(bg='#f0f0f0')
        self.button_style = {'bg': '#4CAF50', 'fg': 'white', 'font': ('Arial', 12)}
        self.label_style = {'bg': '#f0f0f0', 'font': ('Arial', 14)}
        self.result_style = {'bg': '#f0f0f0', 'font': ('Arial', 16, 'bold'), 'fg': '#2196F3'}
        
        # Create widgets
        self.create_widgets()
        
        # Load model (show error if not found)
        try:
            self.model = joblib.load('expression_classifier.joblib')
        except FileNotFoundError:
            messagebox.showerror("Error", "Model not found. Please train the model first.")
            self.root.destroy()
    
    def create_widgets(self):
        # Title
        tk.Label(self.root, text="Facial Expression Classifier", 
                font=('Arial', 20, 'bold'), bg='#f0f0f0').pack(pady=20)
        
        # Image display
        self.image_frame = tk.Frame(self.root, bg='white', width=600, height=400)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame, bg='white')
        self.image_label.pack(expand=True, fill='both')
        
        # Select button
        tk.Button(self.root, text="Select Image", command=self.load_image, 
                **self.button_style).pack(pady=10)
        
        # Result display
        self.result_label = tk.Label(self.root, text="", **self.result_style)
        self.result_label.pack(pady=20)
        
        # Instructions
        tk.Label(self.root, text="Select an image containing a face to classify its expression",
                **self.label_style).pack(pady=10)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
            
        try:
            # Display image
            img = Image.open(file_path)
            img.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep reference
            
            # Classify image
            self.classify_image(file_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not process image: {str(e)}")
    
    def classify_image(self, image_path):
        try:
            # Extract features
            features = extract_single_image_features(image_path)
            if features is None:
                self.result_label.config(text="No face detected", fg='red')
                return
            
            # Apply PCA and predict
            pca_scores = apply_pca(features.reshape(1, -1))[:, :4]
            prediction = self.model.predict(pca_scores)[0]
            
            # Display result
            self.result_label.config(text=f"Predicted Expression: {prediction.capitalize()}", fg='green')
            
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}", fg='red')

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionClassifierApp(root)
    root.mainloop()