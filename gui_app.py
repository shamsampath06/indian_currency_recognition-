# gui_app.py
import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
from predict import predict_currency

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        image_label.configure(image=img_tk)
        image_label.image = img_tk

        label, confidence = predict_currency(file_path)
        result_label.config(text=f"Prediction: ₹{label}  ({confidence:.2f}% confidence)")

app = tk.Tk()
app.title("Indian Currency Recognition")
app.geometry("400x500")

title = Label(app, text="Upload Indian Currency Note", font=("Helvetica", 16))
title.pack(pady=10)

upload_btn = tk.Button(app, text="Upload Image", command=upload_image)
upload_btn.pack(pady=10)

image_label = Label(app)
image_label.pack()

result_label = Label(app, text="", font=("Helvetica", 14))
result_label.pack(pady=20)

app.mainloop()
