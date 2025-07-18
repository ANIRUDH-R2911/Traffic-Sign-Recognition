import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('TSR.keras') 

classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles > 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing for vehicles > 3.5 tons'
}


top = tk.Tk()
top.geometry('1000x750')
top.title('Traffic Sign Recognition')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)

    pred_probs = model.predict(image)
    pred_class = np.argmax(pred_probs)
    sign = classes[pred_class]
    print(f"Predicted: {sign}")
    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_btn = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_btn.configure(background='#192733', foreground='white', font=('arial', 10, 'bold'))
    classify_btn.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        resized = uploaded.resize((400, 400)) 
        im = ImageTk.PhotoImage(resized)
        sign_image.configure(image=im, width=400, height=400)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print("Error:", e)



upload_btn = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload_btn.configure(background='#121143', foreground='white', font=('arial', 10, 'bold'))
upload_btn.pack(side=tk.BOTTOM, pady=30)

sign_image.pack(side=tk.BOTTOM, expand=True, pady=20)
label.pack(side=tk.BOTTOM, expand=True)

heading = Label(top, text="Traffic sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#672555')
heading.pack()

top.mainloop()
