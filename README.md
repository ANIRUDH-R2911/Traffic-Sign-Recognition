# Traffic-Sign-Recognition

An end-to-end deep learning solution for classifying German traffic signs using a trained CNN model and a user-friendly Tkinter GUI. Built using TensorFlow and trained on the GTSRB dataset with over 98% accuracy on real-world test images.


## Tech Stack
- **Language**: Python 3.7+
- **Deep Learning Framework**: TensorFlow 2.x, Keras
- **GUI Toolkit**: Tkinter
- **Data Processing**: NumPy, Pandas, PIL
- **Model Visualization**: Matplotlib
- **Training Environment**: Google Colab (with GPU support)
- **Dataset**: [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)


## Features
- Trained CNN model with **98.6% external test accuracy**
- Real-time traffic sign classification via a **Tkinter-based desktop GUI**
- Image preprocessing and data augmentation during training
- Model saved in `.keras` format for future use or deployment
- Displays both class index and human-readable traffic sign label
- Compatible with local image uploads for testing


## Architecture
```plaintext
                  ┌─────────────────────────────┐
                  │     Input Image (30x30)     │
                  └──────────────┬──────────────┘
                                 │
            ┌────────────────────▼────────────────────┐
            │   Convolutional Neural Network (CNN)    │
            │  - Conv2D layers + BatchNormalization   │
            │  - MaxPooling2D + Dropout               │
            │  - Dense layer (512) + Softmax output   │
            └────────────────────┬────────────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │   Predicted Label   │
                      └──────────┬──────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │   Mapped to Class Name (dict) │
                 └───────────────┬───────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │  Displayed in GUI   │
                      └─────────────────────┘
```

## Example GUI

Here’s a preview of the traffic sign recognition GUI built using Tkinter. Users can upload an image of a traffic sign, and the model will classify it in real-time with high accuracy.

<p align="center">
  <img src="gui_demo.png" alt="Traffic Sign GUI" width="600"/>
</p>


## Training Summary

- **Dataset**: [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Epochs Trained**: 30
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Data Augmentation**:
  - Rotation, Zoom, Width/Height Shift, Shear
- **Best Validation Accuracy**: **99.95%**
- **External Test Accuracy**: **98.60%**


## Future Enhancements

| Area               | Enhancement Idea                                                  |
|--------------------|----------------------------------------------------------------   |
| GUI UX             | Add **drag-and-drop support** and **real-time webcam prediction** |
| Deployment         | Convert GUI to a **web app using Streamlit or Gradio**            |
| Mobile Inference   | Export the model to **TensorFlow Lite** for Android/iOS apps      |
| Multi-language     | Add **multi-language support** for international traffic signs    |

