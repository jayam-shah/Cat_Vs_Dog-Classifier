# 🐶🐱 Dog vs Cat Classifier using Deep Learning

This project is a deep learning-based image classifier that distinguishes between images of dogs and cats. It was developed and trained using Google Colab with TensorFlow and Keras, using a Convolutional Neural Network (CNN) and the popular Dogs vs Cats dataset.

---

## 📁 Dataset

The model uses the **Dogs vs Cats** dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/salader/dogs-vs-cats 
  
You'll need to upload the extracted dataset manually into your Google Drive or Colab environment.

---

## 🔧 Technologies Used

- Python 3
- TensorFlow
- Keras
- Matplotlib
- NumPy
- Google Colab

---

## 🏗️ Model Architecture

The classifier uses a simple yet effective CNN architecture:
- Convolution Layers with ReLU
- Max Pooling Layers
- Flatten Layer
- Fully Connected (Dense) Layers
- Final Sigmoid Activation for Binary Classification

---

## 📈 Training Process

- **Image Augmentation** was applied using `ImageDataGenerator` for better generalization.
- Model was compiled with:
  - Loss: `binary_crossentropy`
  - Optimizer: `adam`
  - Metrics: `accuracy`
- Training was done for `5 epochs` using training and validation generators.

---

## 📊 Output

The notebook displays:
- Training vs Validation Accuracy and Loss graph
- Model evaluation on test data
- Predictions on sample images

---

## 🚀 How to Run

1. Open the `cat_vs_dog_classifier.ipynb` notebook in Google Colab.
2. Mount your Google Drive to access the dataset.
3. Make sure your dataset folder structure is as follows:

```

dogs-vs-cats/
│
└───train/
├── cat.0.jpg
├── dog.0.jpg
└── ...

````

4. Run each cell step-by-step to:
   - Load and preprocess data
   - Build the CNN model
   - Train the model
   - Evaluate and visualize results

---

## 💾 Model Saving

The trained model is saved as:
```python
model.save("cat_dog_model.h5")
````

You can later load it using:

```python
model = tf.keras.models.load_model("cat_dog_model.h5")
```

---

## 📌 Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

* **Name:**  Jayam D Shah
* **Platform:** Google Colab

---

## 📃 License

This project is intended for educational purposes. 

---

### 📦 requirements.txt

tensorflow \
keras \
matplotlib \
numpy \
pandas 
