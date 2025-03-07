# **CIFAR-10 Image Classification with Custom F-Beta Metric**

## **📌 Project Overview**
This project trains a **deep learning model** for **image classification on CIFAR-10** using **TensorFlow/Keras**. The key novelty in this project is the implementation of a **custom F-Beta score as an evaluation metric**, which balances precision and recall during training.

### **🚀 Key Features**
✅ **Preprocessing CIFAR-10 dataset (Normalization & One-Hot Encoding)**  
✅ **Building a deep neural network (DNN) classifier**  
✅ **Using ReLU activation with multiple Dense layers**  
✅ **Custom F-Beta metric for model evaluation**  
✅ **Training with Stochastic Gradient Descent (SGD)**  
✅ **Visualizing loss and F-Beta metric during training**  

---

## **📌 Dataset: CIFAR-10**
The **CIFAR-10 dataset** consists of **60,000 images** categorized into **10 classes**, including:
- **Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck**
- Each image is **32x32 pixels with 3 color channels**.

### **📌 Data Preprocessing**
The dataset is loaded and normalized for training:
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encoding of labels
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
```

---

## **📌 Model Architecture**
A **fully connected neural network** is designed with **three hidden layers**, each containing **50 neurons** with ReLU activation.
```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
```

---

## **📌 Custom F-Beta Metric**
A custom **F-Beta metric** is implemented to evaluate model performance, prioritizing **both precision and recall**.
```python
def F_beta(y_true, y_pred, beta=2):
    y_pred = tf.clip_by_value(y_pred, 0, 1)
    epsilon = tf.keras.backend.epsilon()
    true_positive = tf.reduce_sum(tf.math.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    false_positive = tf.reduce_sum(tf.math.round(tf.clip_by_value(y_pred - y_true, 0, 1)))
    false_negative = tf.reduce_sum(tf.math.round(tf.clip_by_value(y_true - y_pred, 0, 1)))
    
    P = true_positive / (true_positive + false_positive + epsilon)
    R = true_positive / (true_positive + false_negative + epsilon)
    
    return tf.reduce_mean(((1 + beta**2) * P * R) / ((beta**2 * P) + R + epsilon))
```

---

## **📌 Model Compilation & Training**
The model is compiled using **SGD optimizer** and **categorical crossentropy loss**, with the custom **F-Beta metric** as an additional performance metric.
```python
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=[F_beta])

history = model.fit(x_train, y_train_hot, epochs=50, validation_split=0.2)
```

---

## **📌 Training Performance**
### **📊 Key Observations**
✅ **F-Beta score steadily increases across epochs, indicating improved model performance.**  
✅ **Loss decreases consistently, demonstrating effective learning.**  

### **📌 Visualization of Training Progress**
```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['F_beta'], label='Training F-Beta')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss & F-Beta Score')
plt.title('Training Performance')
plt.show()
```

---

## **📌 Installation & Setup**
### **📌 Prerequisites**
- **Python 3.x**
- **Jupyter Notebook**
- **TensorFlow, NumPy, Matplotlib**

### **📌 Install Required Libraries**
```bash
pip install tensorflow numpy matplotlib
```

---

## **📌 Running the Notebook**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YourGitHubUsername/cifar10-custom-metrics.git
cd cifar10-custom-metrics
```

### **2️⃣ Launch Jupyter Notebook**
```bash
jupyter notebook
```

### **3️⃣ Run the Notebook**
Open `cifar10_fbeta.ipynb` and execute all cells.

---

## **📌 Conclusion**
This project demonstrates a **deep learning approach to CIFAR-10 image classification** using **custom evaluation metrics**. The implementation of an **F-Beta metric** helps balance **precision and recall**, providing a **more informative assessment** of model performance.

---

## **📌 License**
This project is licensed under the **MIT License**.

