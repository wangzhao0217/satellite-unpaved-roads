---
title: "Assessment of Unpaved Road Network Using Satellite Imagery and Machine Learning"

author: 
  - name: Zhao Wang
    affiliation: Leeds Institute for Transport Studies, University of Leeds, UK

format:
  gfm: default

number-sections: true
execute: 
  echo: false
  cache: false
editor: 
  markdown: 
    wrap: sentence
---

```bash
conda create -n ITS python=3.10.14
conda activate ITS
pip install unpaved-road-condition-analysis
# or
pip install dist/unpaved_road_condition_analysis-0.0.3-py3-none-any.whl
```

```{python}
#| eval: false

# Import functions from unpaved_road_condition_analysis package
from unpaved_road_condition_analysis import (
    model_train,
    evaluate_model_performance,
    load_image,
    calculate_hog_features,
    calculate_lbp_features,
    calculate_color_histogram,
    calculate_contour_properties,
    calculate_fourier_transform,
    standardize_images,
    Autokeras_model,
    Color_Moments_GLCM_Complex,
    Color_Moments_GLCM_IMG,
    Color_Moments_GLCM,
    process_road_condition_data,
    process_img_to_PCA,
    multimodal_prediction,
    Autokeras_best_model,
    ak_transfer_learning,
    ak_load_data
)

# Import other libraries
import os
import pickle
import pandas as pd
import numpy as np
import autokeras as ak
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Additional imports
from pycaret.classification import load_model, predict_model
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    roc_auc_score, 
    cohen_kappa_score, 
    matthews_corrcoef, 
    mean_squared_error
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
```

```{python}
def load_and_predict(model_path, data_path, PCA = False, Mask = None):
    model = load_model(model_path)
    if PCA is not True:
        data = pd.read_csv(data_path)
    else:
        data = pd.DataFrame(data_path)
    if Mask is not None:
        data = data.iloc[Mask]
    predictions = predict_model(model, data=data)

    if PCA is not True:
        data.to_csv(f'Output_IA/{os.path.basename(data_path)}.csv', index=False)
    else:
        data.to_csv(f'Output_IA/{os.path.basename(model_path)}.csv', index=False)
    return predictions, model, data

def calculate_metrics(predictions):
    y_true = predictions['Label']
    # Assuming 'prediction_score' contains prediction scores/probabilities for each class
    y_pred_score = predictions.get('prediction_score', predictions['prediction_label'])
    y_pred_label = predictions['prediction_label']  # Assuming this is the predicted class label

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred_label)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_label, average='weighted')

    # RMSE - Using scores if available, else use label as a proxy (not typical)
    if 'prediction_score' in predictions:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_score))
    else:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_label))

    # ROC-AUC for multi-class
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    y_pred_binarized = label_binarize(y_pred_label, classes=np.unique(y_pred_label))
    roc_auc = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='weighted')
    
    # Kappa
    kappa = cohen_kappa_score(y_true, y_pred_label)

    # MCC
    mcc = matthews_corrcoef(y_true, y_pred_label)

    return {
        'Accuracy': accuracy,
        'AUC': roc_auc,
        'RMSE': rmse,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1,
        'Kappa': kappa,
        'MCC': mcc
    }



def plot_confusion_matrix(y_true, y_pred_label, model_name=None, Name=None, cbar=True):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred_label)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Set up the matplotlib figure (with a specified figure size)
    plt.figure(figsize=(10, 7))
    
    # Create a heatmap that displays both normalized and actual counts
    sns.heatmap(cm_normalized, annot=np.array([[f"{int(cm[i, j])}\n({cm_normalized[i, j]:.2f})" for j in range(len(cm))] for i in range(len(cm))]), 
                fmt='', cmap='Blues', annot_kws={"size": 14}, cbar=cbar,
                xticklabels=["Bad", "Poor", "Fair", "Good"], yticklabels=["Bad", "Poor", "Fair", "Good"])
    
    # Set font sizes for tick labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Set labels and title with specified font sizes
    plt.xlabel('Predicted', fontsize=24)
    plt.ylabel('True', fontsize=24)
    plt.title(f'Confusion Matrix: {Name}', fontsize=24)
    
    # Save the figure to a file if model_name is provided
    if model_name:
        plt.savefig(f'D:/OneDrive - University of Leeds/ESA plot/{model_name}_confusion_matrix.png')
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot
    plt.show()

def performance_check(model_path, data_path, PCA, Mask, Name=None, cbar=True):
    # Placeholder for your actual function to load and predict
    predictions, model, data = load_and_predict(model_path, data_path, PCA, Mask)
    
    # Placeholder for your actual function to calculate metrics
    metrics = calculate_metrics(predictions)
    metrics = {k: round(v * 100, 3) for k, v in metrics.items()}

    print(f'Accuracy: {round(metrics["Accuracy"], 3) * 100}%')
    print(f'ROC-AUC Score: {round(metrics["AUC"], 3) * 100}%')
    print(f'RMSE: {round(metrics["RMSE"], 3)}')
    print(f'Recall: {round(metrics["Recall"], 3) * 100}%')
    print(f'Precision: {round(metrics["Precision"], 3) * 100}%')
    print(f'F1 Score: {round(metrics["F1 Score"], 3) * 100}%')
    print(f'Kappa: {round(metrics["Kappa"], 3)}')
    print(f'MCC: {round(metrics["MCC"], 3)}')

    # Plot confusion matrix
    plot_confusion_matrix(predictions['Label'], predictions['prediction_label'], model_name=os.path.basename(model_path), Name=Name, cbar=cbar) 

```


VGG16 Model Training

```{python}
# Define paths
base_dir = './data/SIP_data/'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Set up Image Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Reserve 20% of images for validation
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Specify this is training data
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Specify this is validation data
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical'
)

# Load VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers except the last 4 layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Create custom layers atop the base model
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dense(4, activation='softmax')(x)  # 4 classes

# Combine the model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with Early Stopping
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping]
)

model.summary()

model.save('model/VGG16_URSC_2.h5') 

# Function to visualize original and augmented images
def visualize_augmentation(datagen, original_img):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    ax[0].imshow(original_img.astype('uint8'))
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # Augmented image
    augmented_img = next(datagen.flow(np.expand_dims(original_img, axis=0)))[0]
    ax[1].imshow(augmented_img.astype('uint8'))
    ax[1].set_title("Augmented Image")
    ax[1].axis('off')
    
    plt.show()

# Example usage of visualize_augmentation
# Load a sample image from the training directory
sample_img_path = os.path.join(train_dir, os.listdir(train_dir)[0], os.listdir(os.path.join(train_dir, os.listdir(train_dir)[0]))[0])
sample_img = plt.imread(sample_img_path)

visualize_augmentation(train_datagen, sample_img)

```
```{python}
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

model = load_model("model/VGG16_URSC_2.h5")
class_labels = ['Bad', 'Fair', 'Good', 'Poor']
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# test_generator.reset()
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # No need to shuffle the data during evaluation
)

steps = (test_generator.samples // test_generator.batch_size) + (test_generator.samples % test_generator.batch_size > 0)

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

predictions =  model.predict(test_generator, steps=steps)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(classification_report(true_classes, predicted_classes, target_names=class_labels))
conf_matrix = confusion_matrix(true_classes, predicted_classes)

accuracy = accuracy_score(true_classes, predicted_classes)

predicted_onehot = tf.keras.utils.to_categorical(predicted_classes, num_classes=len(class_labels))
true_onehot = tf.keras.utils.to_categorical(true_classes, num_classes=len(class_labels))

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
auc_score = roc_auc_score(true_onehot, predictions, multi_class='ovr')
kappa = cohen_kappa_score(true_classes, predicted_classes)
mcc = matthews_corrcoef(true_classes, predicted_classes)
precision, recall, f1, _ = precision_recall_fscore_support(true_classes, predicted_classes, average='weighted')

print("Accuracy:", round(accuracy*100,1), "%")
print("AUC:", round(auc_score*100,1), "%")
print("Recall:", round(recall*100,1), "%")
print("Precision:", round(precision*100,1), "%")
print("F1 Score:", round(f1*100,1), "%")
print("Cohen's Kappa:", round(kappa*100,1), "%")
print("Matthews Correlation Coefficient:", round(mcc*100,1), "%")

print("Classification Report:\n", report)


plot_confusion_matrix(true_classes, predicted_classes, model_name=None, Name="VGG16", cbar=True)
```


# prepare color moments and GLCM features
```{python}
data_path_train = "./data/SIP_data/train/"
data_path_test = "./data/SIP_data/test/"

Color_Moments_GLCM(datapath=data_path_train, datatype='train', file_name = 'SIP_High_Res')
Color_Moments_GLCM(datapath=data_path_test, datatype='test', file_name = 'SIP_High_Res')
```
# train the model
```{python}
model_train(df_train = pd.read_csv(f"Output/SVM-hsv-Color_Moments_GLCM_train_SIP_High_Res.csv"), df_test = pd.read_csv(f"Output/SVM-hsv-Color_Moments_GLCM_test_SIP_High_Res.csv"), file_name = 'SIP_High_Res', fix_imbalance=True, use_gpu=True)
```
# Evaluation of the model performance
```{python}
model_path_SIP_High_Res = 'model/best_model_structured_SIP_High_Res'

performance_check(model_path_SIP_High_Res, "Output/SVM-hsv-Color_Moments_GLCM_test_SIP_High_Res.csv", PCA = False, Mask = None, Name = "Base_Model_SIP_High_Res", cbar = False)
```

# Train Model for Binary Classification
```{python}
model_train(df_train = pd.read_csv(f"Output/SVM-hsv-Color_Moments_GLCM_train_SIP_High_Res_Binary.csv"), df_test = pd.read_csv(f"Output/SVM-hsv-Color_Moments_GLCM_test_SIP_High_Res_Binary.csv"), file_name = 'SIP_High_Res_Binary', fix_imbalance=True, use_gpu=True)
```

# Evaluation of the model performance
```{python}
binary_model_path = 'model/best_model_structured_SIP_High_Res_Binary'

performance_check(binary_model_path, "Output/SVM-hsv-Color_Moments_GLCM_test_SIP_High_Res_Binary.csv", PCA = False, Mask = None, Name = "Blended Model: Binary Classification", cbar = True)
```
