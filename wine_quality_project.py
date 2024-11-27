import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import seaborn as sns
import matplotlib.pyplot as plt

# Data loading functions
def load_red_data():
    return pd.read_csv('data/winequality-red.csv', sep=';')

def load_white_data():
    return pd.read_csv('data/winequality-white.csv', sep=';')

# Preprocessing function
def preprocess_data(data):
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Design with Improvements
def build_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(10, activation='softmax')  # 10 quality levels
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training Function with Learning Rate Scheduling
def train_model(data_loader, model_path):
    X_train, X_test, y_train, y_test = preprocess_data(data_loader())
    model = build_model(X_train.shape[1])

    # Early stopping and learning rate scheduler
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    model.fit(X_train, y_train, validation_split=0.2, epochs=80, 
              batch_size=32, callbacks=[early_stopping, reduce_lr])
    
    # Ensure the 'models' directory exists before saving
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Evaluation Function
def evaluate_model(model_path, data_loader):
    X_train, X_test, y_train, y_test = preprocess_data(data_loader())
    model = load_model(model_path)

    y_pred = model.predict(X_test).argmax(axis=1)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(11), yticklabels=range(11))
    plt.title(f'Confusion Matrix: {model_path}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Main Execution
if __name__ == "__main__":
    print("Training red wine model...")
    train_model(load_red_data, 'models/red_wine_quality_model.h5')
    
    print("Training white wine model...")
    train_model(load_white_data, 'models/white_wine_quality_model.h5')
    
    print("Evaluating red wine model...")
    evaluate_model('models/red_wine_quality_model.h5', load_red_data)
    
    print("Evaluating white wine model...")
    evaluate_model('models/white_wine_quality_model.h5', load_white_data)
