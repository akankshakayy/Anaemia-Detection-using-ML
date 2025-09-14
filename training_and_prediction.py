def create_cnn_model(input_shape=(128, 128, 3)):
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Second convolutional block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Third convolutional block
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Fourth convolutional block
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    
    return model

# Create the model
model = create_cnn_model()
model.summary()

# Step 7: Compile the Model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Step 8: Define Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Model checkpoint to save the best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    '/content/drive/MyDrive/Anaemia/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Step 9: Train the Model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Step 10: Evaluate the Model
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training & validation accuracy
    axes[0, 0].plot(history.history['accuracy'])
    axes[0, 0].plot(history.history['val_accuracy'])
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    axes[0, 1].plot(history.history['loss'])
    axes[0, 1].plot(history.history['val_loss'])
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend(['Train', 'Validation'], loc='upper left')
    
    # Plot precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'])
        axes[1, 0].plot(history.history['val_precision'])
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend(['Train', 'Validation'], loc='upper left')
    
    # Plot recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'])
        axes[1, 1].plot(history.history['val_recall'])
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(history)

# Evaluate on test set
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")

# Step 11: Make Predictions and Analyze Results
# Make predictions
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Anaemic', 'Anaemic']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Anaemic', 'Anaemic'], 
            yticklabels=['Non-Anaemic', 'Anaemic'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Display some test images with predictions
plt.figure(figsize=(15, 10))
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(X_test[i])
    true_label = 'Anaemic' if y_test[i] == 1 else 'Non-Anaemic'
    pred_label = 'Anaemic' if y_pred[i] == 1 else 'Non-Anaemic'
    confidence = y_pred_proba[i][0] if y_pred[i] == 1 else 1 - y_pred_proba[i][0]
    color = 'green' if y_test[i] == y_pred[i] else 'red'
    plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Step 12: Save the Model and Results
# Save the final model
model.save('/content/drive/MyDrive/Anaemia/anaemia_detection_model.h5')
print("Model saved as '/content/drive/MyDrive/Anaemia/anaemia_detection_model.h5'")

# Save training history to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('/content/drive/MyDrive/Anaemia/training_history.csv', index=False)
print("Training history saved as '/content/drive/MyDrive/Anaemia/training_history.csv'")

# Save test results
results = {
    'test_accuracy': test_accuracy,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': 2 * (test_precision * test_recall) / (test_precision + test_recall)
}

results_df = pd.DataFrame([results])
results_df.to_csv('/content/drive/MyDrive/Anaemia/test_results.csv', index=False)
print("Test results saved as '/content/drive/MyDrive/Anaemia/test_results.csv'")

print("\nTraining completed successfully!")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")