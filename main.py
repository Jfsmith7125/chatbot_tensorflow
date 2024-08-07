import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
import random

# Load the intents data
def load_intents():
    with open('intents.json') as file:
        return json.load(file)

data = load_intents()

# Data preprocessing
def preprocess_data(data):
    training_sentences = []
    training_labels = []
    labels = []
    responses = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    num_classes = len(labels)
    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    return training_sentences, training_labels, labels, responses, num_classes, lbl_encoder

training_sentences, training_labels, labels, responses, num_classes, lbl_encoder = preprocess_data(data)

# Vectorization
def create_vectorizer(training_sentences):
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=2000, output_mode='int', output_sequence_length=20)
    vectorizer.adapt(training_sentences)
    return vectorizer

vectorizer = create_vectorizer(training_sentences)

# Create the model
def create_model(num_classes):
    model = Sequential()
    model.add(tf.keras.Input(shape=(20,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model(num_classes)

# Train the model
def train_model(model, vectorizer, training_sentences, training_labels):
    model.fit(vectorizer(np.array(training_sentences)), np.array(training_labels), epochs=200)
    model.save('basic_model.keras')
    with open('label_encoder.json', 'w') as f:
        json.dump(lbl_encoder.classes_.tolist(), f)
    with open('vectorizer_config.json', 'w') as f:
        json.dump(vectorizer.get_config(), f)

# Load the model and configurations
def load_model_and_configurations():
    model = tf.keras.models.load_model('basic_model.keras')

    with open('label_encoder.json') as f:
        lbl_classes = json.load(f)
    lbl_encoder = LabelEncoder()
    lbl_encoder.classes_ = np.array(lbl_classes)

    with open('vectorizer_config.json') as f:
        vectorizer_config = json.load(f)
    vectorizer = tf.keras.layers.TextVectorization.from_config(vectorizer_config)

    training_sentences = [pattern for intent in data['intents'] for pattern in intent['patterns']]
    vectorizer.adapt(training_sentences)  # Re-adapt the vectorizer

    return model, lbl_encoder, vectorizer

# Function to get a response from the chatbot
def get_response(user_input, model, lbl_encoder, vectorizer):
    input_seq = vectorizer([user_input])
    predictions = model.predict(input_seq)
    predicted_class = np.argmax(predictions[0])
    label = lbl_encoder.classes_[predicted_class]

    for intent in data['intents']:
        if intent['tag'] == label:
            response = random.choice(intent['responses'])
            return response

    return "Sorry, I didn't understand that."

# Function to retrain the model
def retrain_model():
    global model, lbl_encoder, vectorizer
    print("Retraining model...")
    training_sentences, training_labels, _, _, num_classes, lbl_encoder = preprocess_data(load_intents())
    vectorizer = create_vectorizer(training_sentences)
    model = create_model(num_classes)
    train_model(model, vectorizer, training_sentences, training_labels)
    print("Model retrained successfully!")

# Main interactive loop
def main():
    global model, lbl_encoder, vectorizer
    model, lbl_encoder, vectorizer = load_model_and_configurations()
    print("Hi! I'm your chatbot. Type 'quit' to exit. Type 'add_intent' to add a new intent. Type 'retrain_model' to retrain the model.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'retrain_model':
            retrain_model()
        else:
            response = get_response(user_input, model, lbl_encoder, vectorizer)
            print(f"Bot: {response}")

if __name__ == "__main__":
    main()
