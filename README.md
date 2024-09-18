# Ai_chatbot
This repository contains code for an AI-driven chatbot that recommends Ayurvedic herbal plants based on user queries. Using BERT for natural language understanding, the system processes inputs and suggests relevant herbs or remedies. It includes training on a custom dataset, real-time prediction, and saving/loading models for efficient deployment.

Features
Natural Language Processing: The chatbot leverages a pre-trained BERT model to understand and classify user queries.
Real-time Herbal Recommendations: Based on the user’s query, the model predicts and recommends a specific Ayurvedic herbal plant or remedy.
Custom Dataset: The model is trained on a dataset of user queries and their corresponding Ayurvedic herbal plant categories, allowing it to make accurate predictions.
Train and Test Split: The dataset is split into training and testing sets for model evaluation, ensuring reliable performance.
BERT Tokenization: User inputs are tokenized and pre-processed using the BERT tokenizer, making them suitable for model inference.
Model Fine-Tuning: The BERT model is fine-tuned for sequence classification with multi-class labels representing different herbal plants or remedies.
Cross-Entropy Loss and AdamW Optimizer: The model is trained using Cross-Entropy Loss and optimized with AdamW, allowing efficient and accurate learning.
Model and Tokenizer Saving: After training, the model and tokenizer are saved for future use in real-time prediction without retraining.
Prediction Functionality: A predict function allows real-time prediction of herbal plants based on user input, making the chatbot highly interactive.
How It Works
User Query: The user inputs a natural language query related to herbal plants or remedies.
Tokenization: The query is processed by the BERT tokenizer to generate token IDs and attention masks.
BERT Model Inference: The tokenized query is passed through the fine-tuned BERT model, which classifies the query into a specific herbal plant category.
Prediction: The chatbot returns a relevant herbal plant recommendation based on the classified category.
Real-Time Interaction: Users receive immediate feedback and can continuously interact with the chatbot for more recommendations.
How to Use
Data Preparation: Ensure you have a dataset (final_data.csv) containing user queries and corresponding Ayurvedic herbal plant IDs.
Model Training: The BERT model is fine-tuned on the dataset using PyTorch. You can train the model by running the training script provided in the repository.
Real-Time Prediction: After training, use the chatbot to interact with users and provide herbal plant recommendations.
