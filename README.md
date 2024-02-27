Project Name: Hue Chip - RAG LLM
Description
Hue Chip - RAG LLM is a general-purpose educational chatbot leveraging the power of Retrival Augmented Generation (RAG)  and Large Language Models (LLM). This chatbot is designed to provide a versatile and interactive educational experience for users across various domains.

Key Features
RAG : The chatbot utilizes RAG  to understand and categorize user queries, allowing for a more context-aware and efficient interaction.

Large Language Models (LLM): Powered by advanced LLM, EduBot is equipped to comprehend complex language structures and provide insightful responses tailored to the user's educational needs.

General Purpose Knowledge: EduBot is designed to cover a wide range of educational topics, making it a valuable resource for students, educators, and anyone seeking information.

Interactive Learning: The chatbot engages users in interactive learning experiences, providing explanations, answering questions, and offering additional resources to enhance understanding.

main.py: This file contains the core code for Rapid Automatic Genre (RAG) classification, without including a history feature. The RAG classification is employed to analyze and categorize text data.

need_to_edit: This directory includes the code base for RAG classification but without the history feature. Additionally, it features functionality for converting text to audio (text_to_audio) and audio to text (audio_to_text).

PS: the code along with history feature is saved in a file updated_main in archieve section

synthesise and transcribe: These modules contain code for speech-to-text and text-to-speech functionalities, utilizing live audio input. The synthesis and transcription processes enhance the overall versatility of the library.

synthesis_test and transcribe_test: These modules are dedicated to speech-to-text and text-to-speech functionalities, specifically designed for pre-recorded audio input. These tests provide a way to validate the accuracy and reliability of the library under controlled conditions.

Usage
To make use of the LocalRag with Ollama library, follow these steps:

Clone the repository: git clone https://github.com/Navin77773/rag

Navigate to the project directory: cd local-rag-ollama

Execute the desired functionality from the available modules.

Dependencies:
Ensure that you have the necessary dependencies installed. You can install them using the following:

bash
'''
pip install -r requirements.txt
'''