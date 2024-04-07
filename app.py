import streamlit as st
import pandas as pd
import openpyxl
import csv
from transformers import pipeline

# Load pre-trained language model and tokenizer
question_answering = pipeline("question-answering", model="facebook/bart-base", tokenizer="facebook/bart-base")


def read_data(filename, file_type):
    """Reads data from XLSX or CSV file."""
    if file_type == "xlsx":
        workbook = openpyxl.load_workbook(filename)
        sheet = workbook.active
        data = pd.DataFrame(sheet.values)
    elif file_type == "csv":
        data = pd.read_csv(filename)
    else:
        raise ValueError("Unsupported file type. Please provide XLSX or CSV.")
    return data


def preprocess_data(data):
    """Preprocesses the spreadsheet data for compatibility."""
    # Convert all data to strings
    data = data.astype(str)
    # Remove leading/trailing whitespaces from all cells
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return data


def preprocess_question(question):
    """Preprocesses the user's question."""
    # Lowercase the text
    return question.lower()


def answer_question(data, question):
    """Function to answer user's question based on data and question."""
    # Preprocess data and question
    data = preprocess_data(data)
    question = preprocess_question(question)

    # Use language model for Named Entity Recognition (NER)
    ner_outputs = question_answering(question=question, context=data.to_string())
    named_entities = ner_outputs["answer"]  # Assuming the first answer is the named entity

    # Refine search based on NER and question keywords
    filtered_data = []
    for index, row in data.iterrows():
        if any(word.lower() in str(cell).lower() for word in question.split() for cell in row):
            if named_entities:
                if named_entities in " ".join(map(str, row)):
                    filtered_data.append(row)
            else:
                filtered_data.append(row)

    if not filtered_data:
        return "No answer found in the spreadsheet. Would you like to rephrase your question?"

    # Select a representative data point for initial answer
    answer_data = filtered_data[0]
    initial_answer = " | ".join(map(str, answer_data))

    return initial_answer


def main():
    st.title("Question Answering System for Excel Files")

    # File upload section
    uploaded_file = st.file_uploader("Upload Excel (XLSX) or CSV file", type=["xlsx", "csv"])
    
    # File type selector
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[-1]
    else:
        file_type = None

    # Query input section
    query = st.text_input("Enter your query here")

    if uploaded_file is not None and query:
        try:
            # Read data from uploaded file
            data = read_data(uploaded_file, file_type)
            # Analyze button
            if st.button("Analyze"):
                result = answer_question(data, query)
                st.subheader("Analysis Result")
                st.write("Answer:", result)
        except Exception as e:
            st.error(f"Error reading file or analyzing data: {e}")


if __name__ == "__main__":
    main()
