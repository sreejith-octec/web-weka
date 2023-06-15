import pickle
import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np


# Function to load and return the selected pickle file
def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def com_accuracy_form(model_file):

    file_name = model_file

    # Remove the extension ".pkl"
    file_name_without_extension = file_name.replace(".pkl", "")

    # Split the string by "_" and retrieve the last element
    last_element = file_name_without_extension.split("_")[-1]

    try:
        float_digits = float(last_element)
        print("Float digits:", float_digits)
    except ValueError:
        print("No float digits found.")
    return float_digits

def comparison():

    global accuracy_score_model1, accuracy_score_model2
    # Get a list of pickle files in the "models" folder
    cm_pickle_files = [file for file in os.listdir('models') if file.endswith('.pkl')]

    # Sidebar setup
    st.sidebar.title('Select First Pickle File')
    cm_selected_file_1 = st.sidebar.selectbox('Pick a pickle file', cm_pickle_files, key="cm_file_1_select")
    if cm_selected_file_1:
        accuracy_score_model1 = com_accuracy_form(cm_selected_file_1)

    # Remove the selected file from the list for the second select box
    cm_pickle_files_2 = cm_pickle_files.copy()
    cm_pickle_files_2.remove(cm_selected_file_1)

    # Sidebar setup
    st.sidebar.title('Select Second Pickle File')
    cm_selected_file_2 = st.sidebar.selectbox('Pick a pickle file', cm_pickle_files_2, key="cm_file_2_select")
    if cm_selected_file_2:
        accuracy_score_model2=com_accuracy_form(cm_selected_file_2)
    
    st.subheader("Selected Models are : ")

    # Check if a file is selected
    if cm_selected_file_1:
        file_path = os.path.join('models', cm_selected_file_1)
        loaded_model = load_pickle_file(file_path)
        st.write(loaded_model)
    else:
        st.sidebar.info('No pickle file selected.')

    if cm_selected_file_2:
        file_path = os.path.join('models', cm_selected_file_2)
        loaded_model = load_pickle_file(file_path)
        st.write(loaded_model)
    else:
        st.sidebar.info('No pickle file selected.')


def com_graph():
    st.subheader("Comparison plot:")
    # Create a list of model names
    model_names = ['Model 1', 'Model 2']

    # Create a list of accuracy scores
    accuracy_scores = [accuracy_score_model1, accuracy_score_model2]

    # Plot the accuracy scores
    plt.bar(model_names, accuracy_scores)
    plt.xlabel('Model')
    plt.ylabel('Accuracy Score')
    plt.title('Comparison of Accuracy Scores')
    plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    # Display the plot using st.pyplot()
    st.pyplot(plt)


def main():
    comparison()
    com_graph()
    

if __name__ =='__main__':
    main()