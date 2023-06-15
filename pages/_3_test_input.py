import streamlit as st
from pages import _1_home
import pickle
import pandas as pd
import os



# Function to load and return the selected pickle file
def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def accuracy_form(model_file):
    global float_digits

    file_name = model_file

    # Remove the extension ".pkl"
    file_name_without_extension = file_name.replace(".pkl", "")

    # Split the string by "_" and retrieve the last element
    last_element = file_name_without_extension.split("_")[-1]

    try:
        float_digits = float(last_element)
    except ValueError:
        print("No float digits found.")


def model_selection():
    global loaded_model
    
    # Get a list of pickle files in the "models" folder
    pickle_files = [file for file in os.listdir('models') if file.endswith('.pkl')]

    # Sidebar setup
    st.sidebar.title('Select Pickle File')
    selected_file = st.sidebar.selectbox('Pick a pickle file', pickle_files)
    accuracy_form(selected_file)

    # Check if a file is selected
    if selected_file:
        file_path = os.path.join('models', selected_file)
        loaded_model = load_pickle_file(file_path)

        # Use the loaded model for further processing
        st.write('Pickle file loaded:', selected_file)
        st.subheader("Selected Model: ")
        st.write(loaded_model)
    else:
        st.sidebar.info('No pickle file selected.')


def input_form():
    st.subheader("PREDICT TERM DEPOSIT")
    st.markdown("Please fill in the details below:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age:", min_value=0, step=1)
        job = st.text_input("Job:")
        marital = st.text_input("Marital:")
        education = st.text_input("Education:")
        default = st.text_input("Default:", help="Has credit in default? yes or no")
        balance = st.number_input("Balance:", help="Average yearly balance(in ten thousands)", min_value=0, step=1)

    with col2:
        housing = st.text_input("Housing:", help="Has housing loan? yes or no")
        loan = st.text_input("Loan:", help="Has personel loan? yes or no")
        contact = st.text_input("Contact:", help="Contact communication type? cellular or telephone or unknown")
        month = st.text_input("Month:", help="Last contact month of year? jan to dec")
        day_of_week = st.number_input("Day of Week:", help="Last contact day of month?", min_value=0, step=1)
        duration = st.number_input("Duration:", help="Last contact duration( in seconds)", min_value=0, step=1)

    campaign = st.number_input("Campaign:", help=" No of contacts performe din this campaign and for this client?", min_value=0, step=1)
    pdays = st.number_input("Pdays:",help="No of days that passed by after the client was last contacted from a previous campaign? not contacted -1", min_value=0, step=1)
    previous = st.number_input("Previous:",help="No of contacts performed before this campaign?", min_value=0, step=1)
    poutcome = st.text_input("Poutcome:", help="Outcome of the previous marketting campaign? unknown or other or failure or success")

    submit_button = st.button("Submit")

    if submit_button:
        test_input = [[age, job, marital, education, default, balance, housing, loan, contact, day_of_week, month, duration, campaign, pdays, previous, poutcome]]
        if test_input is not None:
            test_input_df = pd.DataFrame(test_input, columns = ['age', 'job', 'marital', 'education', 'default', 'balance',
                   'housing', 'loan', 'contact', 'day_of_week', 'month',
                   'duration', 'campaign', 'pdays', 'previous', 'poutcome'])
            preprocessed_input = _1_home.preprocess_data(test_input_df)
           
            # Apply the model to the new data
            predictions = loaded_model.predict(preprocessed_input)
            if predictions == 0:
                st.title("This client is not Subscribed to Term Deposit")
            else:
                st.title("This client is Subscribed to Term Deposit")

            st.write("Accuracy:", float_digits)



def main():
    model_selection()
    input_form()

if __name__ == '__main__':
    main()
