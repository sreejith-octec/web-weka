import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Import SessionState class
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

def evaluate_random_forest(data_new, training_method):

    # Split the dataset into features and target variable
    X = data_new.drop("y", axis=1)
    y = data_new['y']
    random_forest = RandomForestClassifier()


    if training_method >= 500 :

        # Train-test split based on the training ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - (training_method-500) / 100), random_state=42)

        # Initialize and train the Random Forest model
        random_forest.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = random_forest.predict(X_test)

        # Calculate prediction accuracy
        accuracy = accuracy_score(y_test, y_pred)

        pickle_file_name = 'rf_model_split_' + str(training_method-500) + '_acc_' + str(accuracy) +'.pkl'
        pickle_file_path = os.path.join('models', pickle_file_name)

        # Check if the file already exists
        if os.path.exists(pickle_file_path):
            print("File already exists. Overwriting...")

        # Save the trained model as a pickle file
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(random_forest, file)

        st.subheader("Evaluation Results")
        st.write("Prediction Accuracy:", accuracy)
        st.write("Confusion Matrix:")
        plot_confusion_matrix(y_test, y_pred)

    else:

        # Perform k-fold cross-validation
        scores = cross_val_score(random_forest, X, y, cv=training_method)

        # Calculate mean accuracy across all folds
        accuracy = scores.mean()

        # Calculate confusion matrix using the last fold
        random_forest.fit(X, y)
        y_pred = random_forest.predict(X)

        pickle_file_name = 'rf_model_kfold_' + str(training_method) + '_acc_' + str(accuracy) +'.pkl'
        pickle_file_path = os.path.join('models', pickle_file_name)

        # Check if the file already exists
        if os.path.exists(pickle_file_path):
            print("File already exists. Overwriting...")

        # Save the trained model as a pickle file
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(random_forest, file)

        st.subheader("Evaluation Results")
        st.write("Prediction Accuracy:", accuracy)
        st.write("Confusion Matrix:")
        plot_confusion_matrix(y, y_pred)

def evaluate_logistic_regression(data_new, training_method):
    
    X = data_new.drop("y", axis=1)
    y = data_new['y']
    logistic_regression = LogisticRegression()

    if training_method >= 500:
        # Train-test split based on the training ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - (training_method-500) / 100), random_state=42)

        # Initialize and train the Logistic Regression model
        logistic_regression.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = logistic_regression.predict(X_test)

        # Calculate prediction accuracy
        accuracy = accuracy_score(y_test, y_pred)

        pickle_file_name = 'lr_model_split_' + str(training_method-500) + '_acc_' + str(accuracy) +'.pkl'
        pickle_file_path = os.path.join('models', pickle_file_name)

        # Check if the file already exists
        if os.path.exists(pickle_file_path):
            print("File already exists. Overwriting...")

        # Save the trained model as a pickle file
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(logistic_regression, file)

        st.subheader("Evaluation Results")
        st.write("Prediction Accuracy:", accuracy)
        st.write("Confusion Matrix:")
        plot_confusion_matrix(y_test, y_pred)

    else:
        # Perform k-fold cross-validation
        scores = cross_val_score(logistic_regression, X, y, cv=training_method)

        # Calculate mean accuracy across all folds
        accuracy = scores.mean()

        # Calculate confusion matrix using the last fold
        logistic_regression.fit(X, y)
        y_pred = logistic_regression.predict(X)

        pickle_file_name = 'lr_model_kfold_' + str(training_method) + '_acc_' + str(accuracy) +'.pkl'
        pickle_file_path = os.path.join('models', pickle_file_name)

        # Check if the file already exists
        if os.path.exists(pickle_file_path):
            print("File already exists. Overwriting...")

        # Save the trained model as a pickle file
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(logistic_regression, file)

        st.subheader("Evaluation Results")
        st.write("Prediction Accuracy:", accuracy)
        st.write("Confusion Matrix:")
        plot_confusion_matrix(y, y_pred)


def evaluate_svm(data_new, training_method):
    # Split the dataset into features and target variable
    X = data_new.drop("y", axis=1)
    y = data_new['y']
    svm = SVC()

    if training_method >= 500:
        # Train-test split based on the training ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - (training_method-500) / 100), random_state=42)

        # Initialize and train the SVM model
        svm.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = svm.predict(X_test)

        # Calculate prediction accuracy
        accuracy = accuracy_score(y_test, y_pred)

        pickle_file_name = 'svm_model_split_' + str(training_method-500) + '_acc_' + str(accuracy) +'.pkl'
        pickle_file_path = os.path.join('models', pickle_file_name)

        # Create the 'models' folder if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')

        # Save the trained model as a pickle file
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(svm, file)
        
        print("SVM model saved as pickle file:", pickle_file_path)

        st.subheader("Evaluation Results")
        st.write("Prediction Accuracy:", accuracy)
        st.write("Confusion Matrix:")
        plot_confusion_matrix(y_test, y_pred)
    else:
        # Perform k-fold cross-validation
        scores = cross_val_score(svm, X, y, cv=training_method)

        # Calculate mean accuracy across all folds
        accuracy = scores.mean()

        # Calculate confusion matrix using cross_val_predict
        y_pred = cross_val_predict(svm, X, y, cv=training_method)

        pickle_file_name = 'svm_model_kfold_' + str(training_method) + '_acc_' + str(accuracy) +'.pkl'
        pickle_file_path = os.path.join('models', pickle_file_name)

        # Create the 'models' folder if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')

        # Save the trained model as a pickle file
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(svm, file)

        print("SVM model saved as pickle file:", pickle_file_path)
        st.subheader("Evaluation Results")
        st.write("Prediction Accuracy:", accuracy)
        st.write("Confusion Matrix:")
        plot_confusion_matrix(y, y_pred)



def preprocess_data(data):
    
    if isinstance(data, pd.DataFrame):
        dataset = data
    else:
        dataset = pd.read_csv(data)

    dataset['age'] = dataset['age'].fillna(dataset['age'].median())
    dataset['balance'] = dataset['balance'].fillna(dataset['balance'].median())

    dataset['month'] = dataset['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                                             'jul': 7, 'aug': 8, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12})

    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    encoder = LabelEncoder()

    for col in categorical_cols:
        dataset[col] = encoder.fit_transform(dataset[col])
    
    if "y" in dataset.columns:
        dataset["y"] = dataset["y"].map({'yes':1, 'no':0})
        st.subheader(" PREPROCESSED DATASET")
        st.write(dataset.head(4))
        st.subheader("Data Description:")
        st.write(dataset.describe())
    else:
        pass
    
    dataset = dataset.dropna()
    return dataset


def main():
    
    global algorithm, training_ratio, k_value
    ratio:int = 0
    value:int = 0

    # Create or get the session state
    if 'session_state' not in st.session_state:
        st.session_state.session_state = SessionState(results={})


    # Sidebar setup
    st.sidebar.title('Sidebar')
    upload_file = st.sidebar.file_uploader('Upload your Dataset', type='csv')

    # Check if file has been uploaded
    if upload_file is not None:
        file_path = os.path.join(os.getcwd(), upload_file.name)
        with open(file_path, "wb") as f:
            f.write(upload_file.getbuffer())
        st.write("File saved:", file_path)
        df = pd.read_csv(upload_file)
        st.session_state.session_state.df = df

    st.subheader("Algorithm:")
    algorithm = st.selectbox("Select Algorithm", ("Random Forest", "Logistic Regression", "SVM"), index=0, key="algorithm")

    st.subheader("Training Method:")
    col1, col2 = st.columns(2)
    with col1:
        training = st.radio("Select Training Method", ("Training Ratio", "K fold Cross Validation"), index=0, key="training")

    if training == "Training Ratio":
        training_ratio = st.slider("Select Training Ratio", 0, 100, 70, 10)
        ratio = training_ratio
        st.write("Training Ratio:", training_ratio, ":", 100-training_ratio)
    elif training == "K fold Cross Validation":
        k_value = st.text_input("Enter K value for Cross Validation", "5")
        value = int(k_value)
        st.write("K value:", k_value)

    evaluate_button = st.button("Evaluate", disabled=False)

    if evaluate_button and upload_file is not None:
        st.subheader("DATASET")
        st.write(df.head())
        data_new = preprocess_data(file_path)
        preprocessed_file_path = os.path.join(os.getcwd(), "Preprocessed_data.csv")
        data_new.to_csv(preprocessed_file_path, index=False)
        st.write("Preprocessed data saved as Preprocessed_data.csv")

        if algorithm == "Random Forest" and ratio > 0:
            evaluate_random_forest(data_new,ratio+500)
        elif algorithm == "Random Forest" and value > 0:
            evaluate_random_forest(data_new,value)
        elif algorithm == "Logistic Regression" and ratio > 0:
            evaluate_logistic_regression(data_new,ratio+500)
        elif algorithm == "Logistic Regression" and value > 0:
            evaluate_logistic_regression(data_new, value)
        elif algorithm == "SVM" and ratio > 0:
            evaluate_svm(data_new,ratio+500)
        elif algorithm == "SVM" and value > 0:
            evaluate_svm(data_new,value)
        else:
            pass

        # Store the results in the session state
        st.session_state.session_state.results['home'] = 'Results from the home page'

    # Display the stored results
    if 'home' in st.session_state.session_state.results:
        st.write('Results:', st.session_state.session_state.results['home'])

if __name__ == '__main__':
    main()
