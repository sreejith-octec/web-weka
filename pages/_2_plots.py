import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():
    # Get the root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct the file path relative to the root directory
    file_path = os.path.join(root_dir, "Preprocessed_data.csv")

    # Read the preprocessed_data.csv file
    data = pd.read_csv(file_path)

    columns = data.columns.tolist()


    st.title("Heat Map")
    fig, ax = plt.subplots(figsize=(25, 15))
    sns.heatmap(data.corr(), annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, ax=ax)
    plt.title("Correlation")
    plt.xlabel("Features")
    plt.ylabel("Features")

    # Display the figure using st.pyplot()
    st.pyplot(fig)

    st.title("Scatter Plots between features:")

    # Sidebar selection
    st.sidebar.title('Select Features')
    x_axis = st.sidebar.selectbox('X-axis', columns)
    y_axis = st.sidebar.selectbox('Y-axis', columns)

    # Generate scatter plot
    fig1, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(data[x_axis], data[y_axis])
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title('Scatter Plot')
    st.pyplot(fig1)


if __name__ == '__main__':
    main()
