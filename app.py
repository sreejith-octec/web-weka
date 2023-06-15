import streamlit as st
from pages import _1_home, _2_plots, _3_test_input,_4_compare

def main():
    st.set_page_config(
    page_title="ML App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    
)
    selected_page = st.sidebar.selectbox("Select a page", ("home", "plots", "test_input", "compare"))

    # Conditional statements based on selected page
    if selected_page == "home":
        _1_home.main()
    elif selected_page == "plots":
        _2_plots.main()
    elif selected_page == "test_input":
        _3_test_input.main()
    elif selected_page == "compare":
        _4_compare.main()


if __name__ =='__main__':
    main()