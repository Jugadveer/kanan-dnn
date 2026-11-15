"""Entry point of the app: the board header and the navigation between pages.

Run from the project folder:
    python -m streamlit run app/Home.py
"""

import streamlit as st

from shared import TITLE, board_header

st.set_page_config(page_title=TITLE, layout="wide")
board_header()
pages = [
    st.Page("pages/0_Board.py", title="Board", default=True),
    st.Page("pages/1_Check_a_student.py", title="Check a student", url_path="check"),
    st.Page("pages/2_Model_report.py", title="Model report", url_path="report"),
    st.Page("pages/3_What_drives_predictions.py", title="What drives predictions", url_path="drivers"),
    st.Page("pages/4_Experiments.py", title="Experiments", url_path="experiments"),
]
st.navigation(pages, position="top").run()
