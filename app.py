# ============================================================
# MAIN STREAMLIT NAVIGATION
# This file controls the names displayed in the left sidebar.
# The actual plotting applications are stored in separate files.
# ============================================================

import streamlit as st


# The original Universal XY plotting application.
# default=True makes this the first page shown when the app opens.
tensile_page = st.Page(
    "tensile_plotter.py",
    title="Tensile and/or Other Data Plotter",
    default=True,
)


# The Raman plotting application.
raman_page = st.Page(
    "pages/2_Raman_Data_Plotter.py",
    title="Raman Data Plotter",
)


# Display both applications in the left sidebar.
selected_page = st.navigation(
    [
        tensile_page,
        raman_page,
    ],
    position="sidebar",
)


# Run whichever page the user selected.
selected_page.run()
