import pandas as pd
import os
import streamlit as st


ASSET_INDICES_CSV = 'assets_indices_map.csv'

def show_asset_indices_admin():
    st.title("Edit Asset/Index Information (Admin)")
    st.warning("Warning: Changes here will permanently modify the asset_indices_map.csv file. Proceed with caution!")

    # Load the CSV
    if os.path.exists(ASSET_INDICES_CSV):
        df = pd.read_csv(ASSET_INDICES_CSV)
    else:
        st.error("CSV file not found!")
        return

    # Show the table
    st.dataframe(df)

    # Edit existing rows
    st.markdown("#### Edit Existing Entries")
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    # Add a new row
    st.markdown("#### Add New Entry")
    with st.form("add_row_form"):
        new_row = {}
        for col in df.columns:
            new_row[col] = st.text_input(f"{col}", key=f"new_{col}")
        add_row = st.form_submit_button("Add Row")
        if add_row:
            edited_df = edited_df.append(new_row, ignore_index=True)
            st.success("New row added (not yet saved)!")

    # Save changes with confirmation
    if st.button("Save Changes"):
        if st.checkbox("I understand this will overwrite the CSV file and cannot be undone.", key="confirm_save"):
            edited_df.to_csv(ASSET_INDICES_CSV, index=False)
            st.success("Changes saved to asset_indices_map.csv!")
        else:
            st.error("You must confirm before saving changes.")

    if st.button("Back"):
        st.session_state['page'] = 1

