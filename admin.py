import pandas as pd
import os
import streamlit as st


ASSET_INDICES_CSV = 'assets_indices_map.csv'
STATIC_INDICES_CSV = 'static_indices.csv'

def show_asset_indices_admin():
    st.logo("logo.png")
    st.title("Admin Panel")
    st.warning("Warning: Changes here will permanently modify CSV files. Proceed with caution!")

    # Tab selection
    tab1, tab2 = st.tabs(["Asset/Index Information", "Static Indices Data"])
    
    with tab1:
        edit_asset_indices()
    
    with tab2:
        edit_static_indices()

def edit_asset_indices():
    st.markdown("### Edit Asset/Index Information")
    
    # Load the CSV or session state
    if 'edited_df' not in st.session_state:
        if os.path.exists(ASSET_INDICES_CSV):
            st.session_state.edited_df = pd.read_csv(ASSET_INDICES_CSV)
        else:
            st.error("CSV file not found!")
            return

    edited_df = st.session_state.edited_df

    # Show the table
    st.dataframe(edited_df)

    # Edit existing rows
    st.markdown("#### Edit Existing Entries")
    edited_df = st.data_editor(edited_df, num_rows="dynamic", use_container_width=True)
    st.session_state.edited_df = edited_df

    # Add a new row
    st.markdown("#### Add New Entry")
    with st.form("add_row_form"):
        new_row = {}
        for col in edited_df.columns:
            new_row[col] = st.text_input(f"{col}", key=f"new_{col}")
        add_row = st.form_submit_button("Add Row")
        if add_row:
            st.session_state.edited_df = pd.concat([st.session_state.edited_df, pd.DataFrame([new_row])], ignore_index=True)
            st.success("New row added (not yet saved)!")

    # Save changes with confirmation
    if st.button("Save Asset/Index Changes"):
        st.session_state.edited_df.to_csv(ASSET_INDICES_CSV, index=False)
        st.success("Changes saved to asset_indices_map.csv!")

def edit_static_indices():
    st.markdown("### Edit Static Indices Data")
    
    # Load the static indices CSV
    if 'edited_static_df' not in st.session_state:
        if os.path.exists(STATIC_INDICES_CSV):
            st.session_state.edited_static_df = pd.read_csv(STATIC_INDICES_CSV)
            # Convert date column to datetime for better editing
            st.session_state.edited_static_df['date'] = pd.to_datetime(st.session_state.edited_static_df['date'])
        else:
            st.error("static_indices.csv file not found!")
            return

    static_df = st.session_state.edited_static_df

    # Filter by index name for easier editing
    st.markdown("#### Filter by Index")
    available_indices = static_df['index_name'].unique()
    selected_index = st.selectbox("Select Index to Edit", 
                                 options=['All'] + list(available_indices),
                                 key="static_index_filter")

    # Filter data based on selection
    if selected_index != 'All':
        filtered_df = static_df[static_df['index_name'] == selected_index].copy()
        st.markdown(f"#### Editing: {selected_index}")
    else:
        filtered_df = static_df.copy()
        st.markdown("#### Editing: All Indices")

    # Display basic info
    if selected_index != 'All':
        st.info(f"Total entries for {selected_index}: {len(filtered_df)}")
        if len(filtered_df) > 0:
            st.info(f"Date range: {filtered_df['date'].min().date()} to {filtered_df['date'].max().date()}")

    # Show the filtered table
    st.dataframe(filtered_df, use_container_width=True)

    # Edit existing rows
    st.markdown("#### Edit Existing Entries")
    # Configure column types for better editing
    column_config = {
        "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "index_name": st.column_config.TextColumn("Index Name"),
        "last": st.column_config.NumberColumn("Value", format="%.6f")
    }
    
    edited_static_df = st.data_editor(
        filtered_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config=column_config,
        key="static_data_editor"
    )

    # Update the session state with edited data
    if selected_index != 'All':
        # Update only the filtered rows in the main dataframe
        mask = st.session_state.edited_static_df['index_name'] == selected_index
        st.session_state.edited_static_df = st.session_state.edited_static_df[~mask]
        st.session_state.edited_static_df = pd.concat([st.session_state.edited_static_df, edited_static_df], ignore_index=True)
        st.session_state.edited_static_df = st.session_state.edited_static_df.sort_values(['index_name', 'date']).reset_index(drop=True)
    else:
        st.session_state.edited_static_df = edited_static_df

    # Add new entries
    st.markdown("#### Add New Entry")
    with st.form("add_static_row_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_date = st.date_input("Date", key="new_static_date")
        with col2:
            new_index_name = st.selectbox("Index Name", 
                                        options=available_indices, 
                                        key="new_static_index_name")
        with col3:
            new_value = st.number_input("Value", format="%.6f", key="new_static_value")
        
        add_static_row = st.form_submit_button("Add Entry")
        if add_static_row:
            new_static_row = {
                'date': pd.to_datetime(new_date),
                'index_name': new_index_name,
                'last': new_value
            }
            st.session_state.edited_static_df = pd.concat([st.session_state.edited_static_df, pd.DataFrame([new_static_row])], ignore_index=True)
            st.session_state.edited_static_df = st.session_state.edited_static_df.sort_values(['index_name', 'date']).reset_index(drop=True)
            st.success("New entry added (not yet saved)!")
            st.rerun()


    # Save changes with confirmation
    st.markdown("#### Save Changes")
    if st.button("Save Static Indices Changes", type="primary"):
        try:
            # Convert date back to string format for CSV
            save_df = st.session_state.edited_static_df.copy()
            save_df['date'] = save_df['date'].dt.strftime('%Y-%m-%d')
            save_df.to_csv(STATIC_INDICES_CSV, index=False)
            st.success("Changes saved to static_indices.csv!")
        except Exception as e:
            st.error(f"Error saving file: {e}")

    # Back button
    if st.button("Back to Main Menu"):
        st.session_state['page'] = 1