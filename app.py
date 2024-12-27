import streamlit as st
import pandas as pd
import os
from datetime import datetime
from agent.agent_classifier import Agent
from utils.processing import compute_excel_accuracies
from openai import OpenAI

def validate_api_key(api_key):
    """Test if the OpenAI API key is valid by making a simple API call"""
    try:
        client = OpenAI(api_key=api_key)
        # Make a minimal API call to test the key
        response = client.models.list()
        return True
    except Exception as e:
        return False

def main():
    st.title("AMR Research Classification Tool")
    
    # Initialize session state for results DataFrame if it doesn't exist
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
        st.session_state.output_file = None
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API Key input in sidebar
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to continue")
        return
    
    # Validate API key
    if not validate_api_key(api_key):
        st.error("Invalid OpenAI API Key. Please check your key and try again.")
        return
    
    # Set API key for the session
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Model selection
    model = st.sidebar.selectbox(
        "Select Model",
        ["gpt-4o-mini", "o1-mini", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )
    
    # Parameters
    num_runs = st.sidebar.slider("Number of Runs", min_value=1, max_value=10, value=5)
    threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"Loaded {len(df)} entries from dataset")
            
            # Index range selection
            st.subheader("Select Range")
            start_index = st.number_input("Start Index", min_value=0, max_value=len(df)-1, value=0)
            num_entries = st.number_input("Number of Entries", min_value=1, max_value=len(df)-start_index, value=min(4000, len(df)-start_index))
            
            # Create columns for the download button and start button
            col1, col2 = st.columns([3, 1])
            start_button = col1.button("Start Classification")
            
            # Set output file name if not already set
            if st.session_state.output_file is None:
                model_abbreviation = {
                    "o1-mini": "o1",
                    "gpt-4o-mini": "4o",
                    "gemini-1.5-flash": "flash",
                    "gemini-1.5-pro": "pro",
                }.get(model, "4o")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.output_file = f"{model_abbreviation}_{uploaded_file.name.split('.')[0]}_{start_index}_{num_entries}_{timestamp}.xlsx"
            
            # Show current results if they exist
            if st.session_state.results_df is not None:
                st.subheader("Current Results")
                st.dataframe(st.session_state.results_df)
                
                # Save current results and provide download button
                st.session_state.results_df.to_excel(st.session_state.output_file, index=False)
                with open(st.session_state.output_file, "rb") as file:
                    col2.download_button(
                        label="Download Current Results",
                        data=file,
                        file_name=st.session_state.output_file,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            if start_button:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize agent
                agent = Agent(
                    model=model,
                    num_runs=num_runs,
                    threshold=threshold,
                    output_file=st.session_state.output_file
                )
                
                # Process entries
                try:
                    for i, current_index in enumerate(range(start_index, min(start_index + num_entries, len(df)))):
                        status_text.text(f"Processing entry {current_index} of {len(df)}")
                        
                        # Get the row data
                        row = df.iloc[current_index]
                        
                        # Process the entry
                        agent.perform_classification(
                            index=current_index,
                            title=str(row["Title"]),
                            abstract=str(row["Abstract"]),
                            original_id=str(row["Id"]),
                            ground_truth=str(row["Categories"])
                        )
                        
                        # Update progress
                        progress = (i + 1) / num_entries
                        progress_bar.progress(progress)
                        
                        # Update session state with current results
                        st.session_state.results_df = agent.results_df
                        
                        # Rerun to update the display
                        st.experimental_rerun()
                    
                    st.success("Classification completed!")
                    
                    # Compute and display accuracies
                    st.subheader("Results Analysis")
                    with st.spinner("Computing accuracies..."):
                        compute_excel_accuracies(
                            file_path=st.session_state.output_file,
                            print_options={
                                "level_wise": True,
                                "prediction_wise": True,
                                "misclassifications": True,
                                "constellations": True,
                            },
                            viz_options={
                                "visualize_analysis": True,
                                "save_plots": True,
                                "plot_save_dir": "results/plots/",
                            },
                        )
                
                except Exception as e:
                    st.error(f"Error in classification process: {e}")
                    raise
        
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main() 