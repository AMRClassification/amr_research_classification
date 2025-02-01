import streamlit as st
import pandas as pd
import os
from datetime import datetime
from agent.agent_classifier import Agent
from openai import OpenAI
import io
from pathlib import Path

def validate_api_key(api_key):
    """Test if the OpenAI API key is valid by making a simple API call"""
    try:
        client = OpenAI(api_key=api_key)
        # Make a minimal API call to test the key
        response = client.models.list()
        return True
    except Exception as e:
        return False

def setup_data_directory():
    """Create data directory if it doesn't exist"""
    data_dir = Path("data/results")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def get_saved_results():
    """Get list of all saved result files"""
    data_dir = setup_data_directory()
    return sorted(
        [f for f in data_dir.glob("*.xlsx")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

def delete_result(file_path):
    """Delete a result file"""
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        st.error(f"Error deleting file: {e}")
        return False

def main():
    st.title("AMR Research Classification Tool")
    
    # Setup data directory
    data_dir = setup_data_directory()
    
    # Add a section for viewing recent results
    st.sidebar.markdown("---")
    st.sidebar.header("Recent Results")
    
    saved_results = get_saved_results()
    if saved_results:
        for result_file in saved_results:
            col1, col2, col3 = st.sidebar.columns([2, 1, 1])
            
            # Display filename
            col1.write(result_file.name)
            
            # Download button
            with open(result_file, "rb") as file:
                col2.download_button(
                    label="📥",
                    data=file,
                    file_name=result_file.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Delete button
            if col3.button("🗑️", key=f"delete_{result_file.name}"):
                if delete_result(result_file):
                    st.sidebar.success(f"Deleted {result_file.name}")
                    st.rerun()
    else:
        st.sidebar.info("No saved results found")
    
    # Initialize session state variables
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'output_file' not in st.session_state:
        st.session_state.output_file = None
    if 'current_index' not in st.session_state:
        st.session_state.current_index = None
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
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
        ["gpt-4o-mini", "o1-mini"],
        index=0,
        disabled=st.session_state.is_running
    )
    
    st.sidebar.markdown("---")  # Add a separator line

    # Parameters
    num_runs = st.sidebar.slider("Number of Runs", min_value=1, max_value=10, value=5, 
                               disabled=st.session_state.is_running)
    
    # Only show required consistent runs slider if num_runs > 1
    if num_runs > 1:
        required_consistent = st.sidebar.slider("Number of Required Consistent Runs", 
                                             min_value=1, 
                                             max_value=num_runs, 
                                             value=min(4, num_runs),  # Default to 4 or max possible
                                             disabled=st.session_state.is_running)
    else:
        # If only one run, then that run must be consistent
        required_consistent = 1
    
    # Calculate threshold from required consistent results
    threshold = required_consistent / num_runs
    st.sidebar.text(f"Calculated uncertainty threshold: {threshold:.2f}")
    st.sidebar.info("""
    The uncertainty threshold determines when a classification is considered reliable. 
    
    • It's calculated as: (Required Consistent Runs) / (Total Runs)
    • Example: If 4 out of 5 runs must agree → threshold = 0.8
    • Higher threshold = more strict agreement required
    • Lower threshold = more lenient agreement accepted
    
    When the proportion of consistent results meets or exceeds this threshold, 
    the classification is accepted. Otherwise, it's marked as uncertain.""")
    
    st.sidebar.markdown("---")  # Add a separator line

    # File upload
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'], 
                                   disabled=st.session_state.is_running)
    
    if uploaded_file is not None:
        try:
            # Only load the file if it's not already in session state
            if st.session_state.df is None:
                df = pd.read_excel(uploaded_file)
                
                # Add prediction columns if they don't exist
                if "Prediction" not in df.columns:
                    df["Prediction"] = ""
                if "Sector Explanation" not in df.columns:
                    df["Sector Explanation"] = ""
                if "Research Area Explanation" not in df.columns:
                    df["Research Area Explanation"] = ""
                if "Infectious Agent Explanation" not in df.columns:
                    df["Infectious Agent Explanation"] = ""
                
                # Store in session state
                st.session_state.df = df
            
            # Use the DataFrame from session state
            df = st.session_state.df
            
            # Map the actual column names to our expected names
            title_column = "Title"  # This matches
            abstract_column = "Abstract"  # This matches
            
            # Verify required columns exist
            if title_column not in df.columns or abstract_column not in df.columns:
                st.error(f"Excel file must contain '{title_column}' and '{abstract_column}' columns")
                return
            
            # Check if DataFrame is empty
            if df.empty:
                st.error("The uploaded Excel file is empty")
                return
                
            st.success(f"Loaded {len(df)} entries from dataset")
            
            # Display sample of the data
            st.write("First few rows of the dataset:")
            st.dataframe(df[["Id", "Title", "Abstract"]].head())
            
            # Index range selection
            st.subheader("Select Range")
            start_index = st.number_input("Start Index", min_value=0, max_value=len(df)-1, value=0,
                                        disabled=st.session_state.is_running)
            num_entries = st.number_input("Number of Entries", min_value=1, max_value=len(df)-start_index, 
                                        value=min(4000, len(df)-start_index),
                                        disabled=st.session_state.is_running)
            
            # Create columns for the download button and start/stop buttons
            col1, col2 = st.columns([3, 1])
            
            # Initialize start_button with a default value
            start_button = False
            
            if not st.session_state.is_running:
                if col1.button("Start Classification"):
                    # Generate output filename at classification start
                    model_abbreviation = {
                        "o1-mini": "o1",
                        "gpt-4o-mini": "4o",
                        "gemini-1.5-flash": "flash",
                        "gemini-1.5-pro": "pro",
                    }.get(model, "4o")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state.output_file = data_dir / f"{model_abbreviation}_{uploaded_file.name.split('.')[0]}_S{start_index}_N{num_entries}_{timestamp}.xlsx"
                    
                    # Initialize agent with current parameters
                    st.session_state.agent = Agent(
                        model=model,
                        num_runs=num_runs,
                        threshold=threshold,
                        output_file=st.session_state.output_file,
                        eval_mode=False  # Specify app mode
                    )
                    # Reset all state variables
                    st.session_state.current_index = start_index
                    st.session_state.progress = 0
                    st.session_state.is_running = True
                    st.session_state.last_entry = None  # Add this to track the most recent entry
                    st.rerun()
            else:
                # Add a running indicator and stop button
                col1.markdown("🔄 **Classification in progress...**")
                if col1.button("Stop Classification", type="primary"):
                    st.session_state.is_running = False
                    st.session_state.current_index = None
                    st.session_state.progress = 0
                    st.session_state.agent = None
                    st.session_state.output_file = None  # Reset output file path
                    st.warning("Classification stopped. Results are saved in the Excel file.")
                    st.rerun()
            
            # Show current results if they exist
            if st.session_state.df is not None:
                st.subheader("Current Results")
                
                # Create a view of only the processed entries
                if df is not None:
                    last_processed_index = st.session_state.current_index
                    if not st.session_state.is_running:
                        last_processed_index = min(start_index + num_entries, len(df))
                    
                    # Only show entries that have been processed
                    processed_entries = df.iloc[start_index:last_processed_index].copy()
                    
                    # Only show entries that have predictions (non-empty)
                    processed_entries = processed_entries[processed_entries['Prediction'].notna() & 
                                                       (processed_entries['Prediction'] != '')]
                    
                    if not processed_entries.empty:
                        # Show only Id, Title, Abstract, and Prediction
                        display_df = processed_entries[["Id", "Title", "Abstract", "Prediction"]]
                        st.dataframe(display_df)
                        
                        # Create Excel file in memory for download
                        output = io.BytesIO()
                        processed_entries.to_excel(output, index=False)
                        output.seek(0)
                        
                        # Provide download button with in-memory file
                        col2.download_button(
                            label="Download Complete Results",
                            data=output,
                            file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        # Show the most recent entry in detail only if we have a last entry and are running or just finished
                        if st.session_state.is_running or (not st.session_state.is_running and st.session_state.last_entry is not None):
                            st.subheader("Most Recent Classification")
                            last_entry = processed_entries.iloc[-1]
                            st.session_state.last_entry = last_entry  # Store the last entry
                            st.markdown(f"**Index:** {start_index + len(processed_entries) - 1}")
                            st.markdown(f"**ID:** {last_entry['Id']}")
                            st.markdown(f"**Title:** {last_entry['Title']}")
                            st.markdown(f"**Abstract:** {last_entry['Abstract'][:500]}...")
                            # Split prediction into lines and display each part on a new line
                            st.markdown("**Prediction:**")
                            if "less than 500 characters" in str(last_entry['Prediction']):
                                st.error("⚠️ Entry skipped: Content length below minimum threshold (500 characters)")
                            else:
                                predictions = last_entry['Prediction'].split('/')
                                current_category = None
                                for pred in predictions:
                                    pred = pred.strip()
                                    if pred:
                                        if any(category in pred for category in ['Sector', 'Research Area', 'Infectious Agent']):
                                            current_category = pred
                                            st.markdown(f"\n:blue[**{current_category}**]")
                                        else:
                                            st.markdown(f"• {pred}")
                    else:
                        st.info("No processed entries with predictions yet.")
                else:
                    st.info("Waiting for classifications to begin...")
            
            # Create persistent progress bar
            progress_bar = st.progress(st.session_state.progress)
            status_text = st.empty()
            
            # Start or continue classification if running
            if st.session_state.is_running:
                # Process entry
                end_index = start_index + num_entries  # Calculate the end index
                if st.session_state.current_index < end_index:
                    try:
                        current_index = st.session_state.current_index
                        status_text.text(f"Processing entry {current_index - start_index + 1} of {num_entries}")
                        
                        # Process the entry and update DataFrame
                        try:
                            updated_df = st.session_state.agent.perform_classification(
                                index=current_index,
                                title=str(df.iloc[current_index]["Title"]),
                                abstract=str(df.iloc[current_index]["Abstract"]),
                                input_df=df
                            )
                            
                            # Update the session state DataFrame
                            st.session_state.df = updated_df
                            
                            # Save results after each successful classification
                            result_df = updated_df.iloc[start_index:end_index].copy()
                            result_df.to_excel(st.session_state.output_file, index=False)
                            
                            # Update display immediately after classification
                            processed_entries = updated_df.iloc[start_index:current_index + 1]
                            if not processed_entries.empty:
                                st.subheader("Current Results")
                                display_df = processed_entries[["Id", "Title", "Abstract", "Prediction", 
                                                             "Sector Explanation", 
                                                             "Research Area Explanation",
                                                             "Infectious Agent Explanation"]]
                                st.dataframe(display_df)
                                
                                # Show warning for skipped entries
                                last_entry = processed_entries.iloc[-1]
                                if "less than 500 characters" in str(last_entry['Prediction']):
                                    st.warning(f"Entry {current_index} skipped: Content length below minimum threshold")
                        
                        except Exception as e:
                            import traceback
                            st.error("Full error traceback:")
                            st.code(traceback.format_exc())
                            raise
                        
                        # Update progress
                        st.session_state.progress = (current_index - start_index + 1) / num_entries
                        progress_bar.progress(st.session_state.progress)
                        
                        # Increment index for next iteration
                        st.session_state.current_index += 1
                        
                        # Force update of UI
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error in classification process: {e}")
                        st.session_state.current_index = None
                        st.session_state.is_running = False
                        raise
                else:
                    # Classification completed
                    st.session_state.current_index = None
                    st.session_state.is_running = False
                    st.session_state.progress = 0
                    st.success("Classification completed!")
                    st.rerun()  # Force UI update to reset controls
                    
                    # # Compute and display accuracies - commented out
                    # st.subheader("Results Analysis")
                    # with st.spinner("Computing accuracies..."):
                    #     compute_excel_accuracies(
                    #         file_path=st.session_state.output_file,
                    #         print_options={
                    #             "level_wise": True,
                    #             "prediction_wise": True,
                    #             "misclassifications": True,
                    #             "constellations": True,
                    #         },
                    #         viz_options={
                    #             "visualize_analysis": True,
                    #             "save_plots": True,
                    #             "plot_save_dir": "results/plots/",
                    #         },
                    #     )
        
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.session_state.is_running = False

if __name__ == "__main__":
    main() 