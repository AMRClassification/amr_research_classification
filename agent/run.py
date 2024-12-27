import subprocess
import os
import sys

if __name__ == "__main__":
    # Get the directory containing the executable
    if getattr(sys, 'frozen', False):
        # Running as executable
        app_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up the Streamlit command
    streamlit_command = ["streamlit", "run", 
                        os.path.join(app_dir, "app.py"),
                        "--server.address", "localhost",
                        "--server.port", "8080"]
    
    # Run the Streamlit app
    subprocess.run(streamlit_command) 