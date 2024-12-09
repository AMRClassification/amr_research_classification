import os
import sys

sys.path.insert(1, os.getcwd())

from langgraph.graph import END, START
from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod
from PIL import Image as PILImage
import io
import requests
from requests.exceptions import Timeout, RequestException

from agent.agent_classifier import Agent

def create_workflow_visualization():
    """Create a visual representation of the classification workflow."""
    # Create an instance of the classifier to get the workflow
    classifier = Agent(model="gpt-4o-mini")
    
    try:
        # Get the compiled graph
        graph = classifier.app.get_graph()

        # Generate the Mermaid diagram as PNG
        graph_png = graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API)

        if graph_png:
            # Save the PNG to a file
            with open("agent_workflow.png", "wb") as f:
                f.write(graph_png)
            print("Workflow graph saved as 'agent_workflow.png'")

            # Display the image
            display(Image(graph_png))

            # Open the image in default viewer
            image = PILImage.open(io.BytesIO(graph_png))
            image.show()  # This will open the image in your default image viewer
        else:
            print("Failed to generate workflow visualization")

    except Timeout:
        print("Timeout while generating visualization. The server took too long to respond.")
    except RequestException as e:
        print(f"Network error while generating visualization: {e}")
    except Exception as e:
        print(f"Error generating workflow visualization: {e}")

if __name__ == "__main__":
    create_workflow_visualization()
