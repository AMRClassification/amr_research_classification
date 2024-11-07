import matplotlib.pyplot as plt
import seaborn as sns


def plot_error_constellations(
    constellation_patterns, top_n=30, figsize=(15, 10), save_path=None
):
    """
    Creates bar charts for error constellations in each domain.

    Args:
        constellation_patterns (dict): Dictionary of error patterns per domain
        top_n (int): Number of top errors to show
        figsize (tuple): Figure size (width, height)
        save_path (str): Optional path to save the plots
    """
    domains = ["Sector", "Research Area", "Infectious Agent"]

    for domain in domains:
        # Sort patterns by frequency
        sorted_patterns = sorted(
            constellation_patterns[domain].items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        if not sorted_patterns:
            print(f"No error patterns found for {domain}")
            continue

        # Create figure
        plt.figure(figsize=figsize)

        # Extract data for plotting
        patterns = [p[0] for p in sorted_patterns]
        counts = [p[1] for p in sorted_patterns]

        # Create bar plot
        bars = plt.bar(range(len(patterns)), counts)

        # Customize plot
        plt.title(f"Top {top_n} Error Constellations - {domain}", pad=20)
        plt.xlabel("Error Patterns")
        plt.ylabel("Frequency")

        # Rotate x-axis labels for better readability
        plt.xticks(range(len(patterns)), patterns, rotation=45, ha="right")

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            plt.savefig(
                f"{save_path}/error_constellations_{domain.lower().replace(' ', '_')}.png",
                bbox_inches="tight",
                dpi=300,
            )

        plt.show()
