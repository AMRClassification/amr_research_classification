import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, 
                            QPushButton, QFileDialog, QProgressBar, QGroupBox, 
                            QMessageBox, QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pandas as pd
from datetime import datetime
from utils.processing import compute_excel_accuracies
from agent.agent_classifier import Agent
import os

class ClassificationWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    results_updated = pyqtSignal(object)  # For DataFrame updates

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.agent = None

    def run(self):
        try:
            # Initialize agent
            self.agent = Agent(
                model=self.params['model'],
                num_runs=self.params['num_runs'],
                threshold=self.params['threshold'],
                output_file=self.params['output_file']
            )

            df = self.params['df']
            start_index = self.params['start_index']
            num_entries = self.params['num_entries']

            for i, current_index in enumerate(range(start_index, min(start_index + num_entries, len(df)))):
                self.status.emit(f"Processing entry {current_index} of {len(df)}")
                
                # Get the row data
                row = df.iloc[current_index]
                
                # Process the entry
                self.agent.perform_classification(
                    index=current_index,
                    title=str(row["Title"]),
                    abstract=str(row["Abstract"]),
                    original_id=str(row["Id"]),
                    ground_truth=str(row["Categories"])
                )
                
                # Update progress
                progress = int((i + 1) / num_entries * 100)
                self.progress.emit(progress)
                
                # Emit current results
                self.results_updated.emit(self.agent.results_df)

            self.status.emit("Classification completed!")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))

class AMRClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.current_file = None
        self.output_file = None
        self.results_df = None
        self.worker = None
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('AMR Research Classification Tool')
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Configuration Group
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        
        # Model Selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-4o-mini", "o1-mini", "gemini-1.5-flash", "gemini-1.5-pro"])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        config_layout.addLayout(model_layout)
        
        # Number of Runs
        runs_layout = QHBoxLayout()
        runs_label = QLabel("Number of Runs:")
        self.num_runs_spin = QSpinBox()
        self.num_runs_spin.setRange(1, 10)
        self.num_runs_spin.setValue(5)
        runs_layout.addWidget(runs_label)
        runs_layout.addWidget(self.num_runs_spin)
        config_layout.addLayout(runs_layout)
        
        # Threshold
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(0.8)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_spin)
        config_layout.addLayout(threshold_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # File Selection
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.select_file_btn = QPushButton("Select File")
        self.select_file_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_file_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Range Selection
        range_group = QGroupBox("Range Selection")
        range_layout = QVBoxLayout()
        
        start_layout = QHBoxLayout()
        start_label = QLabel("Start Index:")
        self.start_index = QSpinBox()
        self.start_index.setRange(0, 999999)
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_index)
        range_layout.addLayout(start_layout)
        
        num_layout = QHBoxLayout()
        num_label = QLabel("Number of Entries:")
        self.num_entries = QSpinBox()
        self.num_entries.setRange(1, 999999)
        self.num_entries.setValue(4000)
        num_layout.addWidget(num_label)
        num_layout.addWidget(self.num_entries)
        range_layout.addLayout(num_layout)
        
        range_group.setLayout(range_layout)
        layout.addWidget(range_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Classification")
        self.start_button.clicked.connect(self.start_classification)
        self.download_button = QPushButton("Download Results")
        self.download_button.clicked.connect(self.download_results)
        self.download_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.download_button)
        layout.addLayout(button_layout)
        
    def select_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel files (*.xlsx)")
        if filename:
            try:
                self.df = pd.read_excel(filename)
                self.current_file = filename
                self.file_label.setText(f"Loaded {len(self.df)} entries from {os.path.basename(filename)}")
                self.num_entries.setValue(min(4000, len(self.df)))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading dataset: {str(e)}")
    
    def start_classification(self):
        if self.df is None:
            QMessageBox.warning(self, "Error", "Please select a file first")
            return
        
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Info", "Classification is already running")
            return
        
        # Set output file name
        model_abbreviation = {
            "o1-mini": "o1",
            "gpt-4o-mini": "4o",
            "gemini-1.5-flash": "flash",
            "gemini-1.5-pro": "pro",
        }.get(self.model_combo.currentText(), "4o")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f"{model_abbreviation}_{os.path.splitext(os.path.basename(self.current_file))[0]}_{self.start_index.value()}_{self.num_entries.value()}_{timestamp}.xlsx"
        
        # Prepare parameters
        params = {
            'model': self.model_combo.currentText(),
            'num_runs': self.num_runs_spin.value(),
            'threshold': self.threshold_spin.value(),
            'output_file': self.output_file,
            'df': self.df,
            'start_index': self.start_index.value(),
            'num_entries': self.num_entries.value()
        }
        
        # Create and start worker
        self.worker = ClassificationWorker(params)
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.update_status)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.classification_finished)
        self.worker.results_updated.connect(self.update_results)
        
        self.start_button.setEnabled(False)
        self.worker.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        self.status_label.setText(message)
    
    def handle_error(self, error_message):
        QMessageBox.critical(self, "Error", f"Error in classification process: {error_message}")
        self.start_button.setEnabled(True)
    
    def update_results(self, results_df):
        self.results_df = results_df
        self.results_df.to_excel(self.output_file, index=False)
        self.download_button.setEnabled(True)
    
    def classification_finished(self):
        self.start_button.setEnabled(True)
        try:
            compute_excel_accuracies(
                file_path=self.output_file,
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
            QMessageBox.warning(self, "Warning", f"Error computing accuracies: {str(e)}")
    
    def download_results(self):
        if self.output_file and os.path.exists(self.output_file):
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Results",
                os.path.basename(self.output_file),
                "Excel files (*.xlsx)"
            )
            if save_path:
                try:
                    self.results_df.to_excel(save_path, index=False)
                    QMessageBox.information(self, "Success", "Results saved successfully!")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error saving results: {str(e)}")

def main():
    app = QApplication(sys.argv)
    ex = AMRClassifierApp()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
