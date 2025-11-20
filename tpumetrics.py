import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class PerformanceLogAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Performance Log Analyzer")
        self.root.geometry("1200x800")
        
        self.df = None
        self.file_path = None
        
        # Setup matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Data Source", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.file_label = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        ttk.Button(file_frame, text="Browse CSV", command=self.load_csv).grid(row=0, column=1)
        
        # Control panel frame
        control_frame = ttk.LabelFrame(main_frame, text="Analysis Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Metric selection
        ttk.Label(control_frame, text="Select Metric:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.metric_var = tk.StringVar(value="Latency")
        metrics = ["Latency", "FPS", "Jitter"]
        
        for i, metric in enumerate(metrics):
            ttk.Radiobutton(control_frame, text=metric, variable=self.metric_var, 
                           value=metric).grid(row=i+1, column=0, sticky=tk.W, padx=(10, 0))
        
        # Outlier filtering
        ttk.Label(control_frame, text="Outlier Filter:", font=('Arial', 10, 'bold')).grid(row=5, column=0, sticky=tk.W, pady=(20, 5))
        
        self.filter_var = tk.StringVar(value="None")
        filters = [
            ("None (Raw Data)", "None"),
            ("Exclude > 10% from Avg", "10"),
            ("Exclude > 25% from Avg", "25"),
            ("Exclude > 50% from Avg", "50"),
            ("Exclude > 100% from Avg", "100")
        ]
        
        for i, (label, value) in enumerate(filters):
            ttk.Radiobutton(control_frame, text=label, variable=self.filter_var, 
                           value=value).grid(row=i+6, column=0, sticky=tk.W, padx=(10, 0))
        
        # Plot button
        ttk.Button(control_frame, text="Plot Graph", command=self.plot_graph, 
                  style="Accent.TButton").grid(row=12, column=0, pady=(30, 10), sticky=(tk.W, tk.E))
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="10")
        stats_frame.grid(row=13, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.stats_text = tk.Text(stats_frame, height=12, width=30, font=('Courier', 9))
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.stats_text.insert('1.0', "Load data and plot to see statistics")
        self.stats_text.config(state='disabled')
        
        # Graph frame
        graph_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        graph_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        graph_frame.columnconfigure(0, weight=1)
        graph_frame.rowconfigure(0, weight=1)
        
        # Placeholder for matplotlib canvas
        self.canvas_frame = ttk.Frame(graph_frame)
        self.canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initial placeholder
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.text(0.5, 0.5, 'Load CSV and click "Plot Graph"', 
                    ha='center', va='center', fontsize=14, color='gray')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.df = pd.read_csv(file_path)
            
            # Check for latency column (support both ms and us)
            latency_col = None
            if 'latency_us' in self.df.columns:
                latency_col = 'latency_us'
                self.latency_unit = 'us'
                self.latency_multiplier = 1
            elif 'inference_latency_ms' in self.df.columns:
                latency_col = 'inference_latency_ms'
                self.latency_unit = 'ms'
                self.latency_multiplier = 1000  # Convert to us for FPS calc
            else:
                messagebox.showerror("Error", "CSV must contain 'latency_us' or 'inference_latency_ms' column")
                return
            
            # Process data
            self.df['latency'] = pd.to_numeric(self.df[latency_col], errors='coerce')
            
            # Calculate FPS (avoid division by zero)
            self.df['fps'] = self.df['latency'].apply(
                lambda x: 1000000 / (x * self.latency_multiplier) if x > 0 else 0
            )
            
            # Calculate Jitter
            self.df['jitter'] = self.df['latency'].diff().abs()
            
            # Remove NaN values
            self.df = self.df.dropna(subset=['latency'])
            
            if len(self.df) == 0:
                messagebox.showerror("Error", "No valid numeric data found in latency column")
                return
            
            self.file_path = file_path
            self.file_label.config(text=f"Loaded: {file_path.split('/')[-1]} ({len(self.df)} records)", 
                                  foreground="green")
            messagebox.showinfo("Success", f"Successfully loaded {len(self.df)} records")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def apply_outlier_filter(self, data):
        """Apply outlier filtering based on percentage deviation from mean"""
        filter_pct = self.filter_var.get()
        
        if filter_pct == "None":
            return data
        
        threshold = float(filter_pct) / 100.0
        mean_val = data.mean()
        lower_bound = mean_val * (1 - threshold)
        upper_bound = mean_val * (1 + threshold)
        
        filtered = data[(data >= lower_bound) & (data <= upper_bound)]
        return filtered
    
    def calculate_statistics(self, data):
        """Calculate key statistics"""
        stats = {
            'Count': len(data),
            'Min': data.min(),
            'Max': data.max(),
            'Mean': data.mean(),
            'Median': data.median(),
            'Std Dev': data.std(),
            'P99': data.quantile(0.99),
            'P95': data.quantile(0.95),
            'P90': data.quantile(0.90)
        }
        return stats
    
    def plot_graph(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return
        
        metric = self.metric_var.get()
        
        # Map metric to column
        metric_map = {
            'Latency': ('latency', f'Latency ({self.latency_unit})', '#2E86AB'),
            'FPS': ('fps', 'Frames Per Second', '#A23B72'),
            'Jitter': ('jitter', f'Jitter ({self.latency_unit})', '#F18F01')
        }
        
        if metric not in metric_map:
            return
        
        col, label, color = metric_map[metric]
        data = self.df[col].dropna()
        
        if len(data) == 0:
            messagebox.showerror("Error", f"No valid data for {metric}")
            return
        
        # Apply outlier filter
        filtered_data = self.apply_outlier_filter(data)
        
        if len(filtered_data) == 0:
            messagebox.showwarning("Warning", "All data filtered out. Try a less aggressive filter.")
            return
        
        # Calculate statistics
        stats = self.calculate_statistics(filtered_data)
        
        # Update statistics display
        self.stats_text.config(state='normal')
        self.stats_text.delete('1.0', tk.END)
        
        stats_str = f"{metric} Statistics\n{'='*28}\n\n"
        stats_str += f"Count:     {stats['Count']:>10}\n"
        stats_str += f"Min:       {stats['Min']:>10.3f}\n"
        stats_str += f"Max:       {stats['Max']:>10.3f}\n"
        stats_str += f"Mean:      {stats['Mean']:>10.3f}\n"
        stats_str += f"Median:    {stats['Median']:>10.3f}\n"
        stats_str += f"Std Dev:   {stats['Std Dev']:>10.3f}\n"
        stats_str += f"P90:       {stats['P90']:>10.3f}\n"
        stats_str += f"P95:       {stats['P95']:>10.3f}\n"
        stats_str += f"P99:       {stats['P99']:>10.3f}\n"
        
        if len(filtered_data) < len(data):
            filtered_pct = (1 - len(filtered_data)/len(data)) * 100
            stats_str += f"\n{filtered_pct:.1f}% of data filtered"
        
        self.stats_text.insert('1.0', stats_str)
        self.stats_text.config(state='disabled')
        
        # Create new plot
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        # Plot data
        x_values = range(len(filtered_data))
        self.ax.plot(x_values, filtered_data.values, color=color, linewidth=1.5, 
                    alpha=0.8, label=metric)
        
        # Add mean line
        mean_line = stats['Mean']
        self.ax.axhline(y=mean_line, color='red', linestyle='--', linewidth=1, 
                       alpha=0.7, label=f'Mean: {mean_line:.2f}')
        
        # Styling
        self.ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
        self.ax.set_ylabel(label, fontsize=11, fontweight='bold')
        self.ax.set_title(f'{metric} Analysis', fontsize=13, fontweight='bold', pad=15)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(loc='best', framealpha=0.9)
        
        # Tight layout
        self.fig.tight_layout()
        
        # Update canvas
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = PerformanceLogAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
