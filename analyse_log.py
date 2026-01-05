import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
import logging
from scipy.stats import pearsonr

# --- MILITARY BLACK-AND-WHITE STYLE CONFIGURATION ---
plt.style.use('classic')  # Clean, professional base style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'grid.color': '#D3D3D3',  # Light gray grid
    'grid.linewidth': 0.6,
    'grid.alpha': 0.7,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.labelsize': 10,
    'figure.titlesize': 14,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'savefig.dpi': 300,  # High resolution for reports
    'savefig.format': 'png',
    'savefig.bbox': 'tight'
})

# Monochrome color mapping
PRIMARY_BLACK = '#000000'
SECONDARY_GRAY = '#4A4A4A'
ACCENT_GRAY = '#808080'
LIGHT_GRAY = '#CCCCCC'
REFERENCE_LINE = '#8B8B8B'  # Dashed reference lines

# --- Directory Configuration ---
PLOT_DIR = "plots"
OUTPUT_DIR = "output"
LOG_FILE = "analysis_log.txt"

def setup_environment():
    """Maakt benodigde mappen en configureert logging."""
    for folder in [PLOT_DIR, OUTPUT_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_csv(file_path):
    """Laadt CSV met dynamische header-detectie en fallback."""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline()
        
        has_header = 'module' in first_line.lower()
        
        if has_header:
            df = pd.read_csv(file_path)
            logging.info(f"{file_path} geladen met gedetecteerde headers.")
        else:
            default_headers = [
                "produced_ts_epoch_ms", "call_ts_epoch_ms", "module", "event", "thread_id", 
                "cam_frame_id", "cam_exposure_ms", "cam_isp_latency_ms", "cam_buffer_usage_percent",
                "image_proc_ms", "tpu_inference_ms", "tpu_temp_c", "tpu_model_score", "tpu_class_id",
                "logic_target_dist_m", "logic_ballistic_drop_m", "logic_windage_m", "logic_servo_x_cmd",
                "logic_servo_y_cmd", "logic_solution_time_ms", "enc_process_ms", "enc_bitrate_mbps",
                "enc_queue_depth", "sys_cpu_temp_c", "sys_cpu_usage_pct", "sys_ram_usage_pct", "sys_voltage_v"
            ]
            df = pd.read_csv(file_path, names=default_headers)
            logging.warning("Geen headers gedetecteerd. Standaard telemetry schema toegepast.")
        
        return df
    except Exception as e:
        logging.error(f"CSV laden mislukt: {e}")
        sys.exit(1)

def compute_metrics(df):
    """Berekent latency, throughput en end-to-end metrieken."""
    logging.info("Metrieken berekenen...")
    
    # Systeemstatistieken filteren
    sys_df = df[df['module'] == 'SystemMonitor'].copy()
    
    # Pipeline analyses op frame-ID
    pipe_df = df.dropna(subset=['cam_frame_id']).copy()
    frame_stats = pipe_df.groupby('cam_frame_id').agg({
        'produced_ts_epoch_ms': ['min', 'max', 'count'],
        'tpu_inference_ms': 'max',
        'image_proc_ms': 'max',
        'logic_solution_time_ms': 'max'
    })
    frame_stats.columns = ['ts_start', 'ts_end', 'event_count', 'tpu_ms', 'img_proc_ms', 'logic_ms']
    
    # End-to-end latency: van eerste tot laatste event per frame
    frame_stats['e2e_latency_ms'] = frame_stats['ts_end'] - frame_stats['ts_start']
    
    # Throughput (FPS) gebaseerd op frame-intervallen
    frame_stats = frame_stats.sort_index()
    frame_stats['inter_frame_interval'] = frame_stats['ts_start'].diff()
    fps = 1000 / frame_stats['inter_frame_interval'].mean() if not frame_stats.empty else 0
    
    return frame_stats, sys_df, fps

def compute_stats(series, name):
    """Retourneert dictionary met algemene statistieken."""
    if series.empty: return {}
    return {
        "Metriek": name,
        "Min": series.min(),
        "Max": series.max(),
        "Mediaan": series.median(),
        "P95": series.quantile(0.95),
        "P99": series.quantile(0.99),
        "StdDev": series.std()
    }

def plot_time_series(frame_stats, sys_df):
    """Genereert tijdreeksen voor verwerking en systeemgezondheid."""
    logging.info("Tijdreeksdiagrammen genereren...")
    
    # LATENCY OVER TIJD
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frame_stats.index, frame_stats['e2e_latency_ms'], 
            label='E2E Latency', color=PRIMARY_BLACK, linewidth=1.8, marker='o', 
            markersize=3, markevery=20)
    ax.plot(frame_stats.index, frame_stats['tpu_ms'], 
            label='TPU Inference', color=SECONDARY_GRAY, linewidth=1.4, 
            linestyle='--', marker='s', markersize=2.5, markevery=25)
    
    ax.set_title("PIPELINE LATENCY OVER FRAME ID\nToont de verwerkingstijd per frame voor de volledige pipeline en TPU inferentie", 
                 pad=15)
    ax.set_xlabel("Camera Frame ID")
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='-', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f"{PLOT_DIR}/latency_timeseries.png")
    plt.close()

    # SYSTEEMSTATISTIEKEN
    if not sys_df.empty:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # CPU en RAM op primaire as
        ax1.set_xlabel("Tijd (ms)")
        ax1.set_ylabel("Gebruik (%)")
        ax1.plot(sys_df['produced_ts_epoch_ms'], sys_df['sys_cpu_usage_pct'], 
                label='CPU %', color=PRIMARY_BLACK, linewidth=1.5, marker='^', 
                markersize=3, markevery=30)
        ax1.plot(sys_df['produced_ts_epoch_ms'], sys_df['sys_ram_usage_pct'], 
                label='RAM %', color=SECONDARY_GRAY, linewidth=1.5, linestyle=':', 
                marker='v', markersize=3, markevery=30)
        ax1.tick_params(axis='y')
        ax1.set_ylim(0, 100)

        # Temperatuur op secundaire as
        ax2 = ax1.twinx()
        ax2.set_ylabel("Temperatuur (°C)", color=ACCENT_GRAY)
        ax2.plot(sys_df['produced_ts_epoch_ms'], sys_df['sys_cpu_temp_c'], 
                label='CPU Temp', color=ACCENT_GRAY, linewidth=1.2, linestyle='--', 
                marker='d', markersize=2.5, markevery=25)
        ax2.tick_params(axis='y', labelcolor=ACCENT_GRAY)
        
        # Referentielijnen voor kritieke waarden
        ax1.axhline(y=80, color=REFERENCE_LINE, linestyle='-.', linewidth=0.8, alpha=0.6)
        ax2.axhline(y=75, color=REFERENCE_LINE, linestyle='-.', linewidth=0.8, alpha=0.6)
        
        plt.title("SYSTEEMRESOURCE GEBRUIK\nCPU, RAM en temperatuur gedurende de meetperiode", pad=15)
        
        # Gecombineerde legende
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                  bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
        
        ax1.grid(True, linestyle='-', alpha=0.5)
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        fig.tight_layout()
        plt.savefig(f"{PLOT_DIR}/system_performance.png")
        plt.close()

def plot_histograms(frame_stats):
    """Genereert histogrammen voor latency-verdeling."""
    logging.info("Histogrammen genereren...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(frame_stats['e2e_latency_ms'].dropna(), 
                               bins=30, color=LIGHT_GRAY, edgecolor=PRIMARY_BLACK, 
                               linewidth=0.8, alpha=0.8)
    
    # Color bins darker for tactical visual effect
    for i, patch in enumerate(patches):
        if i % 2 == 0:
            patch.set_facecolor(ACCENT_GRAY)
    
    median_val = frame_stats['e2e_latency_ms'].median()
    ax.axvline(median_val, color=PRIMARY_BLACK, linestyle='--', linewidth=2, 
               label=f'Mediaan: {median_val:.1f} ms')
    
    ax.set_title("END-TO-END LATENCY VERDELING\nVerdeling van de totale verwerkingstijd van capture tot servo-commando", pad=15)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Frequentie")
    ax.legend()
    ax.grid(True, linestyle='-', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f"{PLOT_DIR}/e2e_histogram.png")
    plt.close()

def interactive_correlation(df):
    """Laat gebruiker correlatie analyseren tussen twee kolommen."""
    print("\n--- INTERACTIEVE CORRELATIE ANALYSE ---")
    cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
    print("Beschikbare kolommen:")
    for i, col in enumerate(cols):
        print(f"[{i}] {col}")
    
    try:
        idx1 = int(input("\nSelecteer index voor X-as: "))
        idx2 = int(input("Selecteer index voor Y-as: "))
        
        col1, col2 = cols[idx1], cols[idx2]
        temp_df = df[[col1, col2]].dropna()
        corr, _ = pearsonr(temp_df[col1], temp_df[col2])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=col1, y=col2, data=temp_df, scatter_kws={'alpha':0.6, 'color':SECONDARY_GRAY, 's':20}, 
                   line_kws={'color':PRIMARY_BLACK, 'linewidth':1.5}, ax=ax)
        
        ax.set_title(f"CORRELATIE: {col1} vs {col2}\nPearson R: {corr:.4f} - Relatie tussen twee geselecteerde variabelen met regressielijn", 
                     pad=15)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.grid(True, linestyle='-', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        filename = f"corr_{col1}_vs_{col2}.png"
        plt.savefig(f"{PLOT_DIR}/{filename}")
        print(f"Diagram opgeslagen in {PLOT_DIR}/{filename}")
        print(f"Pearson Correlatiecoëfficiënt: {corr:.4f}")
        plt.show()
        
    except Exception as e:
        print(f"Ongeldige invoer of fout: {e}")

def main():
    parser = argparse.ArgumentParser(description="CoralEdgeTpu Telemetry Analyse")
    parser.add_argument("csv", help="Pad naar het telemetry CSV-bestand")
    parser.add_argument("json", nargs='?', help="Optionele run-configuratie JSON")
    args = parser.parse_args()

    setup_environment()
    df = load_csv(args.csv)
    
    # Berekeningen uitvoeren
    frame_stats, sys_df, fps = compute_metrics(df)
    
    # Statistiek samenvatting genereren
    stats_list = []
    stats_list.append(compute_stats(frame_stats['e2e_latency_ms'], "E2E_Latency_ms"))
    stats_list.append(compute_stats(frame_stats['tpu_ms'], "TPU_Inference_ms"))
    stats_list.append(compute_stats(frame_stats['img_proc_ms'], "Image_Proc_ms"))
    
    if not sys_df.empty:
        stats_list.append(compute_stats(sys_df['sys_cpu_usage_pct'], "CPU_Gebruik_Pct"))
        stats_list.append(compute_stats(sys_df['sys_cpu_temp_c'], "CPU_Temp_C"))

    stats_df = pd.DataFrame(stats_list)
    
    # Resultaten weergeven
    print("\n" + "="*40)
    print("RUN STATISTIEKEN SAMENVATTING")
    print("="*40)
    print(f"Gemiddelde Doorvoer: {fps:.2f} FPS")
    print(stats_df.to_string(index=False))
    print("="*40)
    
    stats_df.to_csv(f"{OUTPUT_DIR}/summary_stats.csv", index=False)
    
    # Diagrammen genereren
    plot_time_series(frame_stats, sys_df)
    plot_histograms(frame_stats)
    
    logging.info(f"Analyse voltooid. Resultaten opgeslagen in /{PLOT_DIR} en /{OUTPUT_DIR}")

    # Interactieve correlatie loop
    while True:
        cont = input("\nWilt u een correlatie-analyse uitvoeren? (j/n): ").lower()
        if cont == 'j':
            interactive_correlation(df)
        else:
            break

if __name__ == "__main__":
    main()
