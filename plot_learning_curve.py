import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tb_data(log_dir, tag, step_offset=0):
    """Procura o log mais recente e extrai os dados, aplicando um offset opcional no eixo X."""
    if not os.path.exists(log_dir):
        return None
    
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        return None
    
    latest_event = max(event_files, key=os.path.getmtime)
    
    try:
        ea = EventAccumulator(latest_event, size_guidance={'scalars': 0})
        ea.Reload()
        
        if tag not in ea.Tags()['scalars']:
            return None
            
        events = ea.Scalars(tag)
        # Aplica o deslocamento (offset) para continuar a linha do tempo do NetworkX
        df = pd.DataFrame([{'Wall time': e.wall_time, 'Step': e.step + step_offset, 'Value': e.value} for e in events])
        return df
    except Exception as e:
        print(f"Erro ao ler {latest_event}: {e}")
        return None

def extract_combined_data(log_nx, log_ns3, tag, initial_value):
    """Junta o treino do NetworkX com o fine-tuning do NS-3 em uma única linha do tempo."""
    df_nx = extract_tb_data(log_nx, tag)
    
    if df_nx is None or df_nx.empty:
        return None
        
    df_zero = pd.DataFrame({'Wall time': [df_nx['Wall time'].iloc[0] - 100], 'Step': [0], 'Value': [initial_value]})
    df_nx = pd.concat([df_zero, df_nx], ignore_index=True)
    
    # Descobre onde o NetworkX parou (ex: passo 200.000)
    max_step_nx = df_nx['Step'].max()
    
    df_ns3 = extract_tb_data(log_ns3, tag, step_offset=max_step_nx)
    
    if df_ns3 is not None and not df_ns3.empty:
        df_combined = pd.concat([df_nx, df_ns3], ignore_index=True)
        # Marca onde ocorreu a transferência para o gráfico
        transfer_point = max_step_nx
    else:
        df_combined = df_nx
        transfer_point = None
        
    df_combined['Smoothed'] = df_combined['Value'].ewm(alpha=0.3, adjust=False).mean()
    return df_combined, transfer_point

def plot_comparison(tag, y_label, title, output_file, initial_value):    
    gat_data = extract_combined_data("./escala_50_ppo_gat/", "./escala_50_ppo_gat_ns3/", tag, initial_value)
    mlp_data = extract_combined_data("./escala_50_ppo_mlp/", "./escala_50_ppo_mlp_ns3/", tag, initial_value)
    
    if gat_data is None and mlp_data is None:
        print(f"[ERRO] Nenhum dado encontrado para {tag}.")
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", context="paper")
    
    transfer_line_drawn = False

    if gat_data is not None:
        df_gat, tp_gat = gat_data
        sns.lineplot(data=df_gat, x="Step", y="Value", color="#2ca02c", alpha=0.15, linewidth=1)
        sns.lineplot(data=df_gat, x="Step", y="Smoothed", color="#2ca02c", linewidth=2.5, label="GAT + PPO")
        if tp_gat and not transfer_line_drawn:
            plt.axvline(x=tp_gat, color='red', linestyle='--', alpha=0.7, label='Transferência (Início NS-3)')
            transfer_line_drawn = True

    if mlp_data is not None:
        df_mlp, tp_mlp = mlp_data
        sns.lineplot(data=df_mlp, x="Step", y="Value", color="#ff7f0e", alpha=0.15, linewidth=1)
        sns.lineplot(data=df_mlp, x="Step", y="Smoothed", color="#ff7f0e", linewidth=2.5, label="MLP (Baseline)")
        if tp_mlp and not transfer_line_drawn:
            plt.axvline(x=tp_mlp, color='red', linestyle='--', alpha=0.7, label='Transferência (Início NS-3)')
            transfer_line_drawn = True

    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Passos Totais de Treinamento (NetworkX -> NS-3)", fontsize=12, fontweight='bold')
    plt.ylabel(y_label, fontsize=12, fontweight='bold')
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Gráfico salvo: {output_file}")
    plt.close()

if __name__ == "__main__":
    plot_comparison(
        tag="rollout/ep_rew_mean",
        y_label="Recompensa Média", 
        title="Evolução Contínua da Recompensa (Transfer Learning)", 
        output_file="grafico_recompensa_comparativo.png",
        initial_value=-170.8
    )
    
    plot_comparison(
        tag="rollout/ep_len_mean",
        y_label="Passos até a Violação", 
        title="Eficiência Contínua do Ataque (Transfer Learning)", 
        output_file="grafico_eficiencia_comparativo.png",
        initial_value=35.0
    )
