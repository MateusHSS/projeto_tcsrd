import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_bar_chart(df):
    mlp_mean = df['mlp_steps'].dropna().mean()
    gat_mean = df['gat_steps'].dropna().mean()
    rand_mean = df['random_steps_avg'].dropna().mean()

    labels = ['Ataque Aleatório\n(Baseline)', 'Agente MLP\n(Sem Topologia)', 'Agente GAT + PPO\n(Nosso Framework)']
    values = [rand_mean, mlp_mean, gat_mean]
    colors = ['#7f7f7f', '#ff7f0e', '#2ca02c'] 

    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(9, 6))
    
    bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5, width=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                 f'{height:.1f} ataques',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('Comparativo de Eficiência (Menos ataques para quebrar = Melhor)', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Passos Médios até Colapso (Steps-to-Failure)', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=10)
    plt.ylim(0, max(values) * 1.2)
    
    plt.tight_layout()
    output_file = "grafico_comparacao_pdr.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráfico de Barras salvo: {output_file}")

def plot_boxplot(df):
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid", context="paper")
    
    # Preparando dados para o seaborn
    plot_data = pd.DataFrame({
        'Agente': ['Aleatório (Baseline)'] * len(df) + ['MLP (Sem Topologia)'] * len(df) + ['GAT + PPO (Framework)'] * len(df),
        'Passos': pd.concat([df['random_steps_avg'], df['mlp_steps'], df['gat_steps']], ignore_index=True)
    })
    
    sns.boxplot(x='Agente', y='Passos', data=plot_data, palette=['#7f7f7f', '#ff7f0e', '#2ca02c'], width=0.5)
    
    plt.title('Distribuição e Estabilidade dos Ataques (Boxplot)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Passos necessários para quebrar a rede', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = "grafico_boxplot_estabilidade.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráfico Boxplot salvo: {output_file}")

def plot_survival_curve(df):
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid", context="paper")
    
    max_steps = int(max(df['random_steps_avg'].max(), df['mlp_steps'].max(), df['gat_steps'].max()))
    x_steps = np.arange(1, max_steps + 2)
    
    # Calcula quantos % das redes sobrevivem em cada passo
    def calc_survival(series):
        surv = []
        for s in x_steps:
            surv.append(sum(series >= s) / len(series) * 100.0)
        return surv
        
    y_rand = calc_survival(df['random_steps_avg'])
    y_mlp = calc_survival(df['mlp_steps'])
    y_gat = calc_survival(df['gat_steps'])
    
    plt.plot(x_steps, y_rand, label='Aleatório (Baseline)', color='#7f7f7f', linewidth=2.5, linestyle=':')
    plt.plot(x_steps, y_mlp, label='MLP (Sem Topologia)', color='#ff7f0e', linewidth=2.5, linestyle='--')
    plt.plot(x_steps, y_gat, label='GAT + PPO (Nosso Framework)', color='#2ca02c', linewidth=3)
    
    # Preenchimento embaixo da curva da GAT para dar ênfase
    plt.fill_between(x_steps, y_gat, alpha=0.1, color='#2ca02c')
    
    plt.title('Curva de Sobrevivência da Rede sob Ataque (Kaplan-Meier)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Número de Roteadores Derrubados (Passos)', fontsize=12, fontweight='bold')
    plt.ylabel('Probabilidade da Rede Sobreviver (%)', fontsize=12, fontweight='bold')
    plt.xlim(1, max_steps)
    plt.ylim(0, 105)
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    output_file = "grafico_sobrevivencia_ataque.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráfico de Sobrevivência salvo: {output_file}")

def plot_benchmark_results(csv_path="benchmark_results.csv"):
    if not os.path.exists(csv_path):
        print(f"[ERRO] Arquivo '{csv_path}' não encontrado.")
        return

    try:
        df = pd.read_csv(csv_path)        
        plot_bar_chart(df)
        plot_boxplot(df)
        plot_survival_curve(df)
        
    except Exception as e:
        print(f"[ERRO] Falha ao processar {csv_path}: {e}")

if __name__ == "__main__":
    plot_benchmark_results()
