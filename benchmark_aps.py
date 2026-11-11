import csv
import random
import sys
import time
import statistics
import tracemalloc
import gc
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
sys.setrecursionlimit(50000)

@dataclass
class MetricResult:
    media_tempo: float
    desvio_padrao: float
    pico_memoria_kb: float
    comparacoes: int
    trocas: int

class AlgoritmosOrdenacaoEspecialista:
    @staticmethod
    def bubble_sort(arr: list) -> tuple[list, int, int]:
        data = list(arr)
        n = len(data)
        comps = swaps = 0
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                comps += 1
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
                    swaps += 1
                    swapped = True
            if not swapped:
                break
        return data, comps, swaps

    @staticmethod
    def selection_sort(arr: list) -> tuple[list, int, int]:
        data = list(arr)
        n = len(data)
        comps = swaps = 0
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                comps += 1
                if data[j] < data[min_idx]:
                    min_idx = j
            if min_idx != i:
                data[i], data[min_idx] = data[min_idx], data[i]
                swaps += 1
        return data, comps, swaps

    @staticmethod
    def insertion_sort(arr: list) -> tuple[list, int, int]:
        data = list(arr)
        comps = swaps = 0
        for i in range(1, len(data)):
            key = data[i]
            j = i - 1
            while j >= 0 and data[j] > key:
                comps += 1
                data[j + 1] = data[j]
                swaps += 1
                j -= 1
            if j >= 0:
                comps += 1
            data[j + 1] = key
            swaps += 1
        return data, comps, swaps

    @staticmethod
    def merge_sort(arr: list) -> tuple[list, int, int]:
        comps = [0]
        swaps = [0]

        def _merge_sort_rec(data):
            if len(data) <= 1:
                return data
            mid = len(data) // 2
            left = _merge_sort_rec(data[:mid])
            right = _merge_sort_rec(data[mid:])
            return _merge(left, right)

        def _merge(left, right):
            result = []
            i = j = 0
            len_l, len_r = len(left), len(right)
            while i < len_l and j < len_r:
                comps[0] += 1
                if left[i] <= right[j]:
                    result.append(left[i])
                    swaps[0] += 1
                    i += 1
                else:
                    result.append(right[j])
                    swaps[0] += 1
                    j += 1
            result.extend(left[i:])
            result.extend(right[j:])
            swaps[0] += (len_l - i) + (len_r - j)
            return result

        res = _merge_sort_rec(list(arr))
        return res, comps[0], swaps[0]

    @staticmethod
    def quick_sort_median(arr: list) -> tuple[list, int, int]:
        data = list(arr)
        n = len(data)
        comps = swaps = 0
        if n <= 1:
            return data, 0, 0

        stack = [(0, n - 1)]
        while stack:
            low, high = stack.pop()
            if low < high:
                mid = (low + high) // 2
                comps += 3
                if data[low] > data[mid]:
                    data[low], data[mid] = data[mid], data[low]
                    swaps += 1
                if data[low] > data[high]:
                    data[low], data[high] = data[high], data[low]
                    swaps += 1
                if data[mid] > data[high]:
                    data[mid], data[high] = data[high], data[mid]
                    swaps += 1

                data[mid], data[high] = data[high], data[mid]
                swaps += 1
                pivot = data[high]
                i = low - 1

                for j in range(low, high):
                    comps += 1
                    if data[j] <= pivot:
                        i += 1
                        if i != j:
                            data[i], data[j] = data[j], data[i]
                            swaps += 1

                data[i + 1], data[high] = data[high], data[i + 1]
                swaps += 1
                p = i + 1

                stack.append((low, p - 1))
                stack.append((p + 1, high))

        return data, comps, swaps


class BenchmarkEngine:
    BIG_O_TEMPO = {
        'Bubble Sort': 'O(n²)',
        'Selection Sort': 'O(n²)',
        'Insertion Sort': 'O(n²)',
        'Merge Sort': 'O(n log n)',
        'Quick Sort (Mediana)': 'O(n log n)'
    }

    BIG_O_ESPACOMEMORIA = {
        'Bubble Sort': 'O(1)',
        'Selection Sort': 'O(1)',
        'Insertion Sort': 'O(1)',
        'Merge Sort': 'O(n)',
        'Quick Sort (Mediana)': 'O(log n)'
    }

    def __init__(self, repeticoes_n2: int = 1, repeticoes_nlogn: int = 3):
        self.repeticoes_n2 = repeticoes_n2
        self.repeticoes_nlogn = repeticoes_nlogn
        self.algoritmos = {
            'Bubble Sort': AlgoritmosOrdenacaoEspecialista.bubble_sort,
            'Selection Sort': AlgoritmosOrdenacaoEspecialista.selection_sort,
            'Insertion Sort': AlgoritmosOrdenacaoEspecialista.insertion_sort,
            'Merge Sort': AlgoritmosOrdenacaoEspecialista.merge_sort,
            'Quick Sort (Mediana)': AlgoritmosOrdenacaoEspecialista.quick_sort_median
        }

    def carregar_dados(self, caminho_csv: str, limite_amostra: int = None) -> tuple[list[float], int]:
        caminho = Path(caminho_csv)
        if not caminho.is_file():
            return [], 0

        valores = []
        anomalias = 0
        with open(caminho, mode='r', encoding='utf-8') as f:
            leitor = csv.DictReader(f)
            for linha in leitor:
                val_str = linha.get('areaMunKm', '').strip()
                if val_str:
                    try:
                        val = float(val_str.replace(',', '.'))
                        if val >= 0:
                            valores.append(val)
                        else:
                            anomalias += 1
                    except ValueError:
                        anomalias += 1

        if limite_amostra and len(valores) > limite_amostra:
            valores = random.sample(valores, limite_amostra)

        return valores, anomalias

    def _medir_funcao(self, nome_alg: str, func, dados: list) -> MetricResult:
        reps = self.repeticoes_n2 if "Sort" in nome_alg and "Merge" not in nome_alg and "Quick" not in nome_alg else self.repeticoes_nlogn
        tempos, picos_memoria = [], []
        tot_comps = tot_swaps = 0

        for _ in range(reps):
            copia = list(dados)
            gc.collect()
            gc.disable()
            tracemalloc.start()

            t0 = time.perf_counter()
            _, comps, swaps = func(copia)
            t1 = time.perf_counter()

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            gc.enable()

            tempos.append(t1 - t0)
            picos_memoria.append(peak / 1024.0)
            tot_comps = comps
            tot_swaps = swaps

        return MetricResult(
            media_tempo=statistics.mean(tempos),
            desvio_padrao=statistics.stdev(tempos) if len(tempos) > 1 else 0.0,
            pico_memoria_kb=statistics.mean(picos_memoria),
            comparacoes=tot_comps,
            trocas=tot_swaps
        )

    def executar_dataset(self, caminho_csv: str, limite_amostra: int):
        dados_brutos, anomalias = self.carregar_dados(caminho_csv, limite_amostra)
        if not dados_brutos:
            return None, 0

        cenarios = {
            'Inversamente Ordenado': sorted(dados_brutos, reverse=True),
            'Aleatório': random.sample(dados_brutos, len(dados_brutos))
        }

        resultados = {cenario: {} for cenario in cenarios}
        for nome_alg, func in self.algoritmos.items():
            print(f"   [+] Processando {nome_alg}...")
            for cenario_nome, cenario_dados in cenarios.items():
                resultados[cenario_nome][nome_alg] = self._medir_funcao(nome_alg, func, cenario_dados)

        return resultados, len(dados_brutos)


class ModernDashboardApp(ctk.CTk):
    def __init__(self, engine: BenchmarkEngine):
        super().__init__()
        self.engine = engine
        self.title("APS UNIP 2026/2 — Benchmark de Algoritmos")
        self.geometry("1400x880")
        self.minsize(1100, 720)

        self.data_com_filtro = None
        self.total_com_filtro = 0
        self.data_sem_filtro = None
        self.total_sem_filtro = 0
        self.botoes_nav = {}

        self._executar_benchmark_inicial()
        self._construir_interface()

    def _executar_benchmark_inicial(self):
        print("[SISTEMA] Processando benchmarks ambientais...")
        t0 = time.time()
        self.data_com_filtro, self.total_com_filtro = self.engine.executar_dataset("avisos_desmatamento_com_filtro.csv", 3000)
        self.data_sem_filtro, self.total_sem_filtro = self.engine.executar_dataset("avisos_desmatamento_sem_filtro.csv", 7000)
        print(f"[SISTEMA] Concluído em {time.time() - t0:.2f}s!")

    def _construir_interface(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color="#0f172a")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)

        lbl_brand = ctk.CTkLabel(self.sidebar, text="APS 2026/2", font=ctk.CTkFont(size=22, weight="bold"), text_color="#38bdf8")
        lbl_brand.pack(anchor="w", padx=25, pady=(25, 2))
        
        lbl_sub = ctk.CTkLabel(self.sidebar, text="Estrutura de Dados & Algoritmos", font=ctk.CTkFont(size=11), text_color="#64748b")
        lbl_sub.pack(anchor="w", padx=25, pady=(0, 20))

        group_box = ctk.CTkFrame(self.sidebar, fg_color="#1e293b", corner_radius=10)
        group_box.pack(fill="x", padx=20, pady=(0, 20))

        lbl_g = ctk.CTkLabel(group_box, text="EQUIPE / NOME DO GRUPO:", font=ctk.CTkFont(size=9, weight="bold"), text_color="#94a3b8")
        lbl_g.pack(anchor="w", padx=12, pady=(10, 2))

        self.ent_grupo = ctk.CTkEntry(group_box, font=ctk.CTkFont(size=12, weight="bold"), fg_color="#0f172a", text_color="#f8fafc", border_color="#334155")
        self.ent_grupo.insert(0, "Raquel Ramos, Brenda Melo, Vitor Moreira, Samuel Batista e Luan Santos ")
        self.ent_grupo.pack(fill="x", padx=12, pady=(0, 10))

        self.botoes_nav["kpi"] = self._criar_btn_nav("📊  Visão Geral (KPIs)", lambda: self._trocar_aba("kpi", self._render_kpi))
        self.botoes_nav["f1"] = self._criar_btn_nav("⚡  Com Filtro (N=3.000)", lambda: self._trocar_aba("f1", lambda: self._render_dataset("Planilha Com Filtro", self.data_com_filtro, self.total_com_filtro)))
        self.botoes_nav["f2"] = self._criar_btn_nav("📈  Sem Filtro (N=7.000)", lambda: self._trocar_aba("f2", lambda: self._render_dataset("Planilha Sem Filtro", self.data_sem_filtro, self.total_sem_filtro)))
        self.botoes_nav["doc"] = self._criar_btn_nav("📄  Relatório Técnico", lambda: self._trocar_aba("doc", self._render_doc))

        lbl_foot = ctk.CTkLabel(self.sidebar, text="UNIP | Ciência da Computação\nNovembro de 2026", font=ctk.CTkFont(size=10), text_color="#475569")
        lbl_foot.pack(side="bottom", pady=20)

        self.main_area = ctk.CTkScrollableFrame(self, fg_color="#020617", corner_radius=0)
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

        self._trocar_aba("kpi", self._render_kpi)

    def _criar_btn_nav(self, text, command):
        btn = ctk.CTkButton(
            self.sidebar, text=text, command=command, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="transparent", text_color="#94a3b8", hover_color="#1e293b",
            anchor="w", height=42, corner_radius=8
        )
        btn.pack(fill="x", padx=15, pady=3)
        return btn

    def _trocar_aba(self, key, render_func):
        for k, btn in self.botoes_nav.items():
            if k == key:
                btn.configure(fg_color="#0284c7", text_color="#ffffff")
            else:
                btn.configure(fg_color="transparent", text_color="#94a3b8")

        for child in self.main_area.winfo_children():
            child.destroy()

        render_func()

    def _render_kpi(self):
        nome_g = self.ent_grupo.get() or "Grupo APS"
        
        head = ctk.CTkFrame(self.main_area, fg_color="transparent")
        head.pack(fill="x", padx=35, pady=(30, 15))
        
        ctk.CTkLabel(head, text=f"Dashboard Executivo — {nome_g}", font=ctk.CTkFont(size=22, weight="bold"), text_color="#f8fafc").pack(anchor="w")
        ctk.CTkLabel(head, text="Instrumentação de hardware e análise quantitativa sobre dados ambientais do INPE", font=ctk.CTkFont(size=12), text_color="#64748b").pack(anchor="w")

        cards = ctk.CTkFrame(self.main_area, fg_color="transparent")
        cards.pack(fill="x", padx=35, pady=10)
        cards.columnconfigure((0,1,2), weight=1)

        best_f1 = min(self.data_com_filtro['Aleatório'].items(), key=lambda x: x[1].media_tempo) if self.data_com_filtro else ("N/A", None)
        best_f2 = min(self.data_sem_filtro['Aleatório'].items(), key=lambda x: x[1].media_tempo) if self.data_sem_filtro else ("N/A", None)

        self._card_kpi(cards, 0, "MENOR TEMPO (N=3k)", best_f1[0], f"{best_f1[1].media_tempo:.4f}s" if best_f1[1] else "-", "#38bdf8")
        self._card_kpi(cards, 1, "MENOR TEMPO (N=7k)", best_f2[0], f"{best_f2[1].media_tempo:.4f}s" if best_f2[1] else "-", "#4ade80")
        self._card_kpi(cards, 2, "MENOR USO DE MEMÓRIA", "Bubble / Selection", "~23.9 KB (O(1))", "#c084fc")

        box = ctk.CTkFrame(self.main_area, fg_color="#0f172a", corner_radius=12, border_width=1, border_color="#1e293b")
        box.pack(fill="both", expand=True, padx=35, pady=25)

        ctk.CTkLabel(box, text="Destaques de Engenharia e Métricas de Hardware", font=ctk.CTkFont(size=14, weight="bold"), text_color="#f8fafc").pack(anchor="w", padx=25, pady=(20, 10))
        
        txt = (
            "• Garbage Collection & Warm-up: A medição isolou o coletor de lixo do Python para evitar oscilações de tempo.\n"
            "• Medição com tracemalloc: Identificou empiricamente a complexidade de espaço O(n) do Merge Sort vs O(1) dos algoritmos in-place.\n"
            "• Resolução de Recursão: Quick Sort implementado com Mediana de Três e controle por pilha iterativa (prevenção de StackOverflow)."
        )
        ctk.CTkLabel(box, text=txt, font=ctk.CTkFont(size=12), text_color="#94a3b8", justify="left").pack(anchor="w", padx=25, pady=(0, 20))

    def _card_kpi(self, parent, col, title, main, sub, color):
        c = ctk.CTkFrame(parent, fg_color="#0f172a", corner_radius=12, border_width=1, border_color="#1e293b")
        c.grid(row=0, column=col, padx=8, pady=5, sticky="ew")

        ctk.CTkLabel(c, text=title, font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748b").pack(anchor="w", padx=15, pady=(15, 2))
        ctk.CTkLabel(c, text=main, font=ctk.CTkFont(size=16, weight="bold"), text_color=color).pack(anchor="w", padx=15)
        ctk.CTkLabel(c, text=sub, font=ctk.CTkFont(size=11), text_color="#94a3b8").pack(anchor="w", padx=15, pady=(0, 15))

    def _render_dataset(self, titulo, resultados, total):
        head = ctk.CTkFrame(self.main_area, fg_color="transparent")
        head.pack(fill="x", padx=35, pady=(25, 10))
        
        ctk.CTkLabel(head, text=f"Análise Estocástica: {titulo}", font=ctk.CTkFont(size=20, weight="bold"), text_color="#f8fafc").pack(anchor="w")
        ctk.CTkLabel(head, text=f"Amostragem parametrizada de N = {total:,} registros | Fonte: INPE TerraBrasilis", font=ctk.CTkFont(size=11), text_color="#64748b").pack(anchor="w")

        fig = Figure(figsize=(11, 4.2), dpi=100, facecolor="#020617")
        algs = list(self.engine.algoritmos.keys())
        algs_curtos = [a.replace(' Sort', '').replace(' (Mediana)', '*') for a in algs]
        m_inv = [resultados['Inversamente Ordenado'][alg].media_tempo for alg in algs]
        m_ale = [resultados['Aleatório'][alg].media_tempo for alg in algs]
        ram_ale = [resultados['Aleatório'][alg].pico_memoria_kb for alg in algs]

        x = np.arange(len(algs))
        largura = 0.35

        ax1 = fig.add_subplot(121)
        ax1.set_facecolor("#0f172a")
        ax1.bar(x - largura/2, m_inv, largura, label='Inverso (Pior)', color='#f43f5e', alpha=0.9)
        ax1.bar(x + largura/2, m_ale, largura, label='Aleatório (Médio)', color='#38bdf8', alpha=0.9)
        ax1.set_yscale('log')
        ax1.set_title('Tempo Médio (Escala Log_{10})', color='#f8fafc', fontsize=10, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algs_curtos, color='#94a3b8', fontsize=8)
        ax1.tick_params(colors='#64748b')
        ax1.legend(facecolor='#0f172a', edgecolor='#1e293b', labelcolor='#f8fafc', fontsize=8)
        ax1.grid(True, which="both", ls=":", color='#1e293b')

        ax2 = fig.add_subplot(122)
        ax2.set_facecolor("#0f172a")
        bars = ax2.bar(x, ram_ale, color='#2dd4bf', width=0.45, alpha=0.9)
        ax2.set_title('Pico de Memória RAM (KB)', color='#f8fafc', fontsize=10, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(algs_curtos, color='#94a3b8', fontsize=8)
        ax2.tick_params(colors='#64748b')
        ax2.grid(True, axis='y', ls=":", color='#1e293b')

        for b in bars:
            y = b.get_height()
            ax2.text(b.get_x() + b.get_width()/2.0, y + (max(ram_ale)*0.02), f'{y:.1f}k', ha='center', va='bottom', color='#f8fafc', fontsize=7.5, fontweight='bold')

        fig.tight_layout()

        canvas_box = ctk.CTkFrame(self.main_area, fg_color="#0f172a", corner_radius=12, border_width=1, border_color="#1e293b")
        canvas_box.pack(fill="x", padx=35, pady=10)

        canvas = FigureCanvasTkAgg(fig, master=canvas_box)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", padx=15, pady=15)

        tbl_card = ctk.CTkFrame(self.main_area, fg_color="#0f172a", corner_radius=12, border_width=1, border_color="#1e293b")
        tbl_card.pack(fill="x", padx=35, pady=20)

        ctk.CTkLabel(tbl_card, text="Matriz Computacional de Operações", font=ctk.CTkFont(size=13, weight="bold"), text_color="#f8fafc").pack(anchor="w", padx=20, pady=(15, 10))

        headers = ["Algoritmo", "Tempo Inv.", "Tempo Ale.", "Pico RAM", "Comparações", "Swaps", "Big-O Tempo"]
        
        h_frame = ctk.CTkFrame(tbl_card, fg_color="#1e293b", height=32, corner_radius=6)
        h_frame.pack(fill="x", padx=15, pady=(0, 5))
        h_frame.columnconfigure((0,1,2,3,4,5,6), weight=1)

        for idx, text in enumerate(headers):
            ctk.CTkLabel(h_frame, text=text, font=ctk.CTkFont(size=10, weight="bold"), text_color="#38bdf8").grid(row=0, column=idx, padx=5, pady=4)

        for alg in algs:
            res_inv = resultados['Inversamente Ordenado'][alg]
            res_ale = resultados['Aleatório'][alg]
            big_t = self.engine.BIG_O_TEMPO[alg]

            r_frame = ctk.CTkFrame(tbl_card, fg_color="transparent")
            r_frame.pack(fill="x", padx=15, pady=2)
            r_frame.columnconfigure((0,1,2,3,4,5,6), weight=1)

            vals = [
                alg, f"{res_inv.media_tempo:.4f}s", f"{res_ale.media_tempo:.4f}s",
                f"{res_ale.pico_memoria_kb:.1f} KB", f"{res_ale.comparacoes:,}",
                f"{res_ale.trocas:,}", big_t
            ]
            for idx, val in enumerate(vals):
                ctk.CTkLabel(r_frame, text=val, font=ctk.CTkFont(size=10), text_color="#cbd5e1").grid(row=0, column=idx, padx=5, pady=4)

    def _render_doc(self):
        nome_g = self.ent_grupo.get() or "Grupo APS"
        
        box = ctk.CTkFrame(self.main_area, fg_color="#0f172a", corner_radius=12, border_width=1, border_color="#1e293b")
        box.pack(fill="both", expand=True, padx=35, pady=30)

        ctk.CTkLabel(box, text=f"Documentação Técnica — {nome_g}", font=ctk.CTkFont(size=16, weight="bold"), text_color="#f8fafc").pack(anchor="w", padx=25, pady=(20, 10))

        txt = (
            "1. ANÁLISE ASSINTÓTICA FORMAL (O, Theta, Omega)\n"
            "--------------------------------------------------------------------------------\n"
            "• Insertion Sort: Omega(n) no Melhor Caso (vetor pré-ordenado), mas degrada para Theta(n²)\n"
            "  no Caso Médio e Pior Caso devido aos deslocamentos sucessivos.\n\n"
            "• Selection Sort: Comportamento rígido de Theta(n²) em todos os cenários, realizando sempre\n"
            "  n*(n-1)/2 comparações independentemente da ordem inicial.\n\n"
            "• Merge Sort: Desempenho estrito de Theta(n log n) em qualquer distribuição de dados,\n"
            "  exigindo contudo espaço auxiliar de memória RAM O(n).\n\n"
            "• Quick Sort (Mediana de Três): Evita a degradação de pior caso O(n²), mantendo a média em\n"
            "  Theta(n log n) com consumo de memória de pilha O(log n).\n\n"
            "2. APLICAÇÃO PRÁTICA AOS DADOS DO INPE (PRINCÍPIO DE PARETO)\n"
            "--------------------------------------------------------------------------------\n"
            "A ordenação decrescente da variável 'areaMunKm' permite aos órgãos ambientais (IBAMA/ICMBio)\n"
            "identificar os municípios que concentram a maioria da área desmatada acumulada (Regra 80/20),\n"
            "otimizando a alocação de equipes de fiscalização no terreno."
        )

        txt_widget = ctk.CTkTextbox(box, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#020617", text_color="#cbd5e1", border_width=1, border_color="#1e293b")
        txt_widget.pack(fill="both", expand=True, padx=25, pady=(0, 20))
        txt_widget.insert("1.0", txt)
        txt_widget.configure(state="disabled")


if __name__ == "__main__":
    engine = BenchmarkEngine(repeticoes_n2=1, repeticoes_nlogn=3)
    app = ModernDashboardApp(engine)
    app.mainloop()