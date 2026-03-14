import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import networkx as nx
import math
import ast
import re

# --- STYLES & CONFIGURATION ---
COLORS = {
    'bg': '#F4F7F6',
    'card': '#FFFFFF',
    'text': '#2C3E50',
    'accent': '#2980B9',     
    'success': '#27AE60',    
    'fail': '#C0392B',       
    'edge_inactive': '#BDC3C7',
    'step_box': '#ECF0F1'
}

PROBE_COLORS = [
    '#2980B9', '#8E44AD', '#D35400', '#16A085', '#C0392B', 
    '#F39C12', '#34495E', '#27AE60', '#E74C3C', '#9B59B6'
]

# --- DOCUMENTATION ---
ALGO_DOCS = {
    'grille': dcc.Markdown(r'''
        #### Algorithme 2 : Grille Hiérarchique
        **Stratégie :** Division par Axes et Groupes.
        1. Validation des axes de référence (Col 0, Ligne 0).
        2. Test global des lignes entières.
        3. Test vertical simultané (échelle) de la k-ième arête de toutes les lignes.
        4. Répétition pour les colonnes.
        * **Mode LTP :** Fusionne ces étapes avec une matrice de test combinatoire. Le cheminement garantit qu'aucune arête n'est traversée deux fois dans le même sens (Routage Eulérien).
    '''),
    'lineaire': dcc.Markdown(r'''
        #### Algorithme 1 : Fenêtre Glissante
        **Stratégie :** Recouvrement Progressif.
        La borne théorique est linéaire (W ≈ N/2). Le mode LTP et Naïf sont ici similaires.
    ''', mathjax=True),
    'complet': dcc.Markdown(r'''
        #### Algorithme 3 : Hub & Spoke
        **Stratégie :** Centre vers Périphérie.
        1. **Phase Étoile :** Validation des liens du Hub.
        2. **Phase Cycle :** Validation des liens distants.
    ''', mathjax=True),
    'arbre': dcc.Markdown(r'''
        #### Algorithme 4 : Profondeur
        **Stratégie :** Racine vers Feuilles.
    ''', mathjax=True)
}

# --- OUTILS GRAPHIQUES ET MATRICE ---
def _distance_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _normalize_edge(u, v):
    return tuple(sorted((u, v), key=str))

def _get_highlighted_edges(probe, topo):
    path = probe['path']
    step_id = probe['step_id']
    description = probe['description']
    highlighted_edges = []

    def add_edge(u, v):
        edge = _normalize_edge(u, v)
        if edge not in highlighted_edges:
            highlighted_edges.append(edge)

    if topo == 'complet':
        if step_id == 'STAR' and len(path) >= 2: add_edge(path[0], path[1])
        if step_id == 'CYCLE' and len(path) >= 3: add_edge(path[1], path[2])
        return highlighted_edges

    if topo == 'arbre' or topo == 'lineaire':
        outward_len = (len(path) + 1) // 2
        for i in range(max(0, outward_len - 1)): add_edge(path[i], path[i + 1])
        return highlighted_edges

    if topo == 'grille':
        s = max(max(u[0], u[1]) for u in path) + 1
        subset = []
        if 'ltp_meta' in probe and probe['ltp_meta']:
            meta = probe['ltp_meta']
            row = meta['matrix'][meta['active_test']]
            subset = [meta['items'][i] for i, val in enumerate(row) if val == 1]

        if step_id == '1a':
            for r in range(s-1): add_edge((r,0), (r+1,0))
        elif step_id.startswith('1b'):
            if subset:
                for r in subset: add_edge((r,0), (r+1,0))
            else:
                match = re.search(r'indice\s+(\d+)', description)
                if match: add_edge((int(match.group(1)),0), (int(match.group(1))+1,0))
        elif step_id == '2a':
            for c in range(s-1): add_edge((0,c), (0,c+1))
        elif step_id.startswith('2b'):
            if subset:
                for c in subset: add_edge((0,c), (0,c+1))
            else:
                match = re.search(r'indice\s+(\d+)', description)
                if match: add_edge((0,int(match.group(1))), (0,int(match.group(1))+1))
        elif step_id.startswith('3a'):
            if subset:
                for r in subset:
                    for c in range(s-1): add_edge((r,c), (r,c+1))
            else:
                match = re.search(r'Ligne\s+(\d+)', description)
                if match:
                    for c in range(s-1): add_edge((int(match.group(1)),c), (int(match.group(1)),c+1))
        elif step_id.startswith('3b'):
            if subset:
                for k in subset:
                    for r in range(1, s): add_edge((r,k), (r,k+1))
            else:
                match = re.search(r'indice\s+(\d+)', description)
                if match:
                    for r in range(1, s): add_edge((r,int(match.group(1))), (r,int(match.group(1))+1))
        elif step_id.startswith('4a'):
            if subset:
                for c in subset:
                    for r in range(s-1): add_edge((r,c), (r+1,c))
            else:
                match = re.search(r'Colonne\s+(\d+)', description)
                if match:
                    for r in range(s-1): add_edge((r,int(match.group(1))), (r+1,int(match.group(1))))
        elif step_id.startswith('4b'):
            if subset:
                for k in subset:
                    for c in range(1, s): add_edge((k,c), (k+1,c))
            else:
                match = re.search(r'indice\s+(\d+)', description)
                if match:
                    for c in range(1, s): add_edge((int(match.group(1)),c), (int(match.group(1))+1,c))
    return highlighted_edges

def _build_probe_step_annotations(path, pos, color):
    if len(path) < 2: return []
    occupied = [(pos[n][0], pos[n][1]) for n in pos]
    placed, annotations = [], []
    edge_lengths = [math.sqrt((pos[path[i+1]][0] - pos[path[i]][0])**2 + (pos[path[i+1]][1] - pos[path[i]][1])**2) for i in range(len(path) - 1)]
    avg_len = sum(edge_lengths) / len(edge_lengths) if edge_lengths else 1.0
    base_offset, min_dist_sq = avg_len * 0.18, (avg_len * 0.18 * 0.85) ** 2

    for idx in range(len(path) - 1):
        x0, y0, x1, y1 = pos[path[idx]][0], pos[path[idx]][1], pos[path[idx + 1]][0], pos[path[idx + 1]][1]
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dx, dy = x1 - x0, y1 - y0
        norm = math.sqrt(dx * dx + dy * dy)
        nxn, nyn = (-dy / norm, dx / norm) if norm != 0 else (0.0, 1.0)
        chosen = None
        for mult in [1.2, -1.2, 2.2, -2.2, 3.5, -3.5]:
            cx, cy = mx + nxn * base_offset * mult, my + nyn * base_offset * mult
            if all(_distance_sq((cx, cy), p) >= min_dist_sq for p in occupied + placed):
                chosen = (cx, cy)
                break
        if chosen is None: chosen = (mx + nxn * base_offset * 4.5, my + nyn * base_offset * 4.5)
        placed.append(chosen)
        annotations.append(dict(x=chosen[0], y=chosen[1], xref="x", yref="y", text=f"<b>{idx + 1}</b>", showarrow=False, font=dict(size=11, color=color), bgcolor="rgba(255, 255, 255, 0.85)", bordercolor=color, borderwidth=1.5, opacity=1.0))
    return annotations

def _build_path_html(path, color):
    path_str = " ➔ ".join([f"Nœud {n}" if isinstance(n, int) else str(n) for n in path])
    return html.Div([
        html.H4("Chemin Optique Emprunté", style={'marginTop': '15px', 'marginBottom': '10px', 'fontSize': '14px', 'color': color}),
        html.Div(path_str, style={'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderLeft': f'5px solid {color}', 'borderRadius': '4px', 'fontFamily': 'monospace', 'fontSize': '13px', 'color': '#2C3E50', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'})
    ])

def _build_matrix_html(ltp_meta):
    if not ltp_meta:
        return html.Div([html.P("Mode itératif ou test global (matrice LTP inactive pour cette étape).", style={'color': '#7F8C8D', 'fontStyle': 'italic', 'fontSize': '12px', 'margin': 0})])
    
    items, matrix, active_idx, phase_name = ltp_meta['items'], ltp_meta['matrix'], ltp_meta['active_test'], ltp_meta['phase']
    headers = [html.Th("Sonde", style={'padding': '5px', 'borderBottom': '1px solid #ddd', 'backgroundColor': '#f8f9fa', 'position': 'sticky', 'left': 0, 'zIndex': 2})]
    for item in items:
        headers.append(html.Th(str(item), style={'padding': '5px', 'borderBottom': '1px solid #ddd', 'fontSize': '10px', 'writingMode': 'vertical-rl', 'transform': 'rotate(180deg)', 'height': '70px', 'textAlign': 'center'}))
    
    rows = []
    for r_idx, row_data in enumerate(matrix):
        is_active = (r_idx == active_idx)
        row_color = PROBE_COLORS[r_idx % len(PROBE_COLORS)]
        bg_style = f"rgba({int(row_color[1:3], 16)}, {int(row_color[3:5], 16)}, {int(row_color[5:7], 16)}, 0.15)" if is_active else 'transparent'
        row_style = {'backgroundColor': bg_style, 'fontWeight': 'bold' if is_active else 'normal'}
        tds = [html.Td(f"T{r_idx + 1} (bit {r_idx})", style={'padding': '5px', 'borderBottom': '1px solid #eee', 'position': 'sticky', 'left': 0, 'backgroundColor': bg_style if is_active else '#f8f9fa', 'zIndex': 1, 'fontSize': '11px'})]
        for val in row_data:
            bg = row_color if (val == 1 and is_active) else (COLORS['accent'] if val == 1 else '#ecf0f1')
            col = 'white' if val == 1 else '#bdc3c7'
            opacity = '1.0' if is_active else '0.4'
            tds.append(html.Td(html.Div(str(val), style={'backgroundColor': bg, 'color': col, 'borderRadius': '3px', 'width': '16px', 'height': '16px', 'lineHeight': '16px', 'margin': 'auto', 'opacity': opacity}), style={'padding': '2px', 'textAlign': 'center', 'borderBottom': '1px solid #eee'}))
        rows.append(html.Tr(tds, style=row_style))

    return html.Div([
        html.H4(f"Matrice LTP : {phase_name}", style={'marginTop': '0', 'marginBottom': '5px', 'fontSize': '13px', 'color': COLORS['text'], 'textTransform': 'uppercase'}),
        html.Div(html.Table([html.Thead(html.Tr(headers))] + [html.Tbody(rows)], style={'borderCollapse': 'collapse', 'width': '100%', 'fontSize': '12px', 'minWidth': 'max-content'}), style={'overflowX': 'auto', 'maxHeight': '200px', 'overflowY': 'auto', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': 'white'})
    ])

def _build_math_details_html(ltp_meta, topo_type):
    if not ltp_meta: return html.Div()
    m = len(ltp_meta['items'])
    if m == 0: return html.Div()
    phase_name = ltp_meta['phase']
    m_explanation = ""
    
    if topo_type == 'complet':
        if "Hub" in phase_name: m_explanation = f"Éléments testés : les $N-1$ liens directs vers le Hub."
        elif "Cycle" in phase_name: m_explanation = f"Éléments testés : les combinaisons entre les nœuds périphériques."
    elif topo_type == 'grille':
        if "Col 0" in phase_name or "Ligne 0" in phase_name: m_explanation = "Éléments testés : les arêtes de l'axe de référence."
        elif "Globales" in phase_name: m_explanation = "Éléments testés : les lignes/colonnes entières (macro-élément)."
        elif "Échelle" in phase_name: m_explanation = "Éléments testés : groupement des $k$-ièmes arêtes."
            
    T = math.ceil(math.log2(m + 1))
    return html.Div([
        html.H4("Détails Mathématiques (CGT)", style={'marginTop': '15px', 'marginBottom': '10px', 'fontSize': '14px', 'color': COLORS['accent'], 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
        html.Div([html.Strong("Taille du problème ($m$) : ", style={'fontSize': '12px'}), html.Span(f"{m} éléments", style={'fontSize': '12px'}), dcc.Markdown(m_explanation, mathjax=True, style={'fontSize': '11px', 'color': '#7F8C8D', 'marginTop': '2px', 'marginBottom': '8px'})]),
        html.Div([html.Strong("Nombre de sondes ($T$) : ", style={'fontSize': '12px'}), dcc.Markdown(f"$T = \\lceil \\log_2(m + 1) \\rceil = \\lceil \\log_2({m} + 1) \\rceil = {T}$ sondes.", mathjax=True, style={'fontSize': '12px', 'display': 'inline-block', 'marginLeft': '5px'}), html.P(f"L'algorithme a généré une matrice de {T} lignes pour identifier 1 défaillance parmi {m} éléments.", style={'fontSize': '11px', 'color': '#7F8C8D', 'marginTop': '2px', 'marginBottom': '0'})])
    ], style={'backgroundColor': '#fdfdfe', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #e0e0e0', 'marginTop': '15px'})

# --- LE DÉCODEUR (Rapport Final) ---
def _build_diagnosis_report(results, topo, mode, fault_exists):
    """Calcule la localisation de la panne uniquement en croisant les résultats (0 ou 1) des sondes."""
    if not fault_exists:
        return html.Div([
            html.H4("🧠 Décodage du Contrôleur", style={'marginTop': 0, 'color': COLORS['success']}),
            html.P("Diagnostic : Réseau sain. Tous les syndromes sont nuls (0).")
        ], style={'backgroundColor': '#E8F8F5', 'padding': '15px', 'borderRadius': '5px', 'border': f"2px solid {COLORS['success']}", 'marginTop': '20px'})

    report_content = []

    # 1. DÉCODAGE PAR SYNDROME (Matrice Binaire LTP)
    if mode == 'ltp' and topo in ['complet', 'arbre']:
        phases = {}
        for res in results:
            if 'ltp_meta' in res['probe'] and res['probe']['ltp_meta']:
                meta = res['probe']['ltp_meta']
                phase = meta['phase']
                if phase not in phases:
                    phases[phase] = {'bits': {}, 'items': meta['items']}
                phases[phase]['bits'][meta['active_test']] = 1 if res['failed'] else 0
        
        fault_found = False
        for phase, data in phases.items():
            bits = data['bits']
            if not bits: continue
            syndrome = sum(val * (2**idx) for idx, val in bits.items())
            if syndrome > 0:
                fault_found = True
                fault_item = data['items'][syndrome - 1]
                bit_str = "".join(str(bits[i]) for i in sorted(bits.keys(), reverse=True))
                report_content.append(html.Div([
                    html.Strong(f"Analyse de la phase : {phase}", style={'color': COLORS['accent']}),
                    html.Ul([
                        html.Li(f"Vecteur binaire lu (T_n ... T1) : {bit_str}"),
                        html.Li(f"Conversion en Syndrome décimal : {syndrome}"),
                        html.Li(f"Déduction : L'élément fautif correspond à l'index {syndrome} => {fault_item}", style={'color': COLORS['fail'], 'fontWeight': 'bold'})
                    ], style={'margin': '5px 0', 'paddingLeft': '20px', 'fontSize': '13px'})
                ], style={'marginBottom': '10px'}))

        if not fault_found:
            report_content.append(html.P("Panne détectée par une sonde unitaire globale (Hors Matrice LTP).", style={'fontSize': '13px', 'fontStyle': 'italic'}))

    # 2. DÉCODAGE PAR INTERSECTION GÉOMÉTRIQUE (Grille & Linéaire)
    elif topo in ['lineaire', 'grille']:
        safe_edges = set()
        suspect_edges = None
        for res in results:
            path = res['probe']['path']
            edges = set(_normalize_edge(path[i], path[i+1]) for i in range(len(path)-1))
            if res['failed']:
                if suspect_edges is None: suspect_edges = edges
                else: suspect_edges = suspect_edges.intersection(edges)
            else:
                safe_edges = safe_edges.union(edges)
        
        if suspect_edges is not None:
            final_fault = suspect_edges - safe_edges
            report_content.append(html.Div([
                html.Strong("Logique d'Intersection (Fenêtre Glissante / Grille)", style={'color': COLORS['accent']}),
                html.Ul([
                    html.Li(f"Arêtes suspectes isolées par intersection des échecs."),
                    html.Li(f"Soustraction des arêtes validées par les sondes saines."),
                    html.Li(f"Déduction : L'arête défaillante est {list(final_fault)}", style={'color': COLORS['fail'], 'fontWeight': 'bold'})
                ], style={'margin': '5px 0', 'paddingLeft': '20px', 'fontSize': '13px'})
            ]))

    # 3. DÉCODAGE NAÏF
    else:
        for res in results:
            if res['failed']:
                report_content.append(html.Div([
                    html.Strong("Mode Naïf (Recherche Unitaire)", style={'color': COLORS['accent']}),
                    html.Ul([
                        html.Li(f"Sonde en échec : {res['probe']['step_name']}"),
                        html.Li(f"Déduction : Panne trouvée par balayage séquentiel.", style={'color': COLORS['fail'], 'fontWeight': 'bold'})
                    ], style={'margin': '5px 0', 'paddingLeft': '20px', 'fontSize': '13px'})
                ]))
                break

    return html.Div([
        html.H4("🧠 Rapport du Contrôleur (Décodage Non-Adaptatif)", style={'marginTop': '0', 'borderBottom': '2px solid #ccc', 'paddingBottom': '5px', 'color': '#2C3E50'}),
        html.P("Le contrôleur a collecté l'ensemble des statuts 0 et 1 pour effectuer ses calculs :", style={'fontSize': '12px', 'fontStyle': 'italic', 'color': '#7F8C8D'}),
        *report_content
    ], style={'backgroundColor': '#FDFEFE', 'padding': '15px', 'borderRadius': '5px', 'border': f"2px solid {COLORS['accent']}", 'marginTop': '20px'})


class NetworkEngine:
    def __init__(self, n, topo, mode='naif'):
        self.n = n
        self.type = topo
        self.mode = mode 
        self.G = self._build_graph()
        self.probes = []
        if self.type == 'grille': self._generate_grid_algo()
        elif self.type == 'lineaire': self._generate_linear_algo()
        elif self.type == 'complet': self._generate_complete_algo()
        elif self.type == 'arbre': self._generate_tree_algo()

    def _build_graph(self):
        if self.type == 'lineaire': return nx.path_graph(self.n)
        elif self.type == 'complet': return nx.complete_graph(self.n)
        elif self.type == 'arbre': return nx.random_tree(self.n, seed=42)
        elif self.type == 'grille':
            s = int(math.sqrt(self.n))
            return nx.grid_2d_graph(s, s)
        return nx.Graph()

    def _get_path(self, u, v):
        try: return nx.shortest_path(self.G, u, v)
        except: return []

    def _add_probe(self, path, step_id, step_name, description, ltp_meta=None):
        self.probes.append({'path': path, 'step_id': step_id, 'step_name': step_name, 'description': description, 'id': len(self.probes) + 1, 'ltp_meta': ltp_meta})

    def _get_ltp_subsets(self, items):
        m = len(items)
        if m == 0: return {'subsets': [], 'items': [], 'num_tests': 0, 'matrix': []}
        num_tests = math.ceil(math.log2(m + 1))
        tests = [[] for _ in range(num_tests)]
        matrix = [] 
        for t in range(num_tests):
            row_bits = []
            for i, item in enumerate(items):
                val = i + 1 
                bit = (val >> t) & 1
                row_bits.append(bit)
                if bit: tests[t].append(item)
            matrix.append(row_bits)
        return {'subsets': tests, 'items': items, 'num_tests': num_tests, 'matrix': matrix}

    def _generate_linear_algo(self):
        w = math.ceil(self.n / 2)
        windows = []
        for start in range(self.n):
            end = start + w
            if end >= self.n:
                last_start = max(0, self.n - 1 - w)
                last_end = self.n - 1
                if [last_start, last_end] not in windows: windows.append([last_start, last_end])
                break
            windows.append([start, end])
        for idx, (start, end) in enumerate(windows):
            path_aller = list(range(start, end + 1))
            full_path = path_aller + (path_aller[-2::-1] if len(path_aller) > 1 else [])
            lbl = "LTP-WIN" if self.mode == 'ltp' else "WIN"
            self._add_probe(full_path, f"{lbl}-{idx+1}", "Fenêtre Glissante", f"Test du segment {start} à {end}.")

    def _generate_complete_algo(self):
        if self.mode == 'naif':
            for i in range(1, self.n): self._add_probe([0, i, 0], "STAR", "Validation Hub", f"Test unitaire Hub 0 - Nœud {i}.")
            for u in range(1, self.n):
                for v in range(u+1, self.n): self._add_probe([0, u, v, 0], "CYCLE", "Triangulation", f"Test unitaire arête {u}-{v}.")
        else:
            hub_nodes = list(range(1, self.n))
            ltp_hub = self._get_ltp_subsets(hub_nodes)
            for idx, subset in enumerate(ltp_hub['subsets']):
                if not subset: continue
                path = [0]
                for node in subset: path.extend([node, 0])
                meta = {'matrix': ltp_hub['matrix'], 'items': ltp_hub['items'], 'active_test': idx, 'phase': 'Hub (Nœuds)'}
                self._add_probe(path, f"STAR-LTP-{idx+1}", "LTP Hub", f"Test combinatoire sur les nœuds {subset}.", ltp_meta=meta)
                
            other_edges = [(u, v) for u in range(1, self.n) for v in range(u+1, self.n)]
            ltp_cycles = self._get_ltp_subsets(other_edges)
            for idx, subset in enumerate(ltp_cycles['subsets']):
                if not subset: continue
                M = nx.MultiGraph()
                M.add_edges_from(subset)
                if 0 not in M: M.add_node(0)
                degrees = dict(M.degree())
                odd_nodes = [n for n, d in degrees.items() if d % 2 != 0 and n != 0]
                for node in odd_nodes: M.add_edge(0, node)
                components = list(nx.connected_components(M))
                for comp in components:
                    if 0 not in comp and len(comp) > 1:
                        v = list(comp)[0]
                        M.add_edge(0, v)
                        M.add_edge(0, v)
                circuit = list(nx.eulerian_circuit(M, source=0))
                path = [circuit[0][0]]
                for u, v in circuit: path.append(v)
                meta = {'matrix': ltp_cycles['matrix'], 'items': ltp_cycles['items'], 'active_test': idx, 'phase': 'Cycles (Arêtes)'}
                self._add_probe(path, f"CYCLE-LTP-{idx+1}", "LTP Distant", f"Test combinatoire sur {len(subset)} arêtes distantes.", ltp_meta=meta)

    def _generate_tree_algo(self):
        nodes_by_depth = sorted(self.G.nodes(), key=lambda x: len(self._get_path(0, x)))
        if self.mode == 'naif':
            for i in nodes_by_depth:
                if i == 0: continue
                path = self._get_path(0, i)
                self._add_probe(path + path[-2::-1], "BRANCH", "Sondage Unitaire", f"Validation branche vers {i}.")
        else:
            depth_groups = {}
            for n in nodes_by_depth:
                if n == 0: continue
                d = len(self._get_path(0, n))
                if d not in depth_groups: depth_groups[d] = []
                depth_groups[d].append(n)
            for d, nodes in depth_groups.items():
                ltp_data = self._get_ltp_subsets(nodes)
                for idx, subset in enumerate(ltp_data['subsets']):
                    if not subset: continue
                    path = [0]
                    for node in subset:
                        p = self._get_path(0, node)
                        path.extend(p[1:] + p[-2::-1])
                    meta = {'matrix': ltp_data['matrix'], 'items': ltp_data['items'], 'active_test': idx, 'phase': f'Profondeur {d}'}
                    self._add_probe(path, f"DEPTH-LTP-{d}-{idx}", f"LTP Profondeur {d}", f"Test combinatoire sur les nœuds {subset}.", ltp_meta=meta)

    def _generate_grid_algo(self):
        s = int(math.sqrt(self.n))
        origin = (0,0)

        # 1a. Col 0 Globale
        path = [(r, 0) for r in range(s)] + [(r, 0) for r in range(s-2, -1, -1)]
        self._add_probe(path, "1a", "Axe Vertical", "Test global de la Colonne 0.")

        # 1b. Col 0
        if self.mode == 'naif':
            for r in range(s-1):
                p = self._get_path(origin, (r,0)) + [(r,1), (r+1,1)] + self._get_path((r+1,0), origin)
                self._add_probe(p, "1b", "Détail Col 0", f"Test unitaire arête d'indice {r} : ({r},0)-({r+1},0).")
        else:
            edges_c0 = list(range(s-1))
            ltp_data = self._get_ltp_subsets(edges_c0)
            for idx, subset in enumerate(ltp_data['subsets']):
                if not subset: continue
                p = [(0,0)]
                for r in range(s-1):
                    if r in subset: p.append((r+1, 0))
                    else: p.extend([(r,1), (r+1,1), (r+1,0)]) 
                p.extend(p[-2::-1]) 
                meta = {'matrix': ltp_data['matrix'], 'items': ltp_data['items'], 'active_test': idx, 'phase': 'Colonne 0'}
                self._add_probe(p, f"1b-LTP-{idx+1}", "LTP Col 0", f"Test combinatoire sur Col 0 (bit {idx}).", ltp_meta=meta)

        # 2a. Row 0 Globale
        path = [(0, c) for c in range(s)] + [(0, c) for c in range(s-2, -1, -1)]
        self._add_probe(path, "2a", "Axe Horizontal", "Test global de la Ligne 0.")

        # 2b. Row 0
        if self.mode == 'naif':
            for c in range(s-1):
                p = self._get_path(origin, (0,c)) + [(1,c), (1,c+1)] + self._get_path((0,c+1), origin)
                self._add_probe(p, "2b", "Détail Row 0", f"Test unitaire arête d'indice {c} : (0,{c})-(0,{c+1}).")
        else:
            edges_r0 = list(range(s-1))
            ltp_data = self._get_ltp_subsets(edges_r0)
            for idx, subset in enumerate(ltp_data['subsets']):
                if not subset: continue
                p = [(0,0)]
                for c in range(s-1):
                    if c in subset: p.append((0, c+1))
                    else: p.extend([(1,c), (1,c+1), (0,c+1)])
                p.extend(p[-2::-1])
                meta = {'matrix': ltp_data['matrix'], 'items': ltp_data['items'], 'active_test': idx, 'phase': 'Ligne 0'}
                self._add_probe(p, f"2b-LTP-{idx+1}", "LTP Ligne 0", f"Test combinatoire sur Ligne 0 (bit {idx}).", ltp_meta=meta)

        # 3a. Lignes Globales
        if self.mode == 'naif':
            for r in range(1, s):
                p = self._get_path(origin, (r,0)) 
                p += [(r, c) for c in range(1, s)] 
                p += [(r, c) for c in range(s-2, -1, -1)] 
                p += self._get_path((r,0), origin)[1:]
                self._add_probe(p, "3a", "Ligne Complète", f"Test global de la Ligne {r}.")
        else:
            rows = list(range(1, s))
            ltp_data = self._get_ltp_subsets(rows)
            for idx, subset in enumerate(ltp_data['subsets']):
                if not subset: continue
                p = [(0,0)]
                for r in range(1, s):
                    p.append((r,0))
                    if r in subset:
                        p.extend([(r,c) for c in range(1, s)])
                        p.extend([(r,c) for c in range(s-2, -1, -1)])
                p.extend([(r,0) for r in range(s-2, -1, -1)])
                meta = {'matrix': ltp_data['matrix'], 'items': ltp_data['items'], 'active_test': idx, 'phase': 'Lignes Globales'}
                self._add_probe(p, f"3a-LTP-{idx+1}", "LTP Lignes", f"Test combinatoire sur Lignes (bit {idx}).", ltp_meta=meta)

        # 3b. Échelle Verticale
        if self.mode == 'naif':
            for k in range(s-1):
                path = self._get_path(origin, (0,k))
                for r in range(1, s):
                    if r % 2 != 0: path += self._get_path(path[-1], (r, k))[1:] + [(r, k+1)]
                    else: path += self._get_path(path[-1], (r, k+1))[1:] + [(r, k)]
                path += self._get_path(path[-1], origin)[1:]
                self._add_probe(path, "3b", "Échelle Verticale", f"Test simultané de la {k+1}ème arête d'indice {k} de TOUTES les lignes.")
        else:
            k_indices = list(range(s-1))
            ltp_data = self._get_ltp_subsets(k_indices)
            for idx, subset in enumerate(ltp_data['subsets']):
                if not subset: continue
                p = [(0,0)]
                last_k = 0
                for k in range(s-1):
                    if k in subset:
                        p.extend([(0,c) for c in range(last_k+1, k+1)])
                        last_k = k
                        for r in range(1, s): p.extend([(r,k), (r,k+1), (r,k)])
                        p.extend([(r,k) for r in range(s-2, -1, -1)])
                p.extend([(0,c) for c in range(last_k-1, -1, -1)])
                meta = {'matrix': ltp_data['matrix'], 'items': ltp_data['items'], 'active_test': idx, 'phase': 'Échelle Vert.'}
                self._add_probe(p, f"3b-LTP-{idx+1}", "LTP Échelle V.", f"Test combinatoire des arêtes horizontales (bit {idx}).", ltp_meta=meta)

        # 4a. Colonnes Globales
        if self.mode == 'naif':
            for c in range(1, s):
                p = self._get_path(origin, (0,c))
                p += [(r, c) for r in range(1, s)]
                p += [(r, c) for r in range(s-2, -1, -1)]
                p += self._get_path((0,c), origin)[1:]
                self._add_probe(p, "4a", "Colonne Complète", f"Test global de la Colonne {c}.")
        else:
            cols = list(range(1, s))
            ltp_data = self._get_ltp_subsets(cols)
            for idx, subset in enumerate(ltp_data['subsets']):
                if not subset: continue
                p = [(0,0)]
                for c in range(1, s):
                    p.append((0,c))
                    if c in subset:
                        p.extend([(r,c) for r in range(1, s)])
                        p.extend([(r,c) for r in range(s-2, -1, -1)])
                p.extend([(0,c) for c in range(s-2, -1, -1)])
                meta = {'matrix': ltp_data['matrix'], 'items': ltp_data['items'], 'active_test': idx, 'phase': 'Colonnes Globales'}
                self._add_probe(p, f"4a-LTP-{idx+1}", "LTP Colonnes", f"Test combinatoire sur Colonnes (bit {idx}).", ltp_meta=meta)

        # 4b. Échelle Horizontale
        if self.mode == 'naif':
            for k in range(s-1):
                path = self._get_path(origin, (k,0))
                for c in range(1, s):
                    if c % 2 != 0: path += self._get_path(path[-1], (k, c))[1:] + [(k+1, c)]
                    else: path += self._get_path(path[-1], (k+1, c))[1:] + [(k, c)]
                path += self._get_path(path[-1], origin)[1:]
                self._add_probe(path, "4b", "Échelle Horizontale", f"Test simultané de la {k+1}ème arête d'indice {k} de TOUTES les colonnes.")
        else:
            k_indices = list(range(s-1))
            ltp_data = self._get_ltp_subsets(k_indices)
            for idx, subset in enumerate(ltp_data['subsets']):
                if not subset: continue
                p = [(0,0)]
                last_k = 0
                for k in range(s-1):
                    if k in subset:
                        p.extend([(r,0) for r in range(last_k+1, k+1)])
                        last_k = k
                        for c in range(1, s): p.extend([(k,c), (k+1,c), (k,c)])
                        p.extend([(k,c) for c in range(s-2, -1, -1)])
                p.extend([(r,0) for r in range(last_k-1, -1, -1)])
                meta = {'matrix': ltp_data['matrix'], 'items': ltp_data['items'], 'active_test': idx, 'phase': 'Échelle Horiz.'}
                self._add_probe(p, f"4b-LTP-{idx+1}", "LTP Échelle H.", f"Test combinatoire des arêtes verticales (bit {idx}).", ltp_meta=meta)

    def run_simulation(self, fault):
        fsig = str(tuple(sorted(fault, key=str))) if fault else ""
        results = []
        for probe in self.probes:
            path = probe['path']
            failed = False
            for i in range(len(path)-1):
                if str(tuple(sorted((path[i], path[i+1]), key=str))) == fsig: failed = True
            results.append({'probe': probe, 'failed': failed})
        return results

# --- INTERFACE DASH ---
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'fontFamily': 'Segoe UI', 'padding': '20px'}, children=[
    
    html.Div(style={'textAlign': 'center', 'marginBottom': '30px'}, children=[
        html.H1("Simulateur de Diagnostic Optique", style={'color': COLORS['text'], 'marginBottom': '5px'}),
        html.Div("Comparaison des stratégies de sondage (Naïf vs Combinatorial Group Testing)", style={'color': '#7F8C8D'})
    ]),

    html.Div(className='row', style={'display': 'flex', 'gap': '20px'}, children=[
        html.Div(style={'flex': '1', 'maxWidth': '450px'}, children=[
            
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}, children=[
                html.H3("1. Configuration", style={'marginTop': 0, 'color': COLORS['accent'], 'fontSize': '18px'}),
                
                html.Label("Topologie :"),
                dcc.Dropdown(id='topo', options=[
                    {'label': 'Grille (LTP Hiérarchique)', 'value': 'grille'},
                    {'label': 'Linéaire (Fenêtre Glissante)', 'value': 'lineaire'},
                    {'label': 'Complet (Hub & Cycles)', 'value': 'complet'},
                    {'label': 'Arbre (Profondeur)', 'value': 'arbre'}
                ], value='grille', clearable=False),

                html.Label("Mode d'Algorithme :", style={'marginTop': '15px', 'display': 'block', 'fontWeight': 'bold'}),
                dcc.RadioItems(id='algo-mode', options=[
                    {'label': ' Naïf (Boucle itérative)', 'value': 'naif'},
                    {'label': ' LTP (Article - O(log N))', 'value': 'ltp'}
                ], value='ltp', labelStyle={'display': 'block', 'margin': '5px 0'}),

                html.Label("Taille du réseau (Nœuds) :", style={'marginTop': '15px', 'display': 'block'}),
                dcc.Slider(id='n-slider', min=5, max=25, step=1, value=16, marks={5:'5', 10:'10', 16:'16', 25:'25'}),
                
                html.Button("Générer Réseau", id='btn-build', style={'width': '100%', 'marginTop': '15px', 'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '10px', 'borderRadius': '5px', 'cursor': 'pointer'}),
                html.Hr(),
                html.Label("2. Injecter une Panne :"),
                dcc.Dropdown(id='fault', placeholder="Sélectionner une arête...", searchable=True),
            ]),

            html.Div(style={'backgroundColor': COLORS['step_box'], 'padding': '20px', 'borderRadius': '10px', 'border': f'2px solid {COLORS["accent"]}'}, children=[
                html.H3("Status de l'Algorithme", style={'marginTop': 0, 'color': COLORS['text'], 'fontSize': '18px', 'borderBottom': '1px solid #ccc', 'paddingBottom': '10px'}),
                html.Div(id='step-display', children=[html.P("En attente de simulation...", style={'color': '#7F8C8D'})]),
                html.Div(id='matrix-display', style={'marginTop': '15px'}),
                html.Div(id='math-details-display'),
                html.Div(id='path-display'),
                html.Div(id='diagnosis-report-display')
            ]),
            
            html.Div(id='algo-doc', style={'marginTop': '20px', 'fontSize': '13px', 'color': '#555'})
        ]),

        html.Div(style={'flex': '2'}, children=[
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'height': '600px'}, children=[
                dcc.Graph(id='graph', style={'height': '100%'}, config={'displayModeBar': False})
            ]),

            html.Div(style={'backgroundColor': COLORS['card'], 'marginTop': '20px', 'padding': '20px', 'borderRadius': '10px', 'display': 'flex', 'alignItems': 'center', 'gap': '15px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}, children=[
                html.Button("▶️ Lecture", id='btn-play', style={'backgroundColor': COLORS['success'], 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold'}),
                html.Div(style={'flex': '1'}, children=[
                    dcc.Slider(id='sim-slider', min=1, max=1, step=1, value=1, tooltip={"placement": "bottom", "always_visible": True}, marks=None)
                ]),
                html.Div(id='counter-display', style={'fontWeight': 'bold', 'color': COLORS['text'], 'minWidth': '120px', 'textAlign': 'right'})
            ])
        ])
    ]),

    dcc.Store(id='st-topo'), dcc.Store(id='st-fault'),
    dcc.Interval(id='anim-interval', interval=1000, n_intervals=0, disabled=True)
])

# --- CALLBACKS ---
@app.callback(
    [Output('graph', 'figure'), 
     Output('st-topo', 'data'), 
     Output('st-fault', 'data'),
     Output('fault', 'options'), 
     Output('fault', 'value'),
     Output('sim-slider', 'max'), 
     Output('sim-slider', 'value'), 
     Output('step-display', 'children'),
     Output('matrix-display', 'children'),
     Output('math-details-display', 'children'),
     Output('path-display', 'children'),
     Output('diagnosis-report-display', 'children'),
     Output('counter-display', 'children'),
     Output('algo-doc', 'children'),
     Output('anim-interval', 'disabled'), 
     Output('btn-play', 'children')],
    [Input('btn-build', 'n_clicks'), 
     Input('graph', 'clickData'), 
     Input('fault', 'value'),
     Input('sim-slider', 'value'), 
     Input('btn-play', 'n_clicks'), 
     Input('anim-interval', 'n_intervals'),
     Input('algo-mode', 'value')],
    [State('topo', 'value'), 
     State('n-slider', 'value'), 
     State('st-topo', 'data'), 
     State('st-fault', 'data')]
)
def update_simulation(b_build, click, f_val, s_val, b_play, n_intervals, mode, topo, n_nodes, st_t, st_f):
    ctx_id = ctx.triggered_id
    fig = go.Figure()
    anim_disabled = True
    btn_text = "▶️ Lecture"
    matrix_html, math_details_html, path_html, diagnosis_html = html.Div(), html.Div(), html.Div(), html.Div()
    
    if ctx_id in ['btn-build', 'algo-mode', None] or not st_t:
        n_final = int(math.sqrt(n_nodes))**2 if topo == 'grille' else n_nodes
        st_t = {'n': n_final, 'type': topo, 'mode': mode}
        st_f, f_val, s_val = None, None, 1
    
    if st_f and isinstance(st_f[0], list): st_f = sorted((tuple(st_f[0]), tuple(st_f[1])), key=str)

    eng = NetworkEngine(st_t['n'], st_t['type'], mode=st_t.get('mode', 'ltp'))
    opts = [{'label': f"Lien {u} - {v}", 'value': str(sorted((u, v), key=str))} for u,v in eng.G.edges()]
    
    if ctx_id == 'graph' and click:
        try:
            p = click['points'][0]['customdata']
            st_f = sorted([tuple(p[0]), tuple(p[1])], key=str) if st_t['type'] == 'grille' else sorted(p, key=str)
            f_val, s_val = str(st_f), 1
        except: pass
    elif ctx_id == 'fault' and f_val:
        try:
            v = ast.literal_eval(f_val)
            st_f = sorted([tuple(v[0]), tuple(v[1])], key=str) if st_t['type'] == 'grille' else sorted(v, key=str)
            s_val = 1
        except: pass

    results = eng.run_simulation(st_f)
    max_s = max(len(results), 1)
    if s_val is None or s_val > max_s: s_val = 1

    if ctx_id == 'btn-play':
        anim_disabled = False
        if s_val >= max_s: s_val = 1
    elif ctx_id == 'anim-interval':
        if s_val < max_s:
            s_val += 1
            if results[s_val - 1]['failed']: anim_disabled, btn_text = True, "▶️ Reprendre"
            else: anim_disabled, btn_text = False, "⏸️ Stop"
        else: anim_disabled, btn_text = True, "↺ Reset"
    
    if anim_disabled:
        if s_val >= max_s: btn_text = "↺ Reset"
        elif s_val > 1 and results[s_val-1]['failed']: btn_text = "▶️ Reprendre"
        else: btn_text = "▶️ Lecture"
    else: btn_text = "⏸️ Stop"

    pos = {}
    if st_t['type'] == 'lineaire': pos = {n: (n, 0) for n in eng.G.nodes()}
    elif st_t['type'] == 'grille': pos = {n: (n[1], -n[0]) for n in eng.G.nodes()}
    elif st_t['type'] == 'arbre': pos = nx.kamada_kawai_layout(eng.G)
    elif st_t['type'] == 'complet': pos = nx.circular_layout(eng.G)

    edge_x, edge_y = [], []
    for u, v in eng.G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        fig.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(width=15, color='rgba(0,0,0,0)'), hoverinfo='text', text=f"Lien {u}-{v}", customdata=[[u,v],[u,v],[u,v]], showlegend=False))
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color=COLORS['edge_inactive'], width=2), hoverinfo='skip'))

    if st_f:
        try:
            u_f, v_f = st_f
            fx0, fy0, fx1, fy1 = pos[u_f][0], pos[u_f][1], pos[v_f][0], pos[v_f][1]
            fig.add_trace(go.Scatter(x=[fx0, fx1], y=[fy0, fy1], mode='lines', line=dict(color=COLORS['fail'], width=4, dash='dot')))
            fig.add_trace(go.Scatter(x=[(fx0+fx1)/2], y=[(fy0+fy1)/2], mode='markers', marker=dict(symbol='x', size=15, color=COLORS['fail'])))
        except: pass

    if results:
        curr = results[s_val - 1]
        probe, is_failed = curr['probe'], curr['failed']
        path = probe['path']
        
        test_idx = probe.get('ltp_meta', {}).get('active_test', (s_val - 1)) if probe.get('ltp_meta') else (s_val - 1)
        dynamic_color = PROBE_COLORS[test_idx % len(PROBE_COLORS)]
        
        px, py = [pos[n][0] for n in path], [pos[n][1] for n in path]
        
        highlighted_edges = _get_highlighted_edges(probe, st_t['type'])
        for highlighted_edge in highlighted_edges:
            try:
                hu, hv = highlighted_edge
                hx0, hy0 = pos[hu]; hx1, hy1 = pos[hv]
                fig.add_trace(go.Scatter(x=[hx0, hx1], y=[hy0, hy1], mode='lines', line=dict(width=9, color='#E74C3C'), opacity=0.8, hoverinfo='skip', showlegend=False))
            except: pass

        fig.add_trace(go.Scatter(x=px, y=py, mode='lines', line=dict(width=4, color=dynamic_color), opacity=0.9))
        
        if st_t['type'] in ['grille', 'lineaire']:
            fig.layout.annotations = _build_probe_step_annotations(path, pos, dynamic_color)
        
        if len(path) > 1:
            fig.add_trace(go.Scatter(x=[px[0]], y=[py[0]], mode='markers', marker=dict(size=12, color=dynamic_color, symbol='circle')))
            fig.add_trace(go.Scatter(x=[px[len(path)//2]], y=[py[len(path)//2]], mode='markers', marker=dict(size=10, color=dynamic_color, symbol='triangle-up')))

        status_col = COLORS['fail'] if is_failed else COLORS['success']
        step_content = [
            html.Div([html.Span("ÉTAPE : ", style={'fontWeight': 'bold', 'color': '#7F8C8D', 'fontSize': '12px'}), html.Span(f"{probe['step_id']} - {probe['step_name']}", style={'fontWeight': 'bold', 'color': dynamic_color})]),
            html.Div([html.Span("DESCRIPTION : ", style={'fontWeight': 'bold', 'color': '#7F8C8D', 'fontSize': '12px'}), html.P(probe['description'], style={'fontSize': '14px', 'margin': '5px 0'})]),
            html.Div(style={'backgroundColor': 'white', 'padding': '10px', 'borderLeft': f'5px solid {status_col}'}, children=[html.Span("STATUT : ", style={'fontWeight': 'bold', 'fontSize': '12px'}), html.Span("❌ ÉCHEC" if is_failed else "✅ SUCCÈS", style={'fontWeight': 'bold', 'color': status_col})])
        ]
        
        matrix_html = _build_matrix_html(probe.get('ltp_meta'))
        math_details_html = _build_math_details_html(probe.get('ltp_meta'), st_t['type'])
        path_html = _build_path_html(path, dynamic_color)

        if s_val == max_s:
            diagnosis_html = _build_diagnosis_report(results, st_t['type'], st_t.get('mode', 'naif'), st_f is not None)
        else:
            diagnosis_html = html.Div(
                "Le rapport de diagnostic mathématique sera généré à la fin de la séquence de test...",
                style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px', 'color': '#7F8C8D', 'fontStyle': 'italic', 'textAlign': 'center', 'marginTop': '20px', 'border': '1px dashed #ccc'}
            )
    else:
        step_content = [html.P("Aucune sonde générée.", style={'color': '#7F8C8D'})]

    nxp, nyp = [pos[n][0] for n in eng.G.nodes()], [pos[n][1] for n in eng.G.nodes()]
    fig.add_trace(go.Scatter(x=nxp, y=nyp, mode='markers+text', text=[str(n) for n in eng.G.nodes()], textposition="top center", textfont=dict(color=COLORS['text'], size=10), marker=dict(size=15, color='white', line=dict(width=2, color=COLORS['text'])), hoverinfo='none'))

    fig.update_layout(margin=dict(l=20,r=20,t=20,b=20), xaxis={'visible':False}, yaxis={'visible':False, 'scaleanchor':'x', 'scaleratio':1 if st_t['type']=='grille' else None}, plot_bgcolor=COLORS['bg'], showlegend=False)

    return fig, st_t, st_f, opts, f_val, max_s, s_val, step_content, matrix_html, math_details_html, path_html, diagnosis_html, f"Total Sondes : {max_s} | Pos : {s_val}", ALGO_DOCS.get(st_t['type'], ""), anim_disabled, btn_text

if __name__ == '__main__':
    app.run(debug=True)