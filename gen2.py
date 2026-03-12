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

# --- DOCUMENTATION ---
ALGO_DOCS = {
    'grille': dcc.Markdown(r'''
        #### Algorithme 2 : Grille Hiérarchique
        **Stratégie :** Division par Axes et Groupes.
        * **Mode Naïf :** Test unitaire de chaque barreau de l'échelle.
        * **Mode LTP (Article) :** Tests de groupe (Combinatorial Group Testing) sur les arêtes d'une même ligne/colonne pour une complexité logarithmique.
    '''),
    'lineaire': dcc.Markdown(r'''
        #### Algorithme 1 : Fenêtre Glissante
        **Stratégie :** Recouvrement Progressif.
        La borne théorique est linéaire (W ≈ N/2). Le mode LTP et Naïf sont ici similaires par nature géométrique de la ligne.
    ''', mathjax=True),
    'complet': dcc.Markdown(r'''
        #### Algorithme 3 : Hub & Spoke
        **Stratégie :** Centre vers Périphérie.
        1. **Phase Étoile :** Validation des liens du Hub.
        2. **Phase Cycle :** Validation des liens distants.
        * **Mode LTP :** Divise massivement le nombre de sondes en testant des combinaisons binaires d'arêtes simultanément.
    ''', mathjax=True),
    'arbre': dcc.Markdown(r'''
        #### Algorithme 4 : Profondeur
        **Stratégie :** Racine vers Feuilles.
        * **Mode Naïf :** Une sonde par branche (force brute).
        * **Mode LTP :** Regroupement des branches par profondeur pour un diagnostic logarithmique accéléré.
    ''', mathjax=True)
}

# --- MOTEUR ALGORITHMIQUE UNIFIÉ ---
def _distance_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

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

def _normalize_edge(u, v):
    return tuple(sorted((u, v), key=str))

class NetworkEngine:
    def __init__(self, n, topo, mode='naif'):
        self.n = n
        self.type = topo
        self.mode = mode # 'naif' ou 'ltp'
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

    def _add_probe(self, path, step_id, step_name, description):
        self.probes.append({
            'path': path, 'step_id': step_id, 'step_name': step_name, 
            'description': description, 'id': len(self.probes) + 1
        })

    # --- NOUVEAU : MOTEUR LOGARITHMIQUE (LTP) ---
    def _get_ltp_subsets(self, items):
        """Génère la matrice de tests combinatoires binaires."""
        m = len(items)
        if m == 0: return []
        num_tests = math.ceil(math.log2(m + 1))
        tests = [[] for _ in range(num_tests)]
        for i, item in enumerate(items):
            val = i + 1 # 0 est réservé au cas "aucune erreur"
            for t in range(num_tests):
                if (val >> t) & 1:
                    tests[t].append(item)
        return tests

    # --- GENERATEURS DE SONDES ---
    def _generate_linear_algo(self):
        # Pour le linéaire, la borne est Theta(n), le LTP n'apporte pas de gain logarithmique direct
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
            # Phase Hub
            for i in range(1, self.n):
                self._add_probe([0, i, 0], "STAR", "Validation Hub", f"Test unitaire Hub 0 - Nœud {i}.")
            # Phase Cycles
            for u in range(1, self.n):
                for v in range(u+1, self.n):
                    self._add_probe([0, u, v, 0], "CYCLE", "Triangulation", f"Test unitaire arête {u}-{v}.")
        else:
            # MODE LTP : Réduction drastique des sondes
            hub_nodes = list(range(1, self.n))
            ltp_hub = self._get_ltp_subsets(hub_nodes)
            for idx, subset in enumerate(ltp_hub):
                if not subset: continue
                path = [0]
                for node in subset: path.extend([node, 0])
                self._add_probe(path, f"STAR-LTP-{idx+1}", "LTP Hub", f"Test combiné (bit {idx}) sur les nœuds {subset}.")
                
            other_edges = [(u, v) for u in range(1, self.n) for v in range(u+1, self.n)]
            ltp_cycles = self._get_ltp_subsets(other_edges)
            for idx, subset in enumerate(ltp_cycles):
                if not subset: continue
                path = [0]
                for u, v in subset: path.extend([u, v, 0])
                self._add_probe(path, f"CYCLE-LTP-{idx+1}", "LTP Distant", f"Test combiné (bit {idx}) sur {len(subset)} arêtes distantes.")

    def _generate_tree_algo(self):
        nodes_by_depth = sorted(self.G.nodes(), key=lambda x: len(self._get_path(0, x)))
        if self.mode == 'naif':
            for i in nodes_by_depth:
                if i == 0: continue
                path = self._get_path(0, i)
                self._add_probe(path + path[-2::-1], "BRANCH", "Sondage Unitaire", f"Validation branche vers {i}.")
        else:
            # Mode LTP : Groupement par profondeur (simplifié pour visualisation)
            depth_groups = {}
            for n in nodes_by_depth:
                if n == 0: continue
                d = len(self._get_path(0, n))
                if d not in depth_groups: depth_groups[d] = []
                depth_groups[d].append(n)
                
            for d, nodes in depth_groups.items():
                ltp_tests = self._get_ltp_subsets(nodes)
                for idx, subset in enumerate(ltp_tests):
                    if not subset: continue
                    path = [0]
                    for node in subset:
                        p = self._get_path(0, node)
                        path.extend(p[1:] + p[-2::-1])
                    self._add_probe(path, f"DEPTH-LTP-{d}-{idx}", f"LTP Profondeur {d}", f"Test combiné sur les nœuds {subset}.")

    def _generate_grid_algo(self):
        s = int(math.sqrt(self.n))
        origin = (0,0)

        # 1a. Col 0 Globale
        path = [(r, 0) for r in range(s)] + [(r, 0) for r in range(s-2, -1, -1)]
        self._add_probe(path, "1a", "Axe Vertical", "Test global de la Colonne 0.")

        if self.mode == 'naif':
            # 1b. Col 0 Unitaire
            for r in range(s-1):
                p = self._get_path(origin, (r,0)) + [(r,1), (r+1,1)] + self._get_path((r+1,0), origin)
                self._add_probe(p, "1b", "Détail Col 0", f"Test unitaire arête ({r},0)-({r+1},0).")
        else:
            # 1b. Col 0 LTP (Via Col 1)
            edges_c0 = [(r, 0) for r in range(s-1)]
            for idx, subset in enumerate(self._get_ltp_subsets(edges_c0)):
                if not subset: continue
                p = [origin]
                for r, _ in subset: p.extend([(r,0), (r,1), (r+1,1), (r+1,0), origin])
                self._add_probe(p, f"1b-LTP-{idx+1}", "LTP Col 0", f"Test combinatoire sur {len(subset)} arêtes de Col 0.")

        # 2a. Row 0 Globale
        path = [(0, c) for c in range(s)] + [(0, c) for c in range(s-2, -1, -1)]
        self._add_probe(path, "2a", "Axe Horizontal", "Test global de la Ligne 0.")

        if self.mode == 'naif':
            for c in range(s-1):
                p = self._get_path(origin, (0,c)) + [(1,c), (1,c+1)] + self._get_path((0,c+1), origin)
                self._add_probe(p, "2b", "Détail Row 0", f"Test unitaire arête (0,{c})-(0,{c+1}).")
        else:
            # 2b. Row 0 LTP (Via Row 1)
            edges_r0 = [(0, c) for c in range(s-1)]
            for idx, subset in enumerate(self._get_ltp_subsets(edges_r0)):
                if not subset: continue
                p = [origin]
                for _, c in subset: p.extend([(0,c), (1,c), (1,c+1), (0,c+1), origin])
                self._add_probe(p, f"2b-LTP-{idx+1}", "LTP Ligne 0", f"Test combinatoire sur {len(subset)} arêtes de Ligne 0.")

        # (Nous simplifions les étapes 3 et 4 pour la lisibilité visuelle du code dans les deux modes)
        for r in range(1, s):
            p = self._get_path(origin, (r,0)) + [(r, c) for c in range(1, s)] + [(r, c) for c in range(s-2, -1, -1)] + self._get_path((r,0), origin)[1:]
            self._add_probe(p, "3a", "Ligne Complète", f"Test global Ligne {r}.")

    # --- EXECUTION ---
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
        html.Div(style={'flex': '1', 'maxWidth': '400px'}, children=[
            
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}, children=[
                html.H3("1. Configuration", style={'marginTop': 0, 'color': COLORS['accent'], 'fontSize': '18px'}),
                
                html.Label("Topologie :"),
                dcc.Dropdown(id='topo', options=[
                    {'label': 'Grille (LTP Hiérarchique)', 'value': 'grille'},
                    {'label': 'Linéaire (Fenêtre Glissante)', 'value': 'lineaire'},
                    {'label': 'Complet (Hub & Cycles)', 'value': 'complet'},
                    {'label': 'Arbre (Profondeur)', 'value': 'arbre'}
                ], value='grille', clearable=False),

                # NOUVEAU BOUTON RADIO POUR LE MODE
                html.Label("Mode d'Algorithme :", style={'marginTop': '15px', 'display': 'block', 'fontWeight': 'bold'}),
                dcc.RadioItems(id='algo-mode', options=[
                    {'label': ' Naïf (Boucle itérative)', 'value': 'naif'},
                    {'label': ' LTP (Article - O(log N))', 'value': 'ltp'}
                ], value='naif', labelStyle={'display': 'block', 'margin': '5px 0'}),

                html.Label("Taille du réseau (Nœuds) :", style={'marginTop': '15px', 'display': 'block'}),
                dcc.Slider(id='n-slider', min=5, max=25, step=1, value=16, marks={5:'5', 10:'10', 16:'16', 25:'25'}),
                
                html.Button("Générer Réseau", id='btn-build', style={'width': '100%', 'marginTop': '15px', 'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '10px', 'borderRadius': '5px', 'cursor': 'pointer'}),
                html.Hr(),
                html.Label("2. Injecter une Panne :"),
                dcc.Dropdown(id='fault', placeholder="Sélectionner une arête...", searchable=True),
            ]),

            html.Div(style={'backgroundColor': COLORS['step_box'], 'padding': '20px', 'borderRadius': '10px', 'border': f'2px solid {COLORS["accent"]}'}, children=[
                html.H3("Status de l'Algorithme", style={'marginTop': 0, 'color': COLORS['text'], 'fontSize': '18px', 'borderBottom': '1px solid #ccc', 'paddingBottom': '10px'}),
                html.Div(id='step-display', children=[html.P("En attente de simulation...", style={'color': '#7F8C8D'})])
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
     Input('algo-mode', 'value')], # Trigger quand on change le mode !
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
    
    if ctx_id in ['btn-build', 'algo-mode', None] or not st_t:
        n_final = int(math.sqrt(n_nodes))**2 if topo == 'grille' else n_nodes
        st_t = {'n': n_final, 'type': topo, 'mode': mode}
        st_f, f_val, s_val = None, None, 1
    
    if st_f and isinstance(st_f[0], list): st_f = sorted((tuple(st_f[0]), tuple(st_f[1])), key=str)

    # Création du moteur avec le MODE sélectionné
    eng = NetworkEngine(st_t['n'], st_t['type'], mode=st_t.get('mode', 'naif'))
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
        col = COLORS['fail'] if is_failed else COLORS['success']
        
        px, py = [pos[n][0] for n in path], [pos[n][1] for n in path]
        fig.add_trace(go.Scatter(x=px, y=py, mode='lines', line=dict(width=4, color=col), opacity=0.9))
        
        if st_t['type'] in ['grille', 'lineaire']:
            fig.layout.annotations = _build_probe_step_annotations(path, pos, col)
        
        if len(path) > 1:
            fig.add_trace(go.Scatter(x=[px[0]], y=[py[0]], mode='markers', marker=dict(size=12, color=col, symbol='circle')))
            fig.add_trace(go.Scatter(x=[px[len(path)//2]], y=[py[len(path)//2]], mode='markers', marker=dict(size=10, color=col, symbol='triangle-up')))

        status_col = COLORS['fail'] if is_failed else COLORS['success']
        step_content = [
            html.Div([html.Span("ÉTAPE : ", style={'fontWeight': 'bold', 'color': '#7F8C8D', 'fontSize': '12px'}), html.Span(f"{probe['step_id']} - {probe['step_name']}", style={'fontWeight': 'bold', 'color': COLORS['accent']})]),
            html.Div([html.Span("DESCRIPTION : ", style={'fontWeight': 'bold', 'color': '#7F8C8D', 'fontSize': '12px'}), html.P(probe['description'], style={'fontSize': '14px', 'margin': '5px 0'})]),
            html.Div(style={'backgroundColor': 'white', 'padding': '10px', 'borderLeft': f'5px solid {status_col}'}, children=[html.Span("STATUT : ", style={'fontWeight': 'bold', 'fontSize': '12px'}), html.Span("❌ ÉCHEC" if is_failed else "✅ SUCCÈS", style={'fontWeight': 'bold', 'color': status_col})])
        ]
    else:
        step_content = [html.P("Aucune sonde générée.", style={'color': '#7F8C8D'})]

    nxp, nyp = [pos[n][0] for n in eng.G.nodes()], [pos[n][1] for n in eng.G.nodes()]
    fig.add_trace(go.Scatter(x=nxp, y=nyp, mode='markers+text', text=[str(n) for n in eng.G.nodes()], textposition="top center", textfont=dict(color=COLORS['text'], size=10), marker=dict(size=15, color='white', line=dict(width=2, color=COLORS['text'])), hoverinfo='none'))

    fig.update_layout(
        margin=dict(l=20,r=20,t=20,b=20), xaxis={'visible':False}, yaxis={'visible':False, 'scaleanchor':'x', 'scaleratio':1 if st_t['type']=='grille' else None}, 
        plot_bgcolor=COLORS['bg'], showlegend=False
    )

    return fig, st_t, st_f, opts, f_val, max_s, s_val, step_content, f"Total Sondes : {max_s} | Pos : {s_val}", ALGO_DOCS.get(st_t['type'], ""), anim_disabled, btn_text

if __name__ == '__main__':
    app.run(debug=True)