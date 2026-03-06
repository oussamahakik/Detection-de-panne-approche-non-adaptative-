import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import networkx as nx
import math
import ast

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
        1. Validation des axes de référence (Col 0, Ligne 0).
        2. Test global des lignes entières.
        3. Test vertical simultané (échelle) de la k-ième arête de toutes les lignes.
        4. Répétition pour les colonnes.
    '''),
    'lineaire': dcc.Markdown(r'''
        #### Algorithme 1 : Fenêtre Glissante
        **Stratégie :** Recouvrement Progressif.
        Une fenêtre de taille $W \approx N/2$ se déplace le long de la ligne.
        * Chaque sonde couvre un sous-ensemble contigu.
        * L'intersection des échecs permet d'isoler le lien fautif.
    ''', mathjax=True),
    'complet': dcc.Markdown(r'''
        #### Algorithme 3 : Hub & Spoke
        **Stratégie :** Centre vers Périphérie.
        1. **Phase Étoile :** Validation de tous les liens connectés au Hub (Nœud 0).
        2. **Phase Cycle :** Validation des liens distants $(u,v)$ via le chemin $0 \to u \to v \to 0$.
    ''', mathjax=True),
    'arbre': dcc.Markdown(r'''
        #### Algorithme 4 : Profondeur (Branche)
        **Stratégie :** Racine vers Feuilles.
        Le contrôleur (Racine) envoie une sonde vers chaque nœud du réseau.
        * Si la sonde vers le père $u$ passe, mais celle vers le fils $v$ échoue, le lien $(u,v)$ est en panne.
    ''', mathjax=True)
}

# --- MOTEUR ALGORITHMIQUE UNIFIÉ ---
def _distance_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _build_probe_step_annotations(path, pos, color):
    """
    Génère les annotations chiffrées pour chaque étape de la sonde.
    Intègre une détection de collision et s'adapte à l'échelle du graphe.
    """
    if len(path) < 2:
        return []

    occupied = [(pos[n][0], pos[n][1]) for n in pos]
    placed = []
    annotations = []

    # 1. Calcul de l'échelle moyenne pour adapter l'offset
    edge_lengths = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        dx = pos[v][0] - pos[u][0]
        dy = pos[v][1] - pos[u][1]
        edge_lengths.append(math.sqrt(dx*dx + dy*dy))
        
    avg_len = sum(edge_lengths) / len(edge_lengths) if edge_lengths else 1.0
    base_offset = avg_len * 0.18
    min_dist_sq = (base_offset * 0.85) ** 2

    for idx in range(len(path) - 1):
        u, v = path[idx], path[idx + 1]
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0

        dx = x1 - x0
        dy = y1 - y0
        norm = math.sqrt(dx * dx + dy * dy)
        
        if norm == 0:
            nxn, nyn = 0.0, 1.0
        else:
            nxn, nyn = -dy / norm, dx / norm

        chosen = None
        multipliers = [1.2, -1.2, 2.2, -2.2, 3.5, -3.5]
        
        for mult in multipliers:
            cx = mx + nxn * base_offset * mult
            cy = my + nyn * base_offset * mult
            
            if all(_distance_sq((cx, cy), p) >= min_dist_sq for p in occupied + placed):
                chosen = (cx, cy)
                break
                
        if chosen is None:
            chosen = (mx + nxn * base_offset * 4.5, my + nyn * base_offset * 4.5)

        placed.append(chosen)
        annotations.append(
            dict(
                x=chosen[0],
                y=chosen[1],
                xref="x",
                yref="y",
                text=f"<b>{idx + 1}</b>",
                showarrow=False,
                font=dict(size=11, color=color, family="Arial, sans-serif"),
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor=color,
                borderwidth=1.5,
                borderpad=3,
                opacity=1.0,
                xanchor="center",
                yanchor="middle",
                captureevents=False
            )
        )

    return annotations


class NetworkEngine:
    def __init__(self, n, topo):
        self.n = n
        self.type = topo
        self.G = self._build_graph()
        self.probes = []
        
        # Génération selon la topologie
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
            'path': path,
            'step_id': step_id,
            'step_name': step_name,
            'description': description,
            'id': len(self.probes) + 1
        })

    # --- GENERATEURS DE SONDES ---

    def _generate_linear_algo(self):
        w = math.ceil(self.n / 2)
        windows = []
        # Calcul des fenêtres
        for start in range(self.n):
            end = start + w
            if end >= self.n:
                last_start = max(0, self.n - 1 - w)
                last_end = self.n - 1
                if [last_start, last_end] not in windows: windows.append([last_start, last_end])
                break
            windows.append([start, end])
        
        # Création des sondes
        for idx, (start, end) in enumerate(windows):
            # AJOUT ICI : Création d'un aller-retour sur la fenêtre glissante
            path_aller = list(range(start, end + 1))
            path_retour = path_aller[-2::-1] if len(path_aller) > 1 else []
            full_path = path_aller + path_retour
            
            self._add_probe(full_path, 
                            f"WIN-{idx+1}", 
                            "Fenêtre Glissante", 
                            f"Test direct du segment couvrant les nœuds {start} à {end} (Aller-Retour).")

    def _generate_complete_algo(self):
        # 1. Phase Hub (Star)
        for i in range(1, self.n):
            self._add_probe([0, i, 0], 
                            "STAR", 
                            "Validation Hub", 
                            f"Test du lien direct (Rayon) entre le Hub 0 et le Nœud {i}.")
            
        # 2. Phase Cycles
        for u in range(1, self.n):
            for v in range(u+1, self.n):
                self._add_probe([0, u, v, 0], 
                                "CYCLE", 
                                "Triangulation", 
                                f"Test du lien distant {u}-{v} via le cycle 0->{u}->{v}->0.")

    def _generate_tree_algo(self):
        nodes_by_depth = sorted(self.G.nodes(), key=lambda x: len(self._get_path(0, x)))
        
        for i in nodes_by_depth:
            if i == 0: continue
            path = self._get_path(0, i)
            full_path = path + path[-2::-1]
            
            self._add_probe(full_path, 
                            "BRANCH", 
                            "Sondage Profondeur", 
                            f"Validation de la branche complète jusqu'au nœud {i}.")

    def _generate_grid_algo(self):
        s = int(math.sqrt(self.n))
        origin = (0,0)

        # 1a. Col 0 Globale
        path = [(r, 0) for r in range(s)] + [(r, 0) for r in range(s-2, -1, -1)]
        self._add_probe(path, "1a", "Axe Vertical", "Test global de la Colonne 0.")

        # 1b. Col 0 Unitaire
        for r in range(s-1):
            p = self._get_path(origin, (r,0)) + [(r,1), (r+1,1)] + self._get_path((r+1,0), origin)
            self._add_probe(p, "1b", "Détail Col 0", f"Test unitaire arête ({r},0)-({r+1},0).")

        # 2a. Row 0 Globale
        path = [(0, c) for c in range(s)] + [(0, c) for c in range(s-2, -1, -1)]
        self._add_probe(path, "2a", "Axe Horizontal", "Test global de la Ligne 0.")

        # 2b. Row 0 Unitaire
        for c in range(s-1):
            p = self._get_path(origin, (0,c)) + [(1,c), (1,c+1)] + self._get_path((0,c+1), origin)
            self._add_probe(p, "2b", "Détail Row 0", f"Test unitaire arête (0,{c})-(0,{c+1}).")

        # 3a. Lignes Globales
        for r in range(1, s):
            p = self._get_path(origin, (r,0)) 
            p += [(r, c) for c in range(1, s)] 
            p += [(r, c) for c in range(s-2, -1, -1)] 
            p += self._get_path((r,0), origin)[1:]
            self._add_probe(p, "3a", "Ligne Complète", f"Test global de la Ligne {r}.")

        # 3b. Échelle Verticale
        for k in range(s-1):
            path = self._get_path(origin, (0,k))
            for r in range(1, s):
                if r % 2 != 0: path += self._get_path(path[-1], (r, k))[1:] + [(r, k+1)]
                else: path += self._get_path(path[-1], (r, k+1))[1:] + [(r, k)]
            path += self._get_path(path[-1], origin)[1:]
            self._add_probe(path, "3b", "Échelle Verticale", f"Test simultané de la {k+1}ème arête de TOUTES les lignes.")

        # 4a. Colonnes Globales
        for c in range(1, s):
            p = self._get_path(origin, (0,c))
            p += [(r, c) for r in range(1, s)]
            p += [(r, c) for r in range(s-2, -1, -1)]
            p += self._get_path((0,c), origin)[1:]
            self._add_probe(p, "4a", "Colonne Complète", f"Test global de la Colonne {c}.")

        # 4b. Échelle Horizontale
        for k in range(s-1):
            path = self._get_path(origin, (k,0))
            for c in range(1, s):
                if c % 2 != 0: path += self._get_path(path[-1], (k, c))[1:] + [(k+1, c)]
                else: path += self._get_path(path[-1], (k+1, c))[1:] + [(k, c)]
            path += self._get_path(path[-1], origin)[1:]
            self._add_probe(path, "4b", "Échelle Horizontale", f"Test simultané de la {k+1}ème arête de TOUTES les colonnes.")

    # --- EXECUTION ---
    def run_simulation(self, fault):
        fsig = str(tuple(sorted(fault, key=str))) if fault else ""
        results = []
        for probe in self.probes:
            path = probe['path']
            failed = False
            for i in range(len(path)-1):
                sig = str(tuple(sorted((path[i], path[i+1]), key=str)))
                if sig == fsig: failed = True
            results.append({'probe': probe, 'failed': failed})
        return results

# --- INTERFACE DASH ---
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'fontFamily': 'Segoe UI, sans-serif', 'padding': '20px'}, children=[
    
    html.Div(style={'textAlign': 'center', 'marginBottom': '30px'}, children=[
        html.H1("Simulateur de Diagnostic Optique", style={'color': COLORS['text'], 'marginBottom': '5px'}),
        html.Div("Comparaison des stratégies de sondage par topologie", style={'color': '#7F8C8D'})
    ]),

    html.Div(className='row', style={'display': 'flex', 'gap': '20px'}, children=[
        
        # COLONNE GAUCHE
        html.Div(style={'flex': '1', 'maxWidth': '400px'}, children=[
            
            # Config
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}, children=[
                html.H3("1. Configuration", style={'marginTop': 0, 'color': COLORS['accent'], 'fontSize': '18px'}),
                
                html.Label("Topologie :"),
                dcc.Dropdown(id='topo', options=[
                    {'label': 'Grille (LTP Hiérarchique)', 'value': 'grille'},
                    {'label': 'Linéaire (Fenêtre Glissante)', 'value': 'lineaire'},
                    {'label': 'Complet (Hub & Cycles)', 'value': 'complet'},
                    {'label': 'Arbre (Profondeur)', 'value': 'arbre'}
                ], value='grille', clearable=False),

                html.Label("Taille du réseau (Nœuds) :", style={'marginTop': '10px'}),
                dcc.Slider(id='n-slider', min=5, max=25, step=1, value=16, marks={5:'5', 10:'10', 16:'16', 25:'25'}),
                
                html.Button("Générer Réseau", id='btn-build', style={'width': '100%', 'marginTop': '15px', 'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none', 'padding': '10px', 'borderRadius': '5px', 'cursor': 'pointer'}),
                html.Hr(),
                html.Label("2. Injecter une Panne :"),
                dcc.Dropdown(id='fault', placeholder="Sélectionner une arête...", searchable=True),
            ]),

            # Narratif
            html.Div(style={'backgroundColor': COLORS['step_box'], 'padding': '20px', 'borderRadius': '10px', 'border': f'2px solid {COLORS["accent"]}'}, children=[
                html.H3("Status de l'Algorithme", style={'marginTop': 0, 'color': COLORS['text'], 'fontSize': '18px', 'borderBottom': '1px solid #ccc', 'paddingBottom': '10px'}),
                html.Div(id='step-display', children=[html.P("En attente de simulation...", style={'color': '#7F8C8D'})])
            ]),
            
            # Doc
            html.Div(style={'marginTop': '20px', 'fontSize': '13px', 'color': '#555'}, children=[
                html.Div(id='algo-doc')
            ])
        ]),

        # COLONNE DROITE
        html.Div(style={'flex': '2'}, children=[
            # Graphique
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'height': '600px'}, children=[
                dcc.Graph(id='graph', style={'height': '100%'}, config={'displayModeBar': False})
            ]),

            # Contrôles
            html.Div(style={'backgroundColor': COLORS['card'], 'marginTop': '20px', 'padding': '20px', 'borderRadius': '10px', 'display': 'flex', 'alignItems': 'center', 'gap': '15px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}, children=[
                html.Button("▶️ Lecture", id='btn-play', style={'backgroundColor': COLORS['success'], 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold'}),
                html.Div(style={'flex': '1'}, children=[
                    dcc.Slider(id='sim-slider', min=1, max=1, step=1, value=1, tooltip={"placement": "bottom", "always_visible": True}, marks=None)
                ]),
                html.Div(id='counter-display', style={'fontWeight': 'bold', 'color': COLORS['text'], 'minWidth': '80px', 'textAlign': 'right'})
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
     Input('anim-interval', 'n_intervals')],
    [State('topo', 'value'), 
     State('n-slider', 'value'), 
     State('st-topo', 'data'), 
     State('st-fault', 'data')]
)
def update_simulation(b_build, click, f_val, s_val, b_play, n_intervals, topo, n_nodes, st_t, st_f):
    ctx_id = ctx.triggered_id
    fig = go.Figure()
    
    # --- INIT / RESET ---
    anim_disabled = True
    btn_text = "▶️ Lecture"
    
    # Init Topologie
    if ctx_id == 'btn-build' or not st_t:
        if topo == 'grille':
            s = int(math.sqrt(n_nodes))
            n_final = s * s
        else:
            n_final = n_nodes
            
        st_t = {'n': n_final, 'type': topo}
        st_f = None
        f_val = None
        s_val = 1
    
    # Gestion Panne
    if st_f and isinstance(st_f[0], list):
         st_f = sorted((tuple(st_f[0]), tuple(st_f[1])), key=str)

    # --- MOTEUR ---
    eng = NetworkEngine(st_t['n'], st_t['type'])
    opts = [{'label': f"Lien {u} - {v}", 'value': str(sorted((u, v), key=str))} for u,v in eng.G.edges()]
    doc_text = ALGO_DOCS.get(st_t['type'], "")

    # --- INTERACTIONS ---
    if ctx_id == 'graph' and click:
        try:
            p = click['points'][0]['customdata']
            if st_t['type'] == 'grille':
                st_f = sorted([tuple(p[0]), tuple(p[1])], key=str)
            else:
                st_f = sorted(p, key=str)
            f_val = str(st_f)
            s_val = 1
        except: pass
    elif ctx_id == 'fault' and f_val:
        try:
            v = ast.literal_eval(f_val)
            if st_t['type'] == 'grille':
                st_f = sorted([tuple(v[0]), tuple(v[1])], key=str)
            else:
                st_f = sorted(v, key=str)
            s_val = 1
        except: pass

    # --- SIMULATION ---
    results = eng.run_simulation(st_f)
    max_s = len(results)
    if s_val is None: s_val = 1

    # Animation
    if ctx_id == 'btn-play':
        anim_disabled = False
        if s_val >= max_s: s_val = 1
    elif ctx_id == 'anim-interval':
        if s_val < max_s:
            s_val += 1
            if results[s_val - 1]['failed']:
                anim_disabled = True
                btn_text = "▶️ Reprendre"
            else:
                anim_disabled = False
                btn_text = "⏸️ Stop"
        else:
            anim_disabled = True
            btn_text = "↺ Reset"
    
    if anim_disabled:
        if s_val >= max_s: btn_text = "↺ Reset"
        elif s_val > 1 and results[s_val-1]['failed']: btn_text = "▶️ Reprendre"
        else: btn_text = "▶️ Lecture"
    else:
        btn_text = "⏸️ Stop"

    # --- VISUALISATION ---
    pos = {}
    if st_t['type'] == 'lineaire':
        for node in eng.G.nodes(): pos[node] = (node, 0)
    elif st_t['type'] == 'grille': 
        for node in eng.G.nodes(): pos[node] = (node[1], -node[0])
    elif st_t['type'] == 'arbre': pos = nx.kamada_kawai_layout(eng.G)
    elif st_t['type'] == 'complet': pos = nx.circular_layout(eng.G)

    # Fond
    edge_x, edge_y = [], []
    for u, v in eng.G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        fig.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(width=15, color='rgba(0,0,0,0)'), hoverinfo='text', text=f"Lien {u}-{v}", customdata=[[u,v],[u,v],[u,v]], showlegend=False))
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color=COLORS['edge_inactive'], width=2), hoverinfo='skip'))

    # Panne
    if st_f:
        try:
            u_f, v_f = st_f
            fx0, fy0 = pos[u_f]; fx1, fy1 = pos[v_f]
            fig.add_trace(go.Scatter(x=[fx0, fx1], y=[fy0, fy1], mode='lines', line=dict(color=COLORS['fail'], width=4, dash='dot'), name='Panne'))
            fig.add_trace(go.Scatter(x=[(fx0+fx1)/2], y=[(fy0+fy1)/2], mode='markers', marker=dict(symbol='x', size=15, color=COLORS['fail'])))
        except: pass

    # Sonde
    current_res = results[s_val - 1]
    probe_data = current_res['probe']
    is_failed = current_res['failed']
    path = probe_data['path']
    col = COLORS['fail'] if is_failed else COLORS['success']
    
    px, py = [], []
    for node in path:
        px.append(pos[node][0])
        py.append(pos[node][1])
    
    fig.add_trace(go.Scatter(x=px, y=py, mode='lines', line=dict(width=4, color=col), opacity=0.9, name='Sonde'))
    
    # Annotations uniquement pour grille et linéaire
    if st_t['type'] in ['grille', 'lineaire']:
        step_annotations = _build_probe_step_annotations(path, pos, col)
    else:
        step_annotations = []
    
    # Flèches
    if len(path) > 1:
        mid = len(path) // 2
        fig.add_trace(go.Scatter(x=[px[0]], y=[py[0]], mode='markers', marker=dict(size=12, color=col, symbol='circle')))
        fig.add_trace(go.Scatter(x=[px[mid]], y=[py[mid]], mode='markers', marker=dict(size=10, color=col, symbol='triangle-up'), showlegend=False))

    # Nœuds
    nxp = [pos[n][0] for n in eng.G.nodes()]
    nyp = [pos[n][1] for n in eng.G.nodes()]
    fig.add_trace(go.Scatter(x=nxp, y=nyp, mode='markers+text', text=[str(n) for n in eng.G.nodes()], textposition="top center", textfont=dict(color=COLORS['text'], size=10), marker=dict(size=15, color='white', line=dict(width=2, color=COLORS['text'])), hoverinfo='none'))

    y_range = [-0.5, 0.5] if st_t['type'] == 'lineaire' else None
    ratio = 1 if st_t['type'] == 'grille' else None

    # Injection des annotations
    fig.update_layout(
        margin=dict(l=20,r=20,t=20,b=20), 
        xaxis={'visible':False, 'fixedrange': True}, 
        yaxis={'visible':False, 'scaleanchor':'x', 'scaleratio':ratio, 'range': y_range, 'fixedrange': True}, 
        plot_bgcolor=COLORS['bg'],
        showlegend=False,
        annotations=step_annotations
    )

    # --- NARRATIF ---
    status_icon = "❌ ÉCHEC" if is_failed else "✅ SUCCÈS"
    status_color = COLORS['fail'] if is_failed else COLORS['success']

    step_content = [
        html.Div([
            html.Span("ÉTAPE : ", style={'fontWeight': 'bold', 'color': '#7F8C8D', 'fontSize': '12px'}),
            html.Span(f"{probe_data['step_id']} - {probe_data['step_name']}", style={'fontWeight': 'bold', 'fontSize': '16px', 'color': COLORS['accent']})
        ], style={'marginBottom': '10px'}),
        
        html.Div([
            html.Span("DESCRIPTION : ", style={'fontWeight': 'bold', 'color': '#7F8C8D', 'fontSize': '12px'}),
            html.P(probe_data['description'], style={'fontSize': '14px', 'margin': '5px 0'})
        ], style={'marginBottom': '15px'}),
        
        html.Div(style={'backgroundColor': 'white', 'padding': '10px', 'borderRadius': '5px', 'borderLeft': f'5px solid {status_color}'}, children=[
            html.Div([
                html.Span("STATUT : ", style={'fontWeight': 'bold', 'fontSize': '12px'}),
                html.Span(status_icon, style={'fontWeight': 'bold', 'color': status_color, 'fontSize': '14px'})
            ])
        ])
    ]

    return fig, st_t, st_f, opts, f_val, max_s, s_val, step_content, f"Sonde {s_val}/{max_s}", doc_text, anim_disabled, btn_text

if __name__ == '__main__':
    app.run(debug=True)