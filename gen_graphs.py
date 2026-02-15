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
    'accent': '#3498DB',     
    'success': '#27AE60',    
    'fail': '#E74C3C',       
    'edge_inactive': '#BDC3C7',
    'window_highlight': 'rgba(52, 152, 219, 0.2)' 
}

# --- DOCUMENTATION ---
ALGO_DOCS = {
    'grille': dcc.Markdown(r'''
        #### Zigzag (Manhattan)
        **Stratégie :**
        1. Validation des axes (Ligne 0, Col 0).
        2. Balayage en Zigzag pour tester les liens internes.
    '''),
    'lineaire': dcc.Markdown(r'''
        #### Fenêtre Glissante
        **Stratégie :**
        Une fenêtre de taille $N/2$ se déplace de gauche à droite.
        * **Numérotation** : Ordre relatif à l'intérieur de la fenêtre (1, 2, 3...).
    ''', mathjax=True),
    'complet': dcc.Markdown(r'''
        #### Hub & Spoke + Cycles
        **Stratégie :**
        1. **Hub (Star)** : Test des liens connectés au nœud 0.
        2. **Cycles** : Test des liens distants $(u, v)$ via le chemin $0 \to u \to v \to 0$.
    ''', mathjax=True),
    'arbre': dcc.Markdown(r'''#### Diagnostic Arbre''')
}

# --- 1. MOTEUR ALGORITHMIQUE ---
class NetworkEngine:
    def __init__(self, n, topo):
        self.n = n
        self.type = topo
        self.G = self._build_graph()
        self.probes = []
        self._gen_probes()

    def _build_graph(self):
        if self.type == 'lineaire': return nx.path_graph(self.n)
        elif self.type == 'complet': return nx.complete_graph(self.n)
        elif self.type == 'arbre': return nx.random_tree(self.n, seed=42)
        elif self.type == 'grille':
            s = int(math.sqrt(self.n))
            return nx.grid_2d_graph(s, s)
        return nx.Graph()

    def _get_path(self, u, v):
        return nx.shortest_path(self.G, u, v)

    def _gen_probes(self):
        self.probes = []
        
        # --- LINEAIRE ---
        if self.type == 'lineaire':
            w = math.ceil(self.n / 2)
            windows = []
            for start in range(self.n):
                end = start + w
                if end >= self.n:
                    last_start = max(0, self.n - 1 - w)
                    last_end = self.n - 1
                    if [last_start, last_end] not in windows:
                        windows.append([last_start, last_end])
                    break
                windows.append([start, end])
            
            for idx, (start, end) in enumerate(windows):
                path_aller = self._get_path(0, start)
                segment_window = list(range(start + 1, end + 1))
                path_retour = self._get_path(end, 0)
                
                # Construction du chemin sans doublons aux jonctions
                if start == 0:
                    full_path = [0] + segment_window + path_retour[1:]
                else:
                    full_path = path_aller + segment_window + path_retour[1:]
                
                self.probes.append({
                    'type': 'WIN',
                    'path': full_path,
                    'name': f'Fenêtre [{start}-{end}]',
                    'window_nodes': (start, end)
                })

        # --- GRILLE (ZIGZAG) ---
        elif self.type == 'grille':
            s = int(math.sqrt(self.n))
            # Axes de référence
            self.probes.append({'type': 'AXIS', 'path': [(0,c) for c in range(s)], 'name': 'Axe Ligne 0'})
            self.probes.append({'type': 'AXIS', 'path': [(r,0) for r in range(s)], 'name': 'Axe Colonne 0'})
            # Zigzags Horizontaux
            for r in range(1, s):
                for c in range(s-1):
                    # Chemin spécifique 0->...->(r,c)->(r,c+1)->...->0
                    p = self._get_path((0,0), (0,c)) + self._get_path((0,c), (r,c))[1:] + [(r, c+1)] + self._get_path((r,c+1), (0,c+1))[1:] + self._get_path((0,c+1), (0,0))[1:]
                    self.probes.append({'type': 'LTP', 'path': p, 'name': f'Zigzag H ({r},{c})'})
            # Zigzags Verticaux
            for c in range(1, s):
                for r in range(s-1):
                    p = self._get_path((0,0), (r,0)) + self._get_path((r,0), (r,c))[1:] + [(r+1, c)] + self._get_path((r+1,c), (r+1,0))[1:] + self._get_path((r+1,0), (0,0))[1:]
                    self.probes.append({'type': 'LTP', 'path': p, 'name': f'Zigzag V ({r},{c})'})

        # --- COMPLET (HUB & SPOKE) ---
        elif self.type == 'complet':
            # 1. Test de l'Etoile (Star)
            for i in range(1, self.n):
                self.probes.append({'type': 'STAR', 'path': [0,i,0], 'name': f'Hub vers {i}'})
            
            # 2. Test des Cycles (Liens distants)
            # On teste tous les liens (u, v) qui ne touchent pas 0
            for u in range(1, self.n):
                for v in range(u+1, self.n):
                    # Cycle 0 -> u -> v -> 0
                    self.probes.append({'type': 'CYCLE', 'path': [0, u, v, 0], 'name': f'Cycle {u}-{v}'})

        # --- ARBRE ---
        elif self.type == 'arbre':
            for i in range(1, self.n):
                path = self._get_path(0, i)
                # Aller-Retour simple sur la branche
                self.probes.append({'type': 'TREE', 'path': path + path[-2::-1], 'name': f'Branche vers {i}'})

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

# --- 2. INTERFACE DASH ---
mathjax_script = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML'
app = dash.Dash(__name__, external_scripts=[mathjax_script], external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div(style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'fontFamily': 'Segoe UI, sans-serif', 'padding': '20px'}, children=[
    html.Div(style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px'}, children=[
        html.H2("Visualiseur de Sondes Optiques", style={'color': COLORS['text'], 'textAlign': 'center', 'fontWeight': 'bold', 'margin': '0'}),
    ]),
    html.Div(className='row', children=[
        html.Div(className='four columns', children=[
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px'}, children=[
                html.Label("1. Configuration", style={'fontWeight': 'bold', 'color': COLORS['accent']}),
                dcc.Dropdown(id='topo', options=[
                    {'label': 'Linéaire (Fenêtre Glissante)', 'value': 'lineaire'},
                    {'label': 'Grille (Zigzag)', 'value': 'grille'},
                    {'label': 'Complet (Hub & Cycles)', 'value': 'complet'},
                    {'label': 'Arbre', 'value': 'arbre'}
                ], value='lineaire', clearable=False),
                
                html.Label("Taille du réseau:", style={'marginTop': '15px'}),
                dcc.Slider(id='n-slider', min=5, max=16, step=1, value=8, marks={5:'5', 8:'8', 12:'12', 16:'16'}),
                html.Button("Générer Réseau", id='btn-build', style={'width':'100%', 'marginTop':'20px', 'backgroundColor': COLORS['accent'], 'color': 'white', 'border': 'none'}),
            ]),
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px'}, children=[
                html.Label("2. Panne (Optionnel)", style={'fontWeight': 'bold', 'color': COLORS['fail']}),
                dcc.Dropdown(id='fault', placeholder="Simuler une coupure...", searchable=True),
            ]),
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '8px'}, children=[
                html.Div(id='algo-doc', style={'fontSize': '13px', 'textAlign': 'justify'})
            ]),
        ]),
        html.Div(className='eight columns', children=[
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '10px', 'borderRadius': '8px', 'height': '60vh'}, children=[
                dcc.Graph(id='graph', style={'height': '100%'}, config={'displayModeBar': False})
            ]),
            
            html.Div(style={'backgroundColor': COLORS['card'], 'marginTop': '15px', 'padding': '20px', 'borderRadius': '8px'}, children=[
                html.H6("Séquenceur de Sondes", style={'margin': '0 0 15px 0', 'fontWeight': 'bold', 'color': COLORS['text']}),
                # Slider avec valeur de retour gérée
                dcc.Slider(id='sim-slider', min=1, max=1, step=1, value=1, tooltip={"placement": "bottom", "always_visible": True}),
                html.Div(id='sim-info', style={'textAlign': 'center', 'marginTop': '15px', 'fontWeight': 'bold', 'fontSize': '16px', 'color': COLORS['accent']})
            ])
        ])
    ]),
    dcc.Store(id='st-topo'), dcc.Store(id='st-fault')
])

@app.callback(
    [Output('graph', 'figure'), Output('st-topo', 'data'), Output('st-fault', 'data'),
     Output('fault', 'options'), Output('fault', 'value'),
     Output('sim-slider', 'max'), Output('sim-slider', 'value'), # Ajout de 'value' ici pour reset
     Output('sim-info', 'children'), Output('algo-doc', 'children')],
    [Input('btn-build', 'n_clicks'), Input('graph', 'clickData'), Input('fault', 'value'),
     Input('sim-slider', 'value')],
    [State('topo', 'value'), State('n-slider', 'value'), State('st-topo', 'data'), State('st-fault', 'data')]
)
def update_all(b_build, click, f_val, s_val, topo, n, st_t, st_f):
    ctx_id = ctx.triggered_id
    fig = go.Figure()
    
    # --- 1. INITIALISATION ---
    # Si on clique sur "Construire" ou au démarrage -> On reset TOUT
    if ctx_id == 'btn-build' or not st_t: 
        st_t = {'type': topo, 'n': n}
        st_f = None
        f_val = None
        s_val = 1 # Force le slider à revenir à 1
    
    # Gestion panne Grille (tuples)
    if st_f and st_t['type'] == 'grille' and isinstance(st_f[0], list):
         st_f = sorted((tuple(st_f[0]), tuple(st_f[1])), key=str)

    # --- 2. MOTEUR & DONNÉES ---
    eng = NetworkEngine(st_t['n'], st_t['type'])
    opts = [{'label': f"Lien {u} - {v}", 'value': str(sorted((u, v), key=str))} for u,v in eng.G.edges()]
    doc_text = ALGO_DOCS.get(st_t['type'], "Pas de documentation.")

    # --- 3. GESTION INTERACTION CLICK / DROPDOWN ---
    if ctx_id == 'graph' and click:
        try:
            p = click['points'][0]['customdata']
            if st_t['type'] == 'grille': p = [tuple(p[0]), tuple(p[1])]
            st_f = sorted(p, key=str); f_val = str(st_f)
        except: pass
    elif ctx_id == 'fault' and f_val:
        try:
            v = ast.literal_eval(f_val)
            if st_t['type'] == 'grille': v = [tuple(v[0]), tuple(v[1])]
            st_f = sorted(v, key=str)
        except: pass

    # --- 4. CALCUL LAYOUT ---
    pos = {}
    if st_t['type'] == 'lineaire':
        for node in eng.G.nodes(): pos[node] = (node, 0)
    elif st_t['type'] == 'grille': 
        for node in eng.G.nodes(): pos[node] = (node[1], -node[0])
    elif st_t['type'] == 'arbre': pos = nx.kamada_kawai_layout(eng.G)
    elif st_t['type'] == 'complet': pos = nx.circular_layout(eng.G)
    else: pos = nx.spring_layout(eng.G)

    # --- 5. DESSIN DU FOND (ARÊTES) ---
    edge_x, edge_y = [], []
    for u, v in eng.G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        # Zone cliquable invisible large
        fig.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(width=10, color='rgba(0,0,0,0)'), hoverinfo='text', text=f"Lien {u}-{v}", customdata=[[u,v],[u,v],[u,v]], showlegend=False))
    
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color=COLORS['edge_inactive'], width=2), hoverinfo='skip'))

    # Panne (Croix rouge)
    if st_f:
        try:
            u_f, v_f = st_f
            fx0, fy0 = pos[u_f]; fx1, fy1 = pos[v_f]
            fig.add_trace(go.Scatter(x=[fx0, fx1], y=[fy0, fy1], mode='lines', line=dict(color=COLORS['fail'], width=4, dash='dot'), name='Panne'))
            fig.add_trace(go.Scatter(x=[(fx0+fx1)/2], y=[(fy0+fy1)/2], mode='markers', marker=dict(symbol='x', size=15, color=COLORS['fail'])))
        except: pass

    # --- 6. SIMULATION SONDES ---
    sim_info, max_s = "Aucune sonde générée", 1
    
    # Simulation de TOUTES les sondes
    all_results = eng.run_simulation(st_f)
    max_s = len(all_results)
    
    # Sécurité slider
    if s_val is None: s_val = 1
    current_idx = min(s_val, max_s) - 1
    
    if all_results:
        r = all_results[current_idx]
        
        status_text = "PASSE ✅"
        col = COLORS['success']
        if st_f and r['failed']:
            status_text = "ÉCHEC ❌"
            col = COLORS['fail']
        elif not st_f:
            status_text = "PASSE (Réseau Sain) ✅"
        
        sim_info = f"Sonde {current_idx + 1}/{max_s} : {r['probe']['name']} ➔ {status_text}"
        path = r['probe']['path']

        # Trace Sonde
        px, py = [], []
        text_x, text_y, text_val = [], [], []
        
        # Rectangle pour linéaire
        if 'window_nodes' in r['probe'] and st_t['type'] == 'lineaire':
            ws, we = r['probe']['window_nodes']
            fig.add_shape(type="rect",
                x0=ws - 0.4, x1=we + 0.4, y0=-0.3, y1=0.3,
                fillcolor=COLORS['window_highlight'], line=dict(width=0), layer="below"
            )

        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            px.extend([pos[u][0], pos[v][0], None])
            py.extend([pos[u][1], pos[v][1], None])
            
            # --- Etiquetage ---
            show_label = True
            lbl_txt = str(i + 1)
            
            # Logique spécifique Linéaire (Relatif + Masquage transport)
            if st_t['type'] == 'lineaire':
                start_win, end_win = r['probe']['window_nodes']
                is_in_window = (u >= start_win) and (v <= end_win)
                is_forward = (v > u)
                if is_in_window and is_forward:
                    lbl_txt = str(u - start_win + 1)
                else:
                    show_label = False 
            
            # Pour Complet/Arbre/Grille, on montre tout ou on peut simplifier
            # Ici on laisse tout pour les autres topos car c'est utile de voir le chemin

            if show_label:
                mx = (pos[u][0] + pos[v][0]) / 2
                base_my = (pos[u][1] + pos[v][1]) / 2
                offset = 0.2 if st_t['type'] == 'lineaire' else 0
                text_x.append(mx)
                text_y.append(base_my + offset)
                text_val.append(lbl_txt)

        fig.add_trace(go.Scatter(x=px, y=py, mode='lines', line=dict(width=3, color=col), opacity=0.8, name='Chemin Sonde'))
        fig.add_trace(go.Scatter(x=text_x, y=text_y, mode='markers+text', text=text_val, textposition='middle center', textfont=dict(color='white', size=10, weight='bold'), marker=dict(size=18, color=col, line=dict(color='white', width=1)), hoverinfo='skip'))

    # --- 7. NŒUDS ---
    nxp = [pos[n][0] for n in eng.G.nodes()]
    nyp = [pos[n][1] for n in eng.G.nodes()]
    fig.add_trace(go.Scatter(x=nxp, y=nyp, mode='markers+text', text=[str(n) for n in eng.G.nodes()], textposition="middle center", textfont=dict(color='black', weight='bold'), marker=dict(size=25 if st_t['type'] == 'lineaire' else 14, color='white', line=dict(width=2, color=COLORS['text'])), hoverinfo='none'))

    y_range = [-0.5, 0.5] if st_t['type'] == 'lineaire' else None
    ratio = 1 if st_t['type'] == 'grille' else None

    fig.update_layout(showlegend=False, margin=dict(l=20,r=20,t=20,b=20), xaxis={'visible':False, 'fixedrange': True}, yaxis={'visible':False, 'scaleanchor':'x', 'scaleratio':ratio, 'range': y_range, 'fixedrange': True}, plot_bgcolor=COLORS['bg'])
    
    # Retourne s_val dans l'Output pour forcer la mise à jour visuelle du slider
    return fig, st_t, st_f, opts, f_val, max_s, s_val, sim_info, doc_text

if __name__ == '__main__': app.run(debug=True)