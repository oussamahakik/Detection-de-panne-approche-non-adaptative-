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
    'accent': '#3498DB',     # Bleu Recherche
    'success': '#27AE60',    # Vert Succès
    'fail': '#E74C3C',       # Rouge Echec
    'edge_inactive': '#BDC3C7'
}

# --- DOCUMENTATION ALGORITHMIQUE (TEXTE ACADÉMIQUE) ---
ALGO_DOCS = {
    'grille': dcc.Markdown(r'''
        #### Algorithme 2 : Sondage Zigzag (Manhattan)
        **Principe :** Isolation par intersection d'axes et de chemins en boucle.
        
        1.  **Validation des Axes (Hubs)** :
            La *Ligne 0* et la *Colonne 0* sont testées en premier pour servir de "base saine".
        
        2.  **Procédure LTP (Logarithmic Testing Procedure) / Zigzag** :
            Pour tester une arête cible $e = ((r,c), (r, c+1))$ située au cœur de la grille :
            * La sonde part du NMS $(0,0)$.
            * Elle longe le Hub (Ligne 0) jusqu'à la colonne $c$.
            * Elle descend verticalement jusqu'à la ligne $r$.
            * **Elle traverse l'arête cible.**
            * Elle remonte par la colonne $c+1$ et rentre au NMS.
            
        **Diagnostic :**
        Si une sonde Zigzag échoue alors que les chemins d'accès (verticaux) sont sains, l'intersection logique isole l'arête horizontale unique.
    ''', mathjax=True),

    'lineaire': dcc.Markdown(r'''
        #### Algorithme 1 : Fenêtre Glissante (Sliding Window)
        **Principe :** Test de groupe séquentiel sur structure 1D.
        
        **Définition :**
        Soit un chemin $P_n$ avec les nœuds $\{0, 1, ..., n-1\}$.
        On définit une taille de fenêtre $w \approx \lceil n/2 \rceil$.
        
        **Construction des Sondes :**
        On génère un ensemble de sondes $S_j$ tel que chaque sonde couvre l'intervalle :
        $$ I_j = [j, \min(j+w, n-1)] $$
        Pour $j$ variant de $0$ à $n-w$.
        
        **Contrainte Physique :**
        Comme le contrôleur est en $0$, la sonde doit physiquement parcourir :
        $0 \to \dots \to j \to \dots \to (j+w) \to \dots \to 0$
        
        **Résultat :**
        Chaque arête du réseau appartient à un sous-ensemble unique de fenêtres. L'identification de la panne se fait par signature unique (recoupement des échecs).
    ''', mathjax=True),

    'complet': dcc.Markdown(r'''
        #### Algorithme 3 : Hub & Spoke (Étoile)
        **Principe :** Utilisation d'un nœud central de confiance.
        
        1.  **Phase Étoile :** On teste tous les liens directs $(0, i)$ partant du Hub.
        2.  **Phase Cycle :** Pour tester une arête distante $(u, v)$ (où $u,v \neq 0$) :
            On forme le cycle $0 \to u \to v \to 0$.
            
        Si les liens $(0,u)$ et $(v,0)$ sont validés par la Phase 1, tout échec de la sonde de Phase 2 impute nécessairement l'arête $(u,v)$.
    ''', mathjax=True),
    
    'arbre': dcc.Markdown(r'''
        #### Algorithme 4 : Sondage Racine-Vers-Feuille
        **Principe :** Exploitation de la hiérarchie parent-enfant.
        
        Pour chaque nœud $v$ du réseau (sauf la racine), on lance une sonde :
        $$ P_v : \text{Racine} \to \dots \to \text{Parent}(v) \to v \to \text{Racine} $$
        
        **Logique de Diagnostic :**
        Si la sonde vers le nœud $v$ échoue ($P_v$ Bloquée), mais que la sonde vers son parent réussit ($P_{parent(v)}$ OK), alors la panne est localisée sur l'arête $(\text{Parent}(v), v)$.
    ''', mathjax=True)
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
        """Chemin physique (Shortest Path)"""
        return nx.shortest_path(self.G, u, v)

    def _gen_probes(self):
        """Génération des sondes (Statique / Non-Adaptatif)"""
        self.probes = []
        
        # --- TOPOLOGIE LINÉAIRE (CORRIGÉE) ---
        if self.type == 'lineaire':
            # Taille de fenêtre optimale pour dis criminer toutes les arêtes
            # w = ceil(N/2) permet une intersection unique
            w = math.ceil(self.n / 2)
            
            # On fait glisser la fenêtre de 0 jusqu'à ce qu'elle touche la fin
            # Le pas de glissement est de 1 pour une transition fluide
            max_start = self.n - w if self.n - w > 0 else 1
            
            for start_node in range(max_start + 1):
                end_node = min(start_node + w, self.n - 1)
                
                # Construction PHYSIQUE du chemin :
                # 1. Aller du NMS (0) jusqu'au début de la fenêtre
                path_to_start = self._get_path(0, start_node)
                
                # 2. Traverser la fenêtre
                window_segment = list(range(start_node, end_node + 1))
                
                # 3. Revenir au NMS
                path_back = self._get_path(end_node, 0)
                
                # Fusion propre (éviter les doublons de nœuds aux jointures)
                full_path = path_to_start[:-1] + window_segment + path_back[1:]
                
                self.probes.append({
                    'type': 'WIN',
                    'path': full_path,
                    'name': f'Fenêtre [{start_node}, {end_node}]'
                })

        # --- TOPOLOGIE GRILLE (ZIGZAG) ---
        elif self.type == 'grille':
            s = int(math.sqrt(self.n))
            # Hubs
            self.probes.append({'type': 'AXIS', 'path': [(0,c) for c in range(s)], 'name': 'Ref Row 0'})
            self.probes.append({'type': 'AXIS', 'path': [(r,0) for r in range(s)], 'name': 'Ref Col 0'})
            # Zigzag H
            for r in range(1, s):
                for c in range(s-1):
                    p = self._get_path((0,0), (0,c)) + \
                        self._get_path((0,c), (r,c))[1:] + \
                        [(r, c+1)] + \
                        self._get_path((r,c+1), (0,c+1))[1:] + \
                        self._get_path((0,c+1), (0,0))[1:]
                    self.probes.append({'type': 'LTP', 'path': p, 'name': f'Zigzag H ({r},{c})-({r},{c+1})'})
            # Zigzag V
            for c in range(1, s):
                for r in range(s-1):
                    p = self._get_path((0,0), (r,0)) + \
                        self._get_path((r,0), (r,c))[1:] + \
                        [(r+1, c)] + \
                        self._get_path((r+1,c), (r+1,0))[1:] + \
                        self._get_path((r+1,0), (0,0))[1:]
                    self.probes.append({'type': 'LTP', 'path': p, 'name': f'Zigzag V ({r},{c})-({r+1},{c})'})

        # --- COMPLET ---
        elif self.type == 'complet':
            for i in range(1, self.n):
                self.probes.append({'type': 'STAR', 'path': [0,i,0], 'name': f'Hub-{i}'})
            for u in range(1, self.n):
                for v in range(u+1, self.n):
                    self.probes.append({'type': 'CYCLE', 'path': [0,u,v,0], 'name': f'Cycle {u}-{v}'})

        # --- ARBRE ---
        elif self.type == 'arbre':
            for i in range(1, self.n):
                path = self._get_path(0, i)
                self.probes.append({'type': 'TREE', 'path': path + path[-2::-1], 'name': f'Branche 0->{i}'})

    def diagnose(self, fault):
        if not fault: return [], []
        fsig = str(tuple(sorted(fault, key=str)))
        
        results = []
        suspects = set()
        for u,v in self.G.edges(): suspects.add(str(tuple(sorted((u,v), key=str))))
        
        for probe in self.probes:
            path = probe['path']
            failed = False
            psigs = set()
            for i in range(len(path)-1):
                sig = str(tuple(sorted((path[i], path[i+1]), key=str)))
                psigs.add(sig)
                if sig == fsig: failed = True
            
            results.append({'probe': probe, 'failed': failed})
            if failed: suspects = suspects.intersection(psigs)
            else: suspects = suspects.difference(psigs)
            
        return results, list(suspects)

# --- 2. INTERFACE DASH ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div(style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'fontFamily': 'Georgia, serif', 'padding': '20px'}, children=[
    
    # En-tête Académique
    html.Div(style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}, children=[
        html.H2("Optical Network Fault Diagnosis Workbench", style={'color': COLORS['text'], 'textAlign': 'center', 'fontWeight': 'bold'}),
        html.Div("Simulation of Non-Adaptive Group Testing Algorithms", style={'textAlign': 'center', 'color': '#7F8C8D', 'fontStyle': 'italic'})
    ]),

    html.Div(className='row', children=[
        # Colonne Gauche : Contrôles & Explications
        html.Div(className='four columns', children=[
            
            # Carte Configuration
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '5px', 'border': '1px solid #ddd', 'marginBottom': '20px'}, children=[
                html.H6("1. Topology & Algorithm", style={'fontWeight': 'bold', 'color': COLORS['accent']}),
                dcc.Dropdown(id='topo', options=[
                    {'label': 'Linear Path (Sliding Window)', 'value': 'lineaire'},
                    {'label': 'Grid Manhattan (Zigzag)', 'value': 'grille'},
                    {'label': 'Complete Graph (Hub)', 'value': 'complet'},
                    {'label': 'Random Tree (Branch)', 'value': 'arbre'}
                ], value='lineaire', clearable=False),
                
                html.Label("Network Size (Nodes):", style={'marginTop': '10px'}),
                dcc.Slider(id='n-slider', min=5, max=25, step=1, value=10, marks={5:'5', 10:'10', 16:'16', 25:'25'}),
                html.Button("Initialize Network", id='btn-build', style={'width':'100%', 'marginTop':'15px', 'backgroundColor': COLORS['accent'], 'color': 'white'}),
            ]),

            # Carte Explication Algorithmique (DYNAMIQUE)
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '5px', 'borderLeft': '5px solid #2C3E50'}, children=[
                html.H6("Algorithm Details", style={'fontWeight': 'bold', 'marginTop': '0'}),
                html.Div(id='algo-doc', style={'fontSize': '13px', 'lineHeight': '1.5', 'textAlign': 'justify'})
            ]),

            # Carte Panne
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '5px', 'border': '1px solid #ddd', 'marginTop': '20px'}, children=[
                html.H6("2. Fault Injection", style={'fontWeight': 'bold', 'color': COLORS['fail']}),
                dcc.Dropdown(id='fault', placeholder="Select edge to break...", searchable=True),
                html.Button("Run Diagnosis", id='btn-diag', style={'width':'100%', 'marginTop':'15px', 'backgroundColor': COLORS['success'], 'color': 'white'}),
            ])
        ]),

        # Colonne Droite : Visualisation
        html.Div(className='eight columns', children=[
            # Bannière Résultat
            html.Div(id='res-banner', style={'display': 'none', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '15px', 'fontWeight': 'bold', 'textAlign': 'center'}),
            
            # Graphe
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd', 'height': '60vh'}, children=[
                dcc.Graph(id='graph', style={'height': '100%'})
            ]),

            # Visualiseur Séquence
            html.Div(style={'backgroundColor': COLORS['card'], 'marginTop': '15px', 'padding': '15px', 'borderRadius': '5px', 'border': '1px solid #ddd'}, children=[
                html.H6("Probe Sequence Visualizer", style={'margin': '0 0 10px 0', 'fontWeight': 'bold'}),
                dcc.Slider(id='sim-slider', min=1, max=1, step=1, value=1),
                html.Div(id='sim-info', style={'textAlign': 'center', 'marginTop': '5px', 'fontWeight': 'bold', 'fontFamily': 'monospace', 'color': COLORS['accent']})
            ])
        ])
    ]),
    
    dcc.Store(id='st-topo'), dcc.Store(id='st-fault')
])

@app.callback(
    [Output('graph', 'figure'), Output('st-topo', 'data'), Output('st-fault', 'data'),
     Output('fault', 'options'), Output('fault', 'value'),
     Output('res-banner', 'children'), Output('res-banner', 'style'),
     Output('sim-slider', 'max'), Output('sim-info', 'children'),
     Output('algo-doc', 'children')],
    [Input('btn-build', 'n_clicks'), Input('graph', 'clickData'), Input('fault', 'value'),
     Input('btn-diag', 'n_clicks'), Input('sim-slider', 'value')],
    [State('topo', 'value'), State('n-slider', 'value'), State('st-topo', 'data'), State('st-fault', 'data')]
)
def update(b_build, click, f_val, b_diag, s_val, topo, n, st_t, st_f):
    ctx_id = ctx.triggered_id
    fig = go.Figure()

    # 1. Init
    if ctx_id == 'btn-build' or not st_t:
        st_t = {'type': topo, 'n': n}; st_f = None; f_val = None
    
    # Fix Grid Tuples
    if st_f and st_t['type'] == 'grille':
        try: st_f = sorted((tuple(st_f[0]), tuple(st_f[1])), key=str) if isinstance(st_f[0], list) else st_f
        except: pass

    eng = NetworkEngine(st_t['n'], st_t['type'])
    opts = [{'label': f"Edge {u}-{v}", 'value': str(sorted((u, v), key=str))} for u,v in eng.G.edges()]
    
    # Doc Text
    doc_text = ALGO_DOCS.get(st_t['type'], "No documentation available.")

    # 2. Fault Input
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

    # 3. Layout Calc
    pos = {}
    if st_t['type'] == 'grille': 
        for node in eng.G.nodes(): pos[node] = (node[1], -node[0])
    elif st_t['type'] == 'arbre': pos = nx.kamada_kawai_layout(eng.G)
    elif st_t['type'] == 'complet': pos = nx.circular_layout(eng.G)
    else: pos = nx.spectral_layout(eng.G)

    # 4. Draw Edges
    fsig = str(st_f) if st_f else ""
    for u, v in eng.G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        is_f = (str(sorted((u,v), key=str)) == fsig)
        c = COLORS['fail'] if is_f else COLORS['edge_inactive']
        w = 4 if is_f else 1.5
        fig.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(color=c, width=w), hoverinfo='text', text=f"{u}-{v}", customdata=[[u,v],[u,v],[u,v]]))

    # 5. Simulation Logic
    res_ban = ""; res_sty = {'display': 'none'}
    sim_info = "Waiting for simulation..."; max_s = 1

    if st_f:
        res, susp = eng.diagnose(st_f)
        max_s = len(res)
        
        # Result Banner
        if len(susp) == 1:
            res_ban = f"✅ FAULT ISOLATED: {susp[0]}"; res_sty = {'display': 'block', 'backgroundColor': '#D4EFDF', 'color': '#196F3D', 'border': '1px solid #196F3D'}
        elif len(susp) > 1:
            res_ban = f"⚠️ AMBIGUITY: {len(susp)} candidates."; res_sty = {'display': 'block', 'backgroundColor': '#FCF3CF', 'color': '#9A7D0A', 'border': '1px solid #9A7D0A'}
        else:
            res_ban = "❌ ERROR: No fault found."; res_sty = {'display': 'block', 'backgroundColor': '#FADBD8', 'color': '#922B21', 'border': '1px solid #922B21'}

        # Slider Viz
        if ctx_id == 'btn-diag' or ctx_id == 'sim-slider':
            idx = s_val - 1
            if idx < len(res):
                r = res[idx]
                col = COLORS['fail'] if r['failed'] else COLORS['success']
                stat = "BLOCKED ❌" if r['failed'] else "CLEAR ✅"
                sim_info = f"Probe {s_val}/{max_s}: {r['probe']['name']} -> {stat}"
                
                path = r['probe']['path']
                px = [pos[n][0] for n in path]; py = [pos[n][1] for n in path]
                
                fig.add_trace(go.Scatter(x=px, y=py, mode='lines', line=dict(width=3, color=col), name='Probe'))
                
                # Séquence Numérotée (Correcte)
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    mx = (pos[u][0] + pos[v][0]) / 2
                    my = (pos[u][1] + pos[v][1]) / 2
                    
                    fig.add_trace(go.Scatter(
                        x=[mx], y=[my], mode='markers+text',
                        text=[str(i+1)], 
                        textposition='middle center',
                        textfont=dict(color=col, size=11, weight='bold'),
                        marker=dict(size=20, color='white', line=dict(color=col, width=2)),
                        hoverinfo='skip'
                    ))

    # 6. Draw Nodes
    nxp = [pos[n][0] for n in eng.G.nodes()]; nyp = [pos[n][1] for n in eng.G.nodes()]
    fig.add_trace(go.Scatter(x=nxp, y=nyp, mode='markers+text', text=[str(n) for n in eng.G.nodes()], textposition="top center", marker=dict(size=14, color=COLORS['text'], line=dict(width=1, color='white')), hoverinfo='none'))

    ratio = 1 if st_t['type'] == 'grille' else None
    fig.update_layout(showlegend=False, margin=dict(l=20,r=20,t=20,b=20), xaxis={'visible':False}, yaxis={'visible':False, 'scaleanchor':'x', 'scaleratio':ratio})

    return fig, st_t, st_f, opts, f_val, res_ban, res_sty, max_s, sim_info, doc_text

if __name__ == '__main__':
    app.run(debug=True)