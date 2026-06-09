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
        #### Algorithme 4 : Arbre (Heavy-Light Decomposition)
        **Stratégie Mathématique Exacte (Top-Down) :**
        1. **Poids :** Calcul du nombre de descendants par nœud.
        2. **HLD :** Création des "Chemins Préférés" via les arêtes lourdes.
        3. **Strates :** Sondage combinatoire découpé par *Profondeur Légère* (Strate 0, Strate 1...).
        4. **Variante :** L'article théorique utilise le LTP pour les profondeurs, mais la réalité physique impose parfois un balayage (Diagonale) pour éviter le masquage.
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

    if topo == 'arbre':
        if 'target_edges' in probe and probe['target_edges']:
            for u, v in probe['target_edges']: add_edge(u, v)
        else:
            outward_len = (len(path) + 1) // 2
            for i in range(max(0, outward_len - 1)): add_edge(path[i], path[i + 1])
        return highlighted_edges

    if topo == 'lineaire':
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
        html.H4(f"Matrice : {phase_name}", style={'marginTop': '0', 'marginBottom': '5px', 'fontSize': '13px', 'color': COLORS['text'], 'textTransform': 'uppercase'}),
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
    elif topo_type == 'arbre':
        if "Absolue" in phase_name:
            if ltp_meta.get('variant') == 'physique':
                m_explanation = "Éléments testés : Profondeurs d'éloignement (Matrice Diagonale Anti-Cascade)."
            else:
                m_explanation = "Éléments testés : Profondeurs d'éloignement (Théorie pure LTP de l'article)."
        elif "Strate" in phase_name: 
            m_explanation = "Éléments testés : Chemins Préférés appartenant à cette Strate de profondeur légère."
        elif "Chemins Préférés" in phase_name:
            m_explanation = "Éléments testés : Tous les Chemins Préférés de l'arbre (Théorie pure globale, O(log N))."
            
    if topo_type == 'arbre' and "Absolue" in phase_name and ltp_meta.get('variant') == 'physique':
        T = m
        formula = f"$T = m = {m}$ sondes."
        explanation_text = f"Matrice diagonale séquentielle de {T} lignes pour éviter l'effet de masquage physique du laser."
    else:
        T = math.ceil(math.log2(m + 1))
        formula = f"$T = \\lceil \\log_2(m + 1) \\rceil = {T}$ sondes."
        explanation_text = f"L'algorithme a généré une matrice compressée de {T} lignes pour identifier 1 défaillance parmi {m} éléments."

    return html.Div([
        html.H4("Détails Mathématiques (CGT)", style={'marginTop': '15px', 'marginBottom': '10px', 'fontSize': '14px', 'color': COLORS['accent'], 'borderBottom': '1px solid #eee', 'paddingBottom': '5px'}),
        html.Div([html.Strong("Taille du problème ($m$) : ", style={'fontSize': '12px'}), html.Span(f"{m} éléments", style={'fontSize': '12px'}), dcc.Markdown(m_explanation, mathjax=True, style={'fontSize': '11px', 'color': '#7F8C8D', 'marginTop': '2px', 'marginBottom': '8px'})]),
        html.Div([html.Strong("Nombre de sondes ($T$) : ", style={'fontSize': '12px'}), dcc.Markdown(formula, mathjax=True, style={'fontSize': '12px', 'display': 'inline-block', 'marginLeft': '5px'}), html.P(explanation_text, style={'fontSize': '11px', 'color': '#7F8C8D', 'marginTop': '2px', 'marginBottom': '0'})])
    ], style={'backgroundColor': '#fdfdfe', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #e0e0e0', 'marginTop': '15px'})

def _build_diagnosis_report(results, topo, mode, fault_exists):
    if not fault_exists:
        return html.Div([
            html.H4("🧠 Décodage du Contrôleur", style={'marginTop': 0, 'color': COLORS['success']}),
            html.P("Diagnostic : Réseau sain. Tous les syndromes sont nuls (0).")
        ], style={'backgroundColor': '#E8F8F5', 'padding': '15px', 'borderRadius': '5px', 'border': f"2px solid {COLORS['success']}", 'marginTop': '20px'})

    report_content = []

    if mode == 'ltp' and topo in ['complet', 'arbre', 'grille']:
        phases = {}
        for res in results:
            if 'ltp_meta' in res['probe'] and res['probe']['ltp_meta']:
                meta = res['probe']['ltp_meta']
                phase = meta['phase']
                if phase not in phases:
                    phases[phase] = {'bits': {}, 'items': meta['items'], 'type': meta.get('type'), 'ld': meta.get('ld', 0), 'variant': meta.get('variant')}
                phases[phase]['bits'][meta['active_test']] = 1 if res['failed'] else 0
        
        sorted_phases = []
        if topo == 'arbre':
            depth_phase = [p for p, d in phases.items() if d.get('type') == 'depth']
            path_phases = sorted([p for p, d in phases.items() if d.get('type') == 'path'], key=lambda x: phases[x].get('ld', 0))
            sorted_phases = depth_phase + path_phases
        else:
            sorted_phases = list(phases.keys())

        fault_found = False
        strate_stop = False
        
        for phase in sorted_phases:
            data = phases[phase]
            bits = data['bits']
            if not bits: continue
            
            if topo == 'arbre' and data.get('type') == 'path' and strate_stop:
                continue 
                
            bit_str = "".join(str(bits[i]) for i in sorted(bits.keys(), reverse=True))
            
            if data.get('type') == 'depth' and data.get('variant') == 'physique':
                first_fail = -1
                for idx in sorted(bits.keys()):
                    if bits[idx] == 1:
                        first_fail = idx
                        break
                if first_fail != -1:
                    fault_found = True
                    fault_item = data['items'][first_fail]
                    report_content.append(html.Div([
                        html.Strong(f"Analyse Séquentielle : {phase}", style={'color': COLORS['accent']}),
                        html.Ul([
                            html.Li(f"Vecteur de balayage lu : {bit_str}"),
                            html.Li([f"Déduction (1er blocage) : L'élément fautif => ", html.Span(fault_item, style={'color': COLORS['fail'], 'fontWeight': 'bold'})])
                        ], style={'margin': '5px 0', 'paddingLeft': '20px', 'fontSize': '13px'})
                    ], style={'marginBottom': '10px'}))
            else:
                syndrome = sum(val * (2**idx) for idx, val in bits.items())
                if syndrome > 0:
                    fault_found = True
                    if syndrome - 1 < len(data['items']):
                        fault_item = data['items'][syndrome - 1]
                        warning_text = ""
                        if topo == 'arbre' and data.get('variant') == 'theorique':
                            warning_text = " (⚠️ Syndrome faussé par le masquage physique !)"
                    else:
                        fault_item = f"Index {syndrome} INCONNU"
                        warning_text = " (🚨 Matrice corrompue par l'effet de masquage !)"

                    if topo == 'arbre' and data.get('type') == 'path':
                        strate_stop = True 
                        
                    report_content.append(html.Div([
                        html.Strong(f"Analyse Matrice : {phase}", style={'color': COLORS['accent']}),
                        html.Ul([
                            html.Li(f"Vecteur binaire lu (T_n ... T1) : {bit_str}"),
                            html.Li([f"Déduction Théorique : L'élément fautif correspond à l'index {syndrome} => ", html.Span(fault_item, style={'color': COLORS['fail'], 'fontWeight': 'bold'}), html.Span(warning_text, style={'fontSize': '11px', 'color': '#C0392B'})])
                        ], style={'margin': '5px 0', 'paddingLeft': '20px', 'fontSize': '13px'})
                    ], style={'marginBottom': '10px'}))

        if not fault_found and topo != 'arbre':
            report_content.append(html.P("Panne détectée par une sonde unitaire globale (Hors Matrice LTP).", style={'fontSize': '13px', 'fontStyle': 'italic'}))

    if topo in ['lineaire', 'grille', 'arbre']:
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
                html.Strong("Logique d'Intersection Spatiale (Vérité Physique)", style={'color': COLORS['success']}),
                html.Ul([
                    html.Li("Soustraction des arêtes de routage validées par les sondes saines."),
                    html.Li(f"Déduction finale absolue : L'arête défaillante est {list(final_fault)}", style={'color': COLORS['fail'], 'fontWeight': 'bold'})
                ], style={'margin': '5px 0', 'paddingLeft': '20px', 'fontSize': '13px'})
            ]))

    elif mode == 'naif':
        for res in results:
            if res['failed']:
                report_content.append(html.Div([
                    html.Strong("Mode Naïf (Recherche Unitaire)", style={'color': COLORS['accent']}),
                    html.Ul([
                        html.Li(f"Sonde en échec : {res['probe']['step_name']}"),
                        html.Li(f"Déduction : Panne trouvée par balayage séquentiel direct.", style={'color': COLORS['fail'], 'fontWeight': 'bold'})
                    ], style={'margin': '5px 0', 'paddingLeft': '20px', 'fontSize': '13px'})
                ]))
                break

    return html.Div([
        html.H4("🧠 Rapport du Contrôleur (Décodage Top-Down)", style={'marginTop': '0', 'borderBottom': '2px solid #ccc', 'paddingBottom': '5px', 'color': '#2C3E50'}),
        html.P("Le contrôleur a collecté l'ensemble des statuts 0 et 1 pour effectuer ses calculs :", style={'fontSize': '12px', 'fontStyle': 'italic', 'color': '#7F8C8D'}),
        *report_content
    ], style={'backgroundColor': '#FDFEFE', 'padding': '15px', 'borderRadius': '5px', 'border': f"2px solid {COLORS['accent']}", 'marginTop': '20px'})

def _build_detailed_modal_content(results, topo, mode):
    probe_list = []
    for r in results:
        icon = "❌ ÉCHEC" if r['failed'] else "✅ SUCCÈS"
        col = COLORS['fail'] if r['failed'] else COLORS['success']
        path_str = " ➔ ".join([str(n) for n in r['probe']['path']])
        probe_list.append(html.Div([
            html.Span(f"{icon} | {r['probe']['step_id']} ({r['probe']['step_name']})", style={'fontWeight': 'bold', 'color': col, 'display': 'inline-block', 'width': '250px'}),
            html.Span(f"Chemin: {path_str}", style={'fontSize': '11px', 'color': '#7f8c8d', 'fontFamily': 'monospace'})
        ], style={'padding': '8px 0', 'borderBottom': '1px solid #eee'}))

    logic_steps = []
    
    if mode == 'ltp' and topo in ['complet', 'arbre', 'grille']:
        logic_steps.append(html.Div([
            html.P("💡 Comment fonctionne la matrice ?", style={'fontWeight': 'bold', 'color': COLORS['accent'], 'marginBottom': '5px'}),
            html.P("Chaque sonde est comme une question posée au réseau. Si la sonde échoue, la réponse est '1'. Si elle réussit, la réponse est '0'. En mettant toutes ces réponses bout à bout, on obtient un code d'identification.", style={'fontSize': '13px', 'color': '#555', 'fontStyle': 'italic', 'marginBottom': '15px'})
        ]))
        
        phases = {}
        for res in results:
            if 'ltp_meta' in res['probe'] and res['probe']['ltp_meta']:
                meta = res['probe']['ltp_meta']
                p = meta['phase']
                if p not in phases: phases[p] = {'bits': {}, 'items': meta['items'], 'type': meta.get('type'), 'ld': meta.get('ld', 0), 'variant': meta.get('variant')}
                phases[p]['bits'][meta['active_test']] = 1 if res['failed'] else 0
        
        faulty_depth = None
        faulty_path_str = None
        fault_found = False
        strate_stop = False
        
        sorted_phases = []
        if topo == 'arbre':
            depth_phase = [p for p, d in phases.items() if d.get('type') == 'depth']
            path_phases = sorted([p for p, d in phases.items() if d.get('type') == 'path'], key=lambda x: phases[x].get('ld', 0))
            sorted_phases = depth_phase + path_phases
        else:
            sorted_phases = list(phases.keys())
        
        for phase in sorted_phases:
            data = phases[phase]
            bits = data['bits']
            if not bits: continue
            
            if topo == 'arbre' and data.get('type') == 'path' and strate_stop:
                logic_steps.append(html.Div([
                    html.H5(f"Analyse du groupe : {phase}", style={'color': '#95A5A6', 'margin': '10px 0 5px 0'}),
                    html.P("🛑 Analyse Ignorée (Règle Top-Down) : La panne a été identifiée dans une strate supérieure. Les résultats de cette matrice inférieure sont faussés par le masquage physique, on les ignore.", style={'color': '#7F8C8D', 'fontStyle': 'italic', 'fontSize': '12px', 'margin': '0 0 10px 0'})
                ]))
                continue
                
            bit_str = "".join(str(bits[i]) for i in sorted(bits.keys(), reverse=True))
            
            if data.get('type') == 'depth' and data.get('variant') == 'physique':
                first_fail = -1
                for idx in sorted(bits.keys()):
                    if bits[idx] == 1:
                        first_fail = idx
                        break
                if first_fail != -1:
                    fault_found = True
                    faulty_depth = data['items'][first_fail]
                    fault_res = f"Premier blocage physique détecté à l'étape {first_fail} => {faulty_depth}."
                    res_color = COLORS['fail']
                    calc_str = "Recherche du 1er '1' (Évite l'Effet Cascade)"
                else:
                    fault_res = "Valeur 0 = Toutes les profondeurs sont saines."
                    res_color = COLORS['success']
                    calc_str = "N/A"
            else:
                calc_str = " + ".join([f"{val}*(2^{idx})" for idx, val in bits.items()])
                syndrome = sum(val * (2**idx) for idx, val in bits.items())
                if syndrome > 0:
                    fault_found = True
                    if syndrome - 1 < len(data['items']):
                        fault_item = data['items'][syndrome-1]
                        fault_res = f"La panne théorique est l'élément N°{syndrome} ({fault_item})."
                        res_color = COLORS['fail']
                        if topo == 'arbre':
                            if data.get('type') == 'depth': faulty_depth = fault_item
                            if data.get('type') == 'path': 
                                faulty_path_str = fault_item
                                strate_stop = True
                    else:
                        fault_res = f"🚨 ERREUR MATRICE : Le syndrome {syndrome} est hors limites ! Le code binaire a été corrompu par le masquage physique du laser."
                        res_color = COLORS['fail']
                        faulty_depth = f"Index Inconnu ({syndrome})"
                else:
                    fault_res = "Valeur 0 = Aucun élément de ce groupe n'est en panne (Sain)."
                    res_color = COLORS['success']

            phase_desc = ""
            if "Hub" in phase: phase_desc = " (Les connexions directes au centre)"
            elif "Cycle" in phase: phase_desc = " (Les connexions distantes entre voisins)"
            elif "Colonne 0" in phase: phase_desc = " (Vérification de l'axe vertical de référence)"
            elif "Ligne 0" in phase: phase_desc = " (Vérification de l'axe horizontal de référence)"
            elif "Lignes Globales" in phase: phase_desc = " (Groupement de lignes entières)"
            elif "Colonnes Globales" in phase: phase_desc = " (Groupement de colonnes entières)"
            elif "Échelle Vert" in phase: phase_desc = " (Groupement des k-ièmes arêtes horizontales)"
            elif "Échelle Horiz" in phase: phase_desc = " (Groupement des k-ièmes arêtes verticales)"
            elif "Profondeur Absolue" in phase: phase_desc = " (Matrice LTP ou Diagonale)"
            elif "Strate" in phase: phase_desc = " (Détection Top-Down de l'autoroute HLD)"
            elif "Chemins Préférés" in phase: phase_desc = " (Détection globale théorique)"
            
            logic_steps.append(html.Div([
                html.H5([f"Analyse du groupe : {phase}", html.Span(phase_desc, style={'fontSize': '12px', 'color': '#7F8C8D', 'fontWeight': 'normal', 'marginLeft': '5px'})], style={'color': '#2C3E50', 'margin': '10px 0 5px 0', 'borderBottom': '1px dotted #ccc', 'paddingBottom': '5px'}),
                html.Ul([
                    html.Li([html.Strong("Réponses collectées (Vecteur) : "), html.Span(bit_str, style={'fontFamily': 'monospace', 'backgroundColor': '#ecf0f1', 'padding': '2px 5px', 'borderRadius': '3px'})]),
                    html.Li([html.Strong("Logique de Décodage : "), calc_str]),
                    html.Li([html.Strong("Conclusion : "), html.Span(fault_res, style={'fontWeight': 'bold', 'color': res_color})])
                ], style={'fontSize': '13px', 'backgroundColor': '#f8f9fa', 'padding': '10px 10px 10px 30px', 'borderRadius': '5px', 'listStyleType': 'square'})
            ]))

        if topo == 'arbre' and fault_found and faulty_depth and faulty_path_str:
            logic_steps.append(html.Div([
                html.H5("📍 Détail du Croisement Exact (Heavy-Light Decomposition)", style={'color': COLORS['accent'], 'margin': '15px 0 5px 0', 'borderBottom': '1px solid #ccc'}),
                html.P("L'algorithme croise mathématiquement les 2 coordonnées théoriques :", style={'fontSize': '13px'}),
                html.Ul([
                    html.Li([html.Strong("Coordonnée Y (Niveau) : "), f"L'analyse indique la {faulty_depth}."]),
                    html.Li([html.Strong("Coordonnée X (Autoroute) : "), f"Le syndrome indique le {faulty_path_str}."]),
                    html.Li([html.Strong("Déduction Mathématique : "), html.Span("L'arête en panne est le croisement de cette Autoroute et de cette Profondeur.", style={'color': COLORS['fail'], 'fontWeight': 'bold'})])
                ], style={'fontSize': '13px', 'backgroundColor': '#FDEDEC', 'padding': '10px 10px 10px 30px', 'borderRadius': '5px', 'listStyleType': 'none', 'borderLeft': '4px solid #C0392B'})
            ]))

    if topo in ['lineaire', 'grille', 'arbre']:
        
        def fmt_e(e): return f"{e[0]} ↔ {e[1]}"
        def badge_set(edges, color_bg, color_txt):
            if not edges: return html.Span("∅ (Aucune)", style={'fontStyle': 'italic', 'color': '#7f8c8d', 'fontSize': '12px'})
            return html.Div(
                [html.Span(fmt_e(e), style={'backgroundColor': color_bg, 'color': color_txt, 'padding': '3px 6px', 'borderRadius': '4px', 'fontSize': '11px', 'border': f'1px solid {color_txt}'}) for e in sorted(list(edges), key=str)],
                style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '4px', 'marginTop': '5px', 'marginBottom': '10px'}
            )

        safe_edges = set()
        suspect_edges = None
        intersection_trace = []
        success_probes = []

        for res in results:
            path = res['probe']['path']
            edges = set(_normalize_edge(path[i], path[i+1]) for i in range(len(path)-1))
            name = res['probe']['step_name']
            
            if res['failed']:
                if suspect_edges is None:
                    suspect_edges = set(edges)
                    intersection_trace.append(html.Div([
                        html.Strong(f"🔴 1er Échec détecté : Sonde '{name}'"),
                        html.Div("Toutes les arêtes de cette sonde deviennent suspectes :", style={'fontSize': '12px', 'marginTop': '3px'}),
                        badge_set(suspect_edges, '#FDEDEC', '#C0392B')
                    ], style={'marginBottom': '15px', 'borderLeft': '3px solid #C0392B', 'paddingLeft': '10px'}))
                else:
                    new_suspects = suspect_edges.intersection(edges)
                    intersection_trace.append(html.Div([
                        html.Strong(f"🔴 Nouvel Échec : Sonde '{name}'"),
                        html.Div("Arêtes traversées par cette nouvelle sonde :", style={'fontSize': '12px', 'marginTop': '3px'}),
                        badge_set(edges, '#FADBD8', '#E74C3C'),
                        html.Div("👉 INTERSECTION (On ne garde que les suspects en commun avec le précédent) :", style={'fontSize': '12px', 'marginTop': '3px', 'fontWeight': 'bold'}),
                        badge_set(new_suspects, '#FDEDEC', '#C0392B')
                    ], style={'marginBottom': '15px', 'borderLeft': '3px solid #C0392B', 'paddingLeft': '10px'}))
                    suspect_edges = new_suspects
            else:
                success_probes.append(name)
                safe_edges = safe_edges.union(edges)

        if suspect_edges is not None:
            final_fault = suspect_edges - safe_edges
            
            logic_steps.append(html.Div([
                html.P("💡 Trace Visuelle de l'Intersection Physique", style={'fontWeight': 'bold', 'color': COLORS['success'], 'marginBottom': '10px'}),
                html.P("La théorie peut être faussée par la physique. Mais l'algorithme recoupe toujours les chemins géographiques étape par étape pour vérifier la vérité physique :", style={'fontSize': '13px', 'color': '#555', 'fontStyle': 'italic', 'marginBottom': '15px'}),
                html.Div(intersection_trace, style={'backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '5px', 'border': '1px solid #ddd', 'marginBottom': '15px'}),
                html.Div([
                    html.Strong(f"✅ Phase d'Innocentation (Soustraction)"),
                    html.Div(f"Les sondes qui ont réussi ({len(success_probes)} sondes) ont prouvé que les arêtes suivantes sont 100% saines :", style={'fontSize': '12px', 'marginTop': '3px'}),
                    badge_set(safe_edges, '#E8F8F5', '#27AE60'),
                    html.Div("👉 SOUSTRACTION FINALE (Suspects restants MOINS Arêtes saines) :", style={'fontSize': '13px', 'marginTop': '10px', 'fontWeight': 'bold', 'color': '#2C3E50'}),
                    badge_set(final_fault, '#FEF9E7', '#D35400')
                ], style={'backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '5px', 'border': '2px solid #27AE60'})
            ]))

    return html.Div([
        html.H4("1. Historique Brut des Sondes (La Collecte)", style={'color': '#2C3E50', 'marginTop': '10px', 'borderBottom': '1px solid #ccc', 'paddingBottom': '5px'}),
        html.P("Ceci est le journal de bord du contrôleur réseau. Il a envoyé toutes ces sondes à l'aveugle et a noté leur statut.", style={'fontSize': '12px', 'color': '#7F8C8D'}),
        html.Div(probe_list, style={'backgroundColor': '#fff', 'border': '1px solid #ddd', 'padding': '10px', 'borderRadius': '5px', 'maxHeight': '250px', 'overflowY': 'auto', 'marginBottom': '20px'}),
        html.H4("2. Démonstration Mathématique (Le Diagnostic)", style={'color': '#2C3E50', 'borderBottom': '1px solid #ccc', 'paddingBottom': '5px'}),
        html.Div(logic_steps)
    ])

class NetworkEngine:
    def __init__(self, n, topo, mode='naif', tree_variant='theorique'):
        self.n = n
        self.type = topo
        self.mode = mode 
        self.tree_variant = tree_variant 
        self.G = self._build_graph()
        self.probes = []
        
        self.weights = {}
        self.heavy_edges = set()
        self.light_edges = set()
        self.preferred_paths = []
        self.light_depth_of_path = {}
        
        if self.type == 'grille': self._generate_grid_algo()
        elif self.type == 'lineaire': self._generate_linear_algo()
        elif self.type == 'complet': self._generate_complete_algo()
        elif self.type == 'arbre': self._generate_tree_algo()

    def _build_graph(self):
        if self.type == 'lineaire': return nx.path_graph(self.n)
        elif self.type == 'complet': return nx.complete_graph(self.n)
        elif self.type == 'arbre': return nx.random_labeled_tree(self.n, seed=42)
        elif self.type == 'grille':
            s = int(math.sqrt(self.n))
            return nx.grid_2d_graph(s, s)
        return nx.Graph()

    def _get_path(self, u, v):
        try: return nx.shortest_path(self.G, u, v)
        except: return []

    def _add_probe(self, path, step_id, step_name, description, ltp_meta=None, target_edges=None):
        self.probes.append({
            'path': path, 'step_id': step_id, 'step_name': step_name, 
            'description': description, 'id': len(self.probes) + 1, 
            'ltp_meta': ltp_meta, 'target_edges': target_edges or []
        })

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
        self.weights = {}
        self.heavy_edges = set()
        self.light_edges = set()
        
        def dfs_weight(u, p):
            w = 1
            max_c_w = -1
            heavy_c = None
            children = []
            for v in self.G.neighbors(u):
                if v != p:
                    children.append(v)
                    cw = dfs_weight(v, u)
                    w += cw
                    if cw > max_c_w:
                        max_c_w = cw
                        heavy_c = v
            self.weights[u] = w
            if heavy_c is not None:
                self.heavy_edges.add(_normalize_edge(u, heavy_c))
                for v in children:
                    if v != heavy_c:
                        self.light_edges.add(_normalize_edge(u, v))
            return w
        
        dfs_weight(0, None)
        
        self.preferred_paths = []
        self.light_depth_of_path = {}
        visited = {0}
        
        def extract_path(start_node, parent, current_ld):
            curr = start_node
            path = []
            if parent is not None: path.append(parent)
            path.append(curr)
            visited.add(curr)
            
            while True:
                nxt = None
                for v in self.G.neighbors(curr):
                    if v not in visited and _normalize_edge(curr, v) in self.heavy_edges:
                        nxt = v
                        break
                if nxt:
                    path.append(nxt)
                    visited.add(nxt)
                    curr = nxt
                else:
                    break
                    
            if len(path) > 1:
                pid = len(self.preferred_paths)
                self.preferred_paths.append(path)
                self.light_depth_of_path[pid] = current_ld
                
            for node in path:
                if node == parent: continue
                for v in self.G.neighbors(node):
                    if v not in visited:
                        extract_path(v, node, current_ld + 1)
                        
        extract_path(0, None, 0)
        
        def get_dfs_route(target_edges):
            if not target_edges: return [0]
            nodes_in_subtree = {0}
            for u, v in target_edges:
                for node in self._get_path(0, u): nodes_in_subtree.add(node)
                for node in self._get_path(0, v): nodes_in_subtree.add(node)
                
            route = []
            def dfs_build(u, p):
                route.append(u)
                for v in self.G.neighbors(u):
                    if v != p and v in nodes_in_subtree:
                        dfs_build(v, u)
                        route.append(u)
            dfs_build(0, None)
            return route

        if self.mode == 'naif':
            nodes_by_depth = sorted(self.G.nodes(), key=lambda x: len(self._get_path(0, x)))
            for i in nodes_by_depth:
                if i == 0: continue
                path = self._get_path(0, i)
                self._add_probe(path + path[-2::-1], "BRANCH", "Sondage Unitaire", f"Validation branche vers {i}.")
        else:
            edges_by_depth = {}
            for u, v in self.G.edges():
                d = max(len(self._get_path(0, u)), len(self._get_path(0, v))) - 1
                if d not in edges_by_depth: edges_by_depth[d] = []
                edges_by_depth[d].append((u, v))
                
            depths = sorted(list(edges_by_depth.keys()))
            
            if self.tree_variant == 'physique':
                matrix_depth = []
                for i in range(len(depths)):
                    row = [0] * len(depths)
                    row[i] = 1
                    matrix_depth.append(row)
                for idx, d in enumerate(depths):
                    t_edges = edges_by_depth[d]
                    full_route = get_dfs_route(t_edges)
                    items_str = [f"Prof. {d_item}" for d_item in depths]
                    meta = {'matrix': matrix_depth, 'items': items_str, 'active_test': idx, 'phase': 'Profondeur Absolue', 'type': 'depth', 'variant': 'physique'}
                    self._add_probe(full_route, f"DEPTH-{idx+1}", "Balayage Prof.", f"Sondage diag. de la Prof. {d}.", ltp_meta=meta, target_edges=t_edges)
            else:
                ltp_depth = self._get_ltp_subsets(depths)
                for idx, subset in enumerate(ltp_depth['subsets']):
                    if not subset: continue
                    t_edges = []
                    for d in subset: t_edges.extend(edges_by_depth[d])
                    full_route = get_dfs_route(t_edges)
                    items_str = [f"Prof. {d}" for d in ltp_depth['items']]
                    meta = {'matrix': ltp_depth['matrix'], 'items': items_str, 'active_test': idx, 'phase': 'Profondeur Absolue', 'type': 'depth', 'variant': 'theorique'}
                    self._add_probe(full_route, f"DEPTH-{idx+1}", "LTP Profondeur", f"Sondage O(log N) aux prof. {subset}.", ltp_meta=meta, target_edges=t_edges)

            if self.tree_variant == 'physique':
                paths_by_ld = {}
                for pid, ld in self.light_depth_of_path.items():
                    if ld not in paths_by_ld: paths_by_ld[ld] = []
                    paths_by_ld[ld].append(pid)
                    
                for ld in sorted(paths_by_ld.keys()):
                    pids = paths_by_ld[ld]
                    ltp_paths = self._get_ltp_subsets(pids)
                    
                    for idx, subset in enumerate(ltp_paths['subsets']):
                        if not subset: continue
                        t_edges = []
                        for pid in subset:
                            path_nodes = self.preferred_paths[pid]
                            for i in range(len(path_nodes)-1):
                                t_edges.append((path_nodes[i], path_nodes[i+1]))
                                
                        full_route = get_dfs_route(t_edges)
                        items_str = [f"Chemin {p}" for p in ltp_paths['items']]
                        meta = {'matrix': ltp_paths['matrix'], 'items': items_str, 'active_test': idx, 'phase': f'Strate {ld} (Chemins)', 'type': 'path', 'ld': ld, 'variant': 'physique'}
                        self._add_probe(full_route, f"HLD-L{ld}-{idx+1}", f"LTP Strate {ld}", f"Sondage des Chemins de la Strate {ld} : {subset}.", ltp_meta=meta, target_edges=t_edges)
            else:
                all_pids = list(range(len(self.preferred_paths)))
                ltp_paths = self._get_ltp_subsets(all_pids)
                
                for idx, subset in enumerate(ltp_paths['subsets']):
                    if not subset: continue
                    t_edges = []
                    for pid in subset:
                        path_nodes = self.preferred_paths[pid]
                        for i in range(len(path_nodes)-1):
                            t_edges.append((path_nodes[i], path_nodes[i+1]))
                            
                    full_route = get_dfs_route(t_edges)
                    items_str = [f"Chemin {p}" for p in ltp_paths['items']]
                    meta = {'matrix': ltp_paths['matrix'], 'items': items_str, 'active_test': idx, 'phase': 'Chemins Préférés (Global)', 'type': 'path', 'ld': 0, 'variant': 'theorique'}
                    self._add_probe(full_route, f"HLD-GL-{idx+1}", "LTP Chemins", f"Sondage O(log P) de tous les chemins : {subset}.", ltp_meta=meta, target_edges=t_edges)

    def _generate_grid_algo(self):
        s = int(math.sqrt(self.n))
        origin = (0,0)

        path = [(r, 0) for r in range(s)] + [(r, 0) for r in range(s-2, -1, -1)]
        self._add_probe(path, "1a", "Axe Vertical", "Test global de la Colonne 0.")

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

        path = [(0, c) for c in range(s)] + [(0, c) for c in range(s-2, -1, -1)]
        self._add_probe(path, "2a", "Axe Horizontal", "Test global de la Ligne 0.")

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
                ], value='arbre', clearable=False),

                html.Label("Mode d'Algorithme :", style={'marginTop': '15px', 'display': 'block', 'fontWeight': 'bold'}),
                dcc.RadioItems(id='algo-mode', options=[
                    {'label': ' Naïf (Boucle itérative)', 'value': 'naif'},
                    {'label': ' LTP (Article - O(log N))', 'value': 'ltp'}
                ], value='ltp', labelStyle={'display': 'block', 'margin': '5px 0'}),

                html.Label("Mode d'Affichage :", style={'marginTop': '15px', 'display': 'block', 'fontWeight': 'bold', 'color': COLORS['success']}),
                dcc.RadioItems(id='anim-mode', options=[
                    {'label': ' Statique (Chemin complet)', 'value': 'statique'},
                    {'label': ' Animé (Voyage fluide du laser)', 'value': 'anime'}
                ], value='statique', labelStyle={'display': 'block', 'margin': '5px 0', 'fontSize': '13px'}),

                html.Div(id='tree-variant-container', children=[
                    html.Label("Variante Arbre (Diagnostic des profondeurs) :", style={'marginTop': '15px', 'display': 'block', 'fontWeight': 'bold', 'color': COLORS['fail']}),
                    dcc.RadioItems(id='tree-variant', options=[
                        {'label': ' Théorique (Article strict - Soumis au masquage)', 'value': 'theorique'},
                        {'label': ' Pratique (Anti-masquage physique)', 'value': 'physique'}
                    ], value='theorique', labelStyle={'display': 'block', 'margin': '5px 0', 'fontSize': '13px'}),
                ], style={'display': 'block'}), 

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
                html.Div(id='diagnosis-report-display'),
                
                html.Div(id='modal-btn-container', style={'display': 'none'}, children=[
                    html.Button("🔍 Voir le détail complet des sondes et du calcul", id='btn-open-modal', style={'marginTop': '10px', 'padding': '10px', 'backgroundColor': '#34495E', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'width': '100%', 'fontWeight': 'bold'})
                ])
            ]),
            
            html.Div(id='algo-doc', style={'marginTop': '20px', 'fontSize': '13px', 'color': '#555'}),
            html.Div(id='hld-details-display')
            
        ]),

        html.Div(style={'flex': '2'}, children=[
            html.Div(style={'backgroundColor': COLORS['card'], 'padding': '10px', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'height': '600px'}, children=[
                dcc.Graph(id='graph', style={'height': '100%'}, config={'displayModeBar': False})
            ]),

            html.Div(style={'backgroundColor': COLORS['card'], 'marginTop': '20px', 'padding': '20px', 'borderRadius': '10px', 'display': 'flex', 'alignItems': 'center', 'gap': '15px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}, children=[
                html.Button("▶️ Lecture", id='btn-play', style={'backgroundColor': COLORS['success'], 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold'}),
                html.Div(style={'flex': '1'}, children=[
                    dcc.Slider(id='sim-slider', min=0, max=1, step=1, value=0, tooltip={"placement": "bottom", "always_visible": True}, marks=None)
                ]),
                html.Div(id='counter-display', style={'fontWeight': 'bold', 'color': COLORS['text'], 'minWidth': '120px', 'textAlign': 'right'})
            ])
        ])
    ]),

    html.Div(id='modal-overlay', style={'display': 'none'}, children=[
        html.Div(style={'backgroundColor': 'white', 'margin': '5% auto', 'padding': '30px', 'border': '1px solid #888', 'width': '80%', 'maxWidth': '800px', 'borderRadius': '10px', 'maxHeight': '80vh', 'overflowY': 'auto', 'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'}, children=[
            html.H2("🔍 Historique et Déduction Détaillée", style={'marginTop': 0, 'color': COLORS['accent'], 'borderBottom': '2px solid #eee', 'paddingBottom': '10px'}),
            html.Div(id='modal-content'),
            html.Button("Fermer la fenêtre", id='btn-close-modal', style={'marginTop': '20px', 'padding': '10px 20px', 'backgroundColor': COLORS['fail'], 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'fontWeight': 'bold', 'width': '100%'})
        ])
    ]),

    dcc.Store(id='st-topo'), 
    dcc.Store(id='st-fault'), 
    dcc.Store(id='st-node-idx', data=0),
    dcc.Store(id='st-anim-progress', data=0.0), # NOUVEAU: Pour l'interpolation fluide
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
     Output('modal-btn-container', 'style'),
     Output('modal-content', 'children'),
     Output('counter-display', 'children'),
     Output('algo-doc', 'children'),
     Output('anim-interval', 'disabled'), 
     Output('btn-play', 'children'),
     Output('hld-details-display', 'children'),
     Output('tree-variant-container', 'style'),
     Output('st-node-idx', 'data'),
     Output('st-anim-progress', 'data'), # NOUVEAU
     Output('anim-interval', 'interval')],
    [Input('btn-build', 'n_clicks'), 
     Input('graph', 'clickData'), 
     Input('fault', 'value'),
     Input('sim-slider', 'value'), 
     Input('btn-play', 'n_clicks'), 
     Input('anim-interval', 'n_intervals'),
     Input('algo-mode', 'value'),
     Input('tree-variant', 'value'),
     Input('anim-mode', 'value')],
    [State('topo', 'value'), 
     State('n-slider', 'value'), 
     State('st-topo', 'data'), 
     State('st-fault', 'data'),
     State('st-node-idx', 'data'),
     State('st-anim-progress', 'data')] # NOUVEAU
)
def update_simulation(b_build, click, f_val, s_val, b_play, n_intervals, mode, variant, anim_mode, topo, n_nodes, st_t, st_f, st_node_idx, st_progress):
    ctx_id = ctx.triggered_id
    fig = go.Figure()
    anim_disabled = True
    btn_text = "▶️ Lecture"
    matrix_html, math_details_html, path_html, diagnosis_html = html.Div(), html.Div(), html.Div(), html.Div()
    modal_btn_style = {'display': 'none'}
    modal_content_html = html.Div()
    hld_details_html = html.Div()
    
    interval_speed = 80 if anim_mode == 'anime' else 1000 # 50ms = 20 fps pour la fluidité
    variant_style = {'display': 'block'} if topo == 'arbre' else {'display': 'none'}
    step_size = 0.25 # On fait avancer la ligne de 25% à chaque 50ms

    if ctx_id in ['btn-build', 'algo-mode', 'tree-variant', 'anim-mode', None] or not st_t:
        n_final = int(math.sqrt(n_nodes))**2 if topo == 'grille' else n_nodes
        st_t = {'n': n_final, 'type': topo, 'mode': mode, 'variant': variant}
        st_f, f_val, s_val, st_node_idx, st_progress = None, None, 0, 0, 0.0
    
    if st_f and isinstance(st_f[0], list): st_f = sorted((tuple(st_f[0]), tuple(st_f[1])), key=str)

    eng = NetworkEngine(st_t['n'], st_t['type'], mode=st_t.get('mode', 'ltp'), tree_variant=st_t.get('variant', 'theorique'))
    opts = [{'label': f"Lien {u} - {v}", 'value': str(sorted((u, v), key=str))} for u,v in eng.G.edges()]
    
    if st_t['type'] == 'arbre' and hasattr(eng, 'preferred_paths') and eng.preferred_paths:
        paths_divs = []
        for i, p in enumerate(eng.preferred_paths):
            paths_divs.append(html.Li(f"Chemin {i} : {' ➔ '.join(map(str, p))}"))
        h_edges = [f"{u}-{v}" for u,v in eng.heavy_edges]
        l_edges = [f"{u}-{v}" for u,v in eng.light_edges]
        hld_details_html = html.Div([
            html.H4("🌳 Détails de la Décomposition Lourd-Léger", style={'color': COLORS['accent'], 'marginTop': '0', 'borderBottom': '1px solid #ccc', 'paddingBottom': '5px'}),
            html.P("Ceci est la cartographie exacte générée par l'algorithme :", style={'fontSize': '12px', 'color': '#7F8C8D', 'fontStyle': 'italic'}),
            html.Strong("Arêtes Lourdes (Gras) : ", style={'fontSize': '13px'}), html.Span(", ".join(h_edges) if h_edges else "Aucune", style={'fontSize': '13px'}), html.Br(),
            html.Strong("Arêtes Légères (Pointillés) : ", style={'fontSize': '13px'}), html.Span(", ".join(l_edges) if l_edges else "Aucune", style={'fontSize': '13px'}), html.Br(),
            html.Strong("Chemins Préférés (Autoroutes) :", style={'fontSize': '13px'}),
            html.Ul(paths_divs, style={'margin': '5px 0', 'paddingLeft': '20px', 'fontSize': '13px', 'fontFamily': 'monospace'})
        ], style={'backgroundColor': '#E8F8F5', 'padding': '15px', 'borderRadius': '5px', 'marginTop': '15px', 'border': f"1px solid {COLORS['success']}"})

    if ctx_id == 'graph' and click:
        try:
            p = click['points'][0]['customdata']
            st_f = sorted([tuple(p[0]), tuple(p[1])], key=str) if st_t['type'] == 'grille' else sorted(p, key=str)
            f_val, s_val, st_node_idx, st_progress = str(st_f), 0, 0, 0.0
        except: pass
    elif ctx_id == 'fault' and f_val:
        try:
            v = ast.literal_eval(f_val)
            st_f = sorted([tuple(v[0]), tuple(v[1])], key=str) if st_t['type'] == 'grille' else sorted(v, key=str)
            s_val, st_node_idx, st_progress = 0, 0, 0.0
        except: pass

    results = eng.run_simulation(st_f)
    max_s = len(results)
    if s_val is None or s_val > max_s: s_val = 0

    if ctx_id == 'sim-slider':
        st_node_idx = len(results[s_val-1]['probe']['path']) - 1 if s_val > 0 else 0
        st_progress = 0.0

    if ctx_id == 'btn-play':
        if s_val >= max_s: 
            s_val, st_node_idx, st_progress = 1, 0, 0.0
        elif s_val == 0 and max_s > 0:
            s_val, st_node_idx, st_progress = 1, 0, 0.0
        anim_disabled = False
    elif ctx_id == 'anim-interval':
        if anim_mode == 'statique':
            if s_val < max_s:
                s_val += 1
                st_node_idx, st_progress = 0, 0.0
                if results[s_val - 1]['failed']: anim_disabled, btn_text = True, "▶️ Reprendre"
                else: anim_disabled, btn_text = False, "⏸️ Stop"
            else: anim_disabled, btn_text = True, "↺ Reset"
        else: # anim_mode == 'anime' (Interpolation fluide)
            if s_val == 0:
                if max_s > 0:
                    s_val, st_node_idx, st_progress = 1, 0, 0.0
                    anim_disabled, btn_text = False, "⏸️ Stop"
                else: anim_disabled, btn_text = True, "▶️ Lecture"
            else:
                path = results[s_val - 1]['probe']['path']
                if st_node_idx < len(path) - 1:
                    u, v = path[st_node_idx], path[st_node_idx+1]
                    is_failed_edge = (str(tuple(sorted((u, v), key=str))) == str(st_f))
                    
                    if is_failed_edge and st_progress >= 0.5:
                        # On arrête la sonde net au milieu de l'arête coupée !
                        anim_disabled, btn_text = True, "▶️ Reprendre"
                    else:
                        st_progress += step_size
                        if st_progress >= 1.0:
                            st_progress = 0.0
                            st_node_idx += 1
                            if st_node_idx >= len(path) - 1:
                                if s_val < max_s:
                                    s_val += 1
                                    st_node_idx = 0
                                    anim_disabled, btn_text = False, "⏸️ Stop"
                                else:
                                    anim_disabled, btn_text = True, "↺ Reset"
                            else:
                                anim_disabled, btn_text = False, "⏸️ Stop"
                        else:
                            anim_disabled, btn_text = False, "⏸️ Stop"
    
    # Gestion du texte du bouton Pause/Lecture
    if anim_disabled:
        if s_val >= max_s and max_s > 0: btn_text = "↺ Reset"
        elif s_val > 0 and results[s_val-1]['failed'] and anim_mode == 'statique': btn_text = "▶️ Reprendre"
        elif s_val > 0 and anim_mode == 'anime' and st_node_idx < len(results[s_val-1]['probe']['path'])-1:
            u, v = results[s_val-1]['probe']['path'][st_node_idx], results[s_val-1]['probe']['path'][st_node_idx+1]
            if str(tuple(sorted((u, v), key=str))) == str(st_f) and st_progress >= 0.5:
                btn_text = "▶️ Reprendre"
            else: btn_text = "▶️ Lecture"
        else: btn_text = "▶️ Lecture"
    else: btn_text = "⏸️ Stop"

    pos = {}
    if st_t['type'] == 'lineaire': pos = {n: (n, 0) for n in eng.G.nodes()}
    elif st_t['type'] == 'grille': pos = {n: (n[1], -n[0]) for n in eng.G.nodes()}
    elif st_t['type'] == 'arbre': pos = nx.kamada_kawai_layout(eng.G)
    elif st_t['type'] == 'complet': pos = nx.circular_layout(eng.G)

    # Tracé de base du graphe
    if st_t['type'] == 'arbre' and hasattr(eng, 'heavy_edges'):
        hx, hy, lx, ly = [], [], [], []
        for u, v in eng.G.edges():
            if _normalize_edge(u, v) in eng.heavy_edges:
                hx.extend([pos[u][0], pos[v][0], None]); hy.extend([pos[u][1], pos[v][1], None])
            else:
                lx.extend([pos[u][0], pos[v][0], None]); ly.extend([pos[u][1], pos[v][1], None])
        fig.add_trace(go.Scatter(x=hx, y=hy, mode='lines', line=dict(color='#2C3E50', width=4), hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=lx, y=ly, mode='lines', line=dict(color='#BDC3C7', width=2, dash='dot'), hoverinfo='skip'))
        for u, v in eng.G.edges():
            fig.add_trace(go.Scatter(x=[pos[u][0], pos[v][0], None], y=[pos[u][1], pos[v][1], None], mode='lines', line=dict(width=15, color='rgba(0,0,0,0)'), hoverinfo='text', text=f"Lien {u}-{v}", customdata=[[u,v],[u,v],[u,v]], showlegend=False))
    else:
        edge_x, edge_y = [], []
        for u, v in eng.G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            fig.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(width=15, color='rgba(0,0,0,0)'), hoverinfo='text', text=f"Lien {u}-{v}", customdata=[[u,v],[u,v],[u,v]], showlegend=False))
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color=COLORS['edge_inactive'], width=2), hoverinfo='skip'))

    # Affichage de la panne (rouge)
    if st_f:
        try:
            u_f, v_f = st_f
            fx0, fy0, fx1, fy1 = pos[u_f][0], pos[u_f][1], pos[v_f][0], pos[v_f][1]
            fig.add_trace(go.Scatter(x=[fx0, fx1], y=[fy0, fy1], mode='lines', line=dict(color=COLORS['fail'], width=4, dash='dot')))
            fig.add_trace(go.Scatter(x=[(fx0+fx1)/2], y=[(fy0+fy1)/2], mode='markers', marker=dict(symbol='x', size=15, color=COLORS['fail'])))
        except: pass

    if results and s_val > 0:
        curr = results[s_val - 1]
        probe, is_failed = curr['probe'], curr['failed']
        full_path = probe['path']
        
        test_idx = probe.get('ltp_meta', {}).get('active_test', (s_val - 1)) if probe.get('ltp_meta') else (s_val - 1)
        dynamic_color = PROBE_COLORS[test_idx % len(PROBE_COLORS)]
        
        highlighted_edges = _get_highlighted_edges(probe, st_t['type'])
        for highlighted_edge in highlighted_edges:
            try:
                hu, hv = highlighted_edge
                hx0, hy0 = pos[hu]; hx1, hy1 = pos[hv]
                fig.add_trace(go.Scatter(x=[hx0, hx1], y=[hy0, hy1], mode='lines', line=dict(width=9, color='#E74C3C'), opacity=0.8, hoverinfo='skip', showlegend=False))
            except: pass

        # --- DESSIN DU CHEMIN DYNAMIQUE (INTERPOLATION FLUIDE) ---
        if anim_mode == 'anime' and len(full_path) > 0 and st_node_idx < len(full_path) - 1:
            u = full_path[st_node_idx]
            v = full_path[st_node_idx + 1]
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            # Calcul précis du point intermédiaire
            curr_x = x0 + (x1 - x0) * st_progress
            curr_y = y0 + (y1 - y0) * st_progress
            
            # Dessine le chemin DEJA validé
            path_to_draw = full_path[:st_node_idx + 1]
            px, py = [pos[n][0] for n in path_to_draw], [pos[n][1] for n in path_to_draw]
            
            # Ajoute le bout de ligne en train d'être traversé
            px.append(curr_x)
            py.append(curr_y)
            
            fig.add_trace(go.Scatter(x=px, y=py, mode='lines', line=dict(width=4, color=dynamic_color), opacity=0.9))
            
            # L'étoile jaune fluo (la sonde) qui glisse
            fig.add_trace(go.Scatter(
                x=[curr_x], y=[curr_y], 
                mode='markers', 
                marker=dict(size=20, color='#F1C40F', symbol='star', line=dict(width=2, color='#D35400')), 
                name='Sonde Laser', hoverinfo='name'
            ))
        else:
            # Mode Statique ou Animation terminée pour cette sonde
            px, py = [pos[n][0] for n in full_path], [pos[n][1] for n in full_path]
            fig.add_trace(go.Scatter(x=px, y=py, mode='lines', line=dict(width=4, color=dynamic_color), opacity=0.9))

        if st_t['type'] in ['grille', 'lineaire'] and len(full_path) > 0:
            fig.layout.annotations = _build_probe_step_annotations(full_path, pos, dynamic_color)
        
        if len(full_path) > 1:
            fig.add_trace(go.Scatter(x=[pos[full_path[0]][0]], y=[pos[full_path[0]][1]], mode='markers', marker=dict(size=12, color=dynamic_color, symbol='circle')))

        is_currently_failed = False
        if is_failed:
            if anim_mode == 'statique':
                is_currently_failed = True
            elif st_node_idx >= len(full_path) - 1:
                is_currently_failed = True
            elif st_node_idx < len(full_path) - 1:
                u, v = full_path[st_node_idx], full_path[st_node_idx+1]
                if str(tuple(sorted((u, v), key=str))) == str(st_f) and st_progress >= 0.5:
                    is_currently_failed = True

        status_col = COLORS['fail'] if is_currently_failed else COLORS['success']
        display_status = "❌ ÉCHEC (Sonde Bloquée)" if is_currently_failed else "✅ EN COURS..." if anim_mode == 'anime' and st_node_idx < len(full_path)-1 else "✅ SUCCÈS"
        
        step_content = [
            html.Div([html.Span("ÉTAPE : ", style={'fontWeight': 'bold', 'color': '#7F8C8D', 'fontSize': '12px'}), html.Span(f"{probe['step_id']} - {probe['step_name']}", style={'fontWeight': 'bold', 'color': dynamic_color})]),
            html.Div([html.Span("DESCRIPTION : ", style={'fontWeight': 'bold', 'color': '#7F8C8D', 'fontSize': '12px'}), html.P(probe['description'], style={'fontSize': '14px', 'margin': '5px 0'})]),
            html.Div(style={'backgroundColor': 'white', 'padding': '10px', 'borderLeft': f'5px solid {status_col}'}, children=[html.Span("STATUT : ", style={'fontWeight': 'bold', 'fontSize': '12px'}), html.Span(display_status, style={'fontWeight': 'bold', 'color': status_col})])
        ]
        
        matrix_html = _build_matrix_html(probe.get('ltp_meta'))
        math_details_html = _build_math_details_html(probe.get('ltp_meta'), st_t['type'])
        path_html = _build_path_html(full_path, dynamic_color)

        if s_val == max_s and (anim_mode == 'statique' or st_node_idx >= len(full_path) - 1):
            diagnosis_html = _build_diagnosis_report(results, st_t['type'], st_t.get('mode', 'naif'), st_f is not None)
            modal_btn_style = {'display': 'block'} 
            modal_content_html = _build_detailed_modal_content(results, st_t['type'], st_t.get('mode', 'naif')) 
        else:
            diagnosis_html = html.Div("Le rapport de diagnostic mathématique sera généré à la fin de la séquence de test...", style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px', 'color': '#7F8C8D', 'fontStyle': 'italic', 'textAlign': 'center', 'marginTop': '20px', 'border': '1px dashed #ccc'})
    else:
        step_content = [
            html.Div([html.Span("ÉTAPE : ", style={'fontWeight': 'bold', 'color': '#7F8C8D', 'fontSize': '12px'}), html.Span("0 - Initialisation", style={'fontWeight': 'bold', 'color': COLORS['text']})]),
            html.Div([html.Span("DESCRIPTION : ", style={'fontWeight': 'bold', 'color': '#7F8C8D', 'fontSize': '12px'}), html.P("Topologie générée. Observez la structure du réseau (arêtes lourdes/légères) avant d'envoyer les sondes.", style={'fontSize': '14px', 'margin': '5px 0'})])
        ]
        matrix_html = html.Div()
        math_details_html = html.Div()
        path_html = html.Div()
        diagnosis_html = html.Div("Le rapport de diagnostic mathématique sera généré à la fin de la séquence de test...", style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px', 'color': '#7F8C8D', 'fontStyle': 'italic', 'textAlign': 'center', 'marginTop': '20px', 'border': '1px dashed #ccc'})

    node_texts = []
    for n in eng.G.nodes():
        if st_t['type'] == 'arbre' and hasattr(eng, 'weights') and n in eng.weights:
            node_texts.append(f"{n}\n(P:{eng.weights[n]})")
        else:
            node_texts.append(str(n))

    nxp, nyp = [pos[n][0] for n in eng.G.nodes()], [pos[n][1] for n in eng.G.nodes()]
    fig.add_trace(go.Scatter(x=nxp, y=nyp, mode='markers+text', text=node_texts, textposition="top center", textfont=dict(color=COLORS['text'], size=9), marker=dict(size=15, color='white', line=dict(width=2, color=COLORS['text'])), hoverinfo='none'))
    fig.update_layout(margin=dict(l=20,r=20,t=20,b=20), xaxis={'visible':False}, yaxis={'visible':False, 'scaleanchor':'x', 'scaleratio':1 if st_t['type']=='grille' else None}, plot_bgcolor=COLORS['bg'], showlegend=False)

    return fig, st_t, st_f, opts, f_val, max_s, s_val, step_content, matrix_html, math_details_html, path_html, diagnosis_html, modal_btn_style, modal_content_html, f"Total Sondes : {max_s} | Pos : {s_val}", ALGO_DOCS.get(st_t['type'], ""), anim_disabled, btn_text, hld_details_html, variant_style, st_node_idx, st_progress, interval_speed

@app.callback(
    Output('modal-overlay', 'style'),
    [Input('btn-open-modal', 'n_clicks'), Input('btn-close-modal', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_modal(n_open, n_close):
    if ctx.triggered_id == 'btn-open-modal': return {'display': 'block', 'position': 'fixed', 'zIndex': 1000, 'left': 0, 'top': 0, 'width': '100%', 'height': '100%', 'backgroundColor': 'rgba(0,0,0,0.6)'}
    return {'display': 'none'}

if __name__ == '__main__':
<<<<<<< HEAD
    app.run(debug=True)

=======
    app.run(debug=True)
>>>>>>> fa15e27 (add animation)
