import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import math
import time
import webbrowser
from sb3_contrib import MaskablePPO
import config
from figure_sudoku_env import FigureSudokuEnv
from shapes import Geometry, Color

import os

# Deklaration der Custom Component
# Wir nutzen den Pfad zum 'frontend' Verzeichnis, wo index.html liegt
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PARENT_DIR, "frontend")
INDEX_PATH = os.path.join(FRONTEND_DIR, "index.html")

# Prüfen ob Verzeichnis existiert
if not os.path.exists(FRONTEND_DIR):
    st.error(f"Frontend-Verzeichnis nicht gefunden: {FRONTEND_DIR}")
if not os.path.exists(INDEX_PATH):
    st.error(f"index.html im Frontend-Verzeichnis nicht gefunden!")

def st_drag_drop_grid(state, key=None):
    """
    Alternative Implementierung: Wir laden die HTML-Datei direkt und betten sie ein.
    Dies umgeht Probleme mit declare_component in manchen Umgebungen.
    """
    if not os.path.exists(INDEX_PATH):
        st.error(f"Frontend-Dateien fehlen unter {FRONTEND_DIR}!")
        return None
    
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Wir nutzen declare_component, da wir bi-direktionale Kommunikation brauchen.
    _st_drag_drop_grid_component = components.declare_component("figure_sudoku_grid_v14", path=FRONTEND_DIR)
    
    # Wir setzen eine feste Höhe (height), um das Kollabieren auf 0px zu verhindern
    return _st_drag_drop_grid_component(state=state, key=key, default=None, height=500)

# Seiteneinstellungen
st.set_page_config(page_title="Figure-Sudoku", page_icon="🧩", layout="wide")

# CSS für das Gitter und die Symbole
st.markdown("""
<style>
    .sudoku-grid {
        display: grid;
        grid-template-columns: repeat(4, 100px);
        grid-template-rows: repeat(4, 100px);
        gap: 2px;
        background-color: #333;
        border: 2px solid #333;
        width: fit-content;
        margin: auto;
    }
    .sudoku-cell {
        width: 100px;
        height: 100px;
        background-color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        font-size: 24px;
        position: relative;
    }
    .sudoku-cell:hover {
        background-color: #f0f0f0;
    }
    .cell-even { background-color: #f9f9f9; }
    .cell-odd { background-color: #ffffff; }
    
    .shape-container {
        width: 80px;
        height: 80px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* SVG Styles */
    .shape-svg {
        width: 100%;
        height: 100%;
    }
    
    /* Partial color style */
    .partial-color {
        width: 70px;
        height: 70px;
        border: 4px dashed;
        border-radius: 5px;
        background-color: rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

def get_shape_svg(geometry, color_val, size=50, is_preview=False):
    color_map = {
        Color.RED.value: "red",
        Color.GREEN.value: "green",
        Color.BLUE.value: "blue",
        Color.YELLOW.value: "yellow",
        Color.EMPTY.value: "gray"
    }
    
    color = color_map.get(color_val, "gray")
    opacity = "0.5" if is_preview else "1.0"
    
    if geometry == Geometry.CIRCLE.value:
        return f'<circle cx="{size/2}" cy="{size/2}" r="{size*0.4}" fill="{color}" fill-opacity="{opacity}" />'
    elif geometry == Geometry.QUADRAT.value:
        p = size * 0.1
        s = size * 0.8
        return f'<rect x="{p}" y="{p}" width="{s}" height="{s}" fill="{color}" fill-opacity="{opacity}" />'
    elif geometry == Geometry.TRIANGLE.value:
        p = size * 0.1
        h = size * 0.8
        w = 2 * h / math.sqrt(3)
        off_x = (size - w) / 2
        points = f"{size/2},{p} {off_x+w},{p+h} {off_x},{p+h}"
        return f'<polygon points="{points}" fill="{color}" fill-opacity="{opacity}" />'
    elif geometry == Geometry.HEXAGON.value:
        p = size * 0.1
        a = (size - 2*p) / 2
        ri = math.sqrt(3) * a / 2
        mx, my = size/2, size/2
        pts = [
            (mx - a, my),
            (mx - a/2, my + ri),
            (mx + a/2, my + ri),
            (mx + a, my),
            (mx + a/2, my - ri),
            (mx - a/2, my - ri)
        ]
        points_str = " ".join([f"{x},{y}" for x, y in pts])
        return f'<polygon points="{points_str}" fill="{color}" fill-opacity="{opacity}" />'
    return ""

def render_cell_content(geometry, color_val, size=80):
    if geometry == Geometry.EMPTY.value and color_val == Color.EMPTY.value:
        return ""
    
    if geometry != Geometry.EMPTY.value and color_val != Color.EMPTY.value:
        svg_content = get_shape_svg(geometry, color_val, size)
    elif geometry != Geometry.EMPTY.value:
        svg_content = get_shape_svg(geometry, Color.EMPTY.value, size)
    else: # Color only
        color_map = {
            Color.RED.value: "red",
            Color.GREEN.value: "green",
            Color.BLUE.value: "blue",
            Color.YELLOW.value: "yellow"
        }
        c = color_map.get(color_val, "gray")
        # Partial color size scales with the given size
        p_size = int(size * 0.875) # 70 if size=80
        b_width = int(size * 0.05) # 4 if size=80
        return f'<div class="partial-color" style="border-color: {c}; width: {p_size}px; height: {p_size}px; border-width: {b_width}px;"></div>'
    
    return f'<svg class="shape-svg" viewBox="0 0 {size} {size}" style="width: {size}px; height: {size}px;">{svg_content}</svg>'

# Session State Initialisierung
if 'env' not in st.session_state:
    st.session_state.env = FigureSudokuEnv()
    try:
        st.session_state.model = MaskablePPO.load(config.MODEL_PATH)
    except:
        st.session_state.model = None
    st.session_state.game_state = np.full((4, 4, 2), Geometry.EMPTY.value, dtype=np.int32)
    st.session_state.level = 10
    st.session_state.selected_tool = None # ('geometry', val) oder ('color', val)
    st.session_state.status = "Bereit"
    st.session_state.is_solving = False
    st.session_state.solve_move_count = 0
    st.session_state.last_move_id = None

def start_new_game():
    obs, _ = st.session_state.env.reset_with_level(level=st.session_state.level)
    st.session_state.game_state = st.session_state.env.state.copy()
    st.session_state.status = "Neues Spiel gestartet"
    st.session_state.is_solving = False
    st.session_state.solve_move_count = 0

def solve_step():
    if not st.session_state.is_solving or st.session_state.model is None:
        return
    
    st.session_state.solve_move_count += 1
    env = st.session_state.env
    env.state = st.session_state.game_state.copy()
    obs = env._get_obs()
    
    action_masks = env.action_masks()
    action, _ = st.session_state.model.predict(obs, action_masks=action_masks, deterministic=True)
    
    obs, reward, terminated, truncated, _ = env.step(action)
    st.session_state.game_state = env.state.copy()
    
    st.session_state.status = f"KI löst... (Zug {st.session_state.solve_move_count})"
    
    if FigureSudokuEnv.is_done(env.state):
        st.session_state.status = f"Gelöst in {st.session_state.solve_move_count} Zügen!"
        st.session_state.is_solving = False
        st.balloons()
    elif terminated or truncated or st.session_state.solve_move_count >= config.MAX_TIMESTEPS:
        st.session_state.status = "Lösen fehlgeschlagen"
        st.session_state.is_solving = False
    
    time.sleep(0.2)
    st.rerun()

def solve_game():
    if st.session_state.model is None:
        st.error("Modell nicht geladen!")
        return
    st.session_state.is_solving = True
    st.session_state.solve_move_count = 0
    st.rerun()

def handle_cell_click(r, c):
    if st.session_state.selected_tool:
        tool_type, value = st.session_state.selected_tool
        curr_g, curr_c = st.session_state.game_state[r, c]
        
        new_g, new_c = curr_g, curr_c
        if tool_type == 'geometry':
            new_g = value
        else:
            new_c = value
            
        # Validierung
        if st.session_state.env.can_move(st.session_state.game_state, r, c, new_g, new_c):
            # Check if something actually changed
            if new_g != curr_g or new_c != curr_c:
                st.session_state.game_state[r, c] = [new_g, new_c]
                st.session_state.env.state = st.session_state.game_state.copy()
                st.session_state.env.invalidate_action_mask()
                
                if FigureSudokuEnv.is_done(st.session_state.game_state):
                    st.session_state.status = "Gelöst! Glückwunsch!"
                    st.balloons()
                else:
                    st.session_state.status = "Bereit"
            else:
                st.session_state.status = "Bereit"
        else:
            st.session_state.status = "Ungültiger Zug!"
            st.toast("Dieser Zug verstößt gegen die Regeln!", icon="⚠️")
    else:
        st.session_state.status = "Wähle zuerst eine Form oder Farbe aus!"

# UI - Sidebar
with st.sidebar:
    st.title("Figure-Sudoku")
    
    # Buttons während des Lösens deaktivieren
    controls_disabled = st.session_state.is_solving
    
    st.session_state.level = st.slider("Level", 1, 12, st.session_state.level, disabled=controls_disabled)
    
    if st.button("Neues Spiel", use_container_width=True, key="new_game_btn", disabled=controls_disabled):
        start_new_game()
        st.rerun()
        
    if st.button("Lösen", use_container_width=True, disabled=st.session_state.model is None or controls_disabled, key="solve_btn"):
        solve_game()
        st.rerun()
        
    st.divider()
    st.subheader("Bedienung")
    st.info("Ziehe eine Form oder Farbe aus der Werkzeugleiste auf ein Feld.")
    
    help_label = "Hilfe ausblenden" if st.session_state.get('show_help', False) else "Hilfe anzeigen"
    if st.button(help_label, use_container_width=True, key="help_toggle"):
        if 'show_help' not in st.session_state: st.session_state.show_help = False
        st.session_state.show_help = not st.session_state.show_help
        st.rerun()

    st.markdown("""
        <style>
        .info-box {
            background-color: rgba(0,0,0,0.03);
            border-top: 2px solid #4a90e2;
            padding: 15px 5px;
            border-radius: 0 0 5px 5px;
            font-size: 0.85rem;
            color: #555;
            line-height: 1.4;
            margin-top: 30px;
        }
        .info-title {
            font-weight: bold;
            color: #ccc;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.75rem;
        }
        .info-item {
            margin-bottom: 10px;
        }
        .info-label {
            font-weight: bold;
            display: block;
            color: #aaa;
            font-size: 0.75rem;
        }
        </style>
        <div class="info-box">
            <div class="info-title">Projekt-Info</div>
            <div class="info-item">
                <span class="info-label">Autor</span>
                Andreas Börzel
            </div>
            <div class="info-item">
                <span class="info-label">GitHub</span>
                <a href="https://github.com/aboerzel/FigureSudoku" target="_blank" style="color: #4a90e2; text-decoration: none;">Figure-Sudoku</a>
            </div>
            <div class="info-item">
                <span class="info-label">Lizenz</span>
                MIT License
            </div>
        </div>
    """, unsafe_allow_html=True)

# Gitter-Darstellung
current_state_list = st.session_state.game_state.tolist()

try:
    # Container für das Gitter
    drag_result = st_drag_drop_grid(state=current_state_list, key="sudoku_drag_grid_v14")
except Exception as e:
    st.error(f"Fehler beim Laden der Komponente: {e}")
    drag_result = None

# Status-Zeile unter dem Gitter anzeigen
st.markdown(f"### Status: {st.session_state.status}")

# Verarbeitung des Drags
if drag_result and drag_result.get('timestamp') != st.session_state.get('last_move_id'):
    r, c = drag_result['row'], drag_result['col']
    tool_type = drag_result['type']
    value = drag_result['value']
    
    st.session_state.last_move_id = drag_result['timestamp']
    
    if tool_type == 'delete':
        st.session_state.game_state[r, c] = [Geometry.EMPTY.value, Color.EMPTY.value]
        st.session_state.env.state = st.session_state.game_state.copy()
        st.session_state.env.invalidate_action_mask()
        st.rerun()
    else:
        curr_g, curr_c = st.session_state.game_state[r, c]
        new_g, new_c = curr_g, curr_c
        if tool_type == 'geometry':
            new_g = value
        else:
            new_c = value
            
        # Validierung (Regeln und Figur-Einzigartigkeit)
        if st.session_state.env.can_move(st.session_state.game_state, r, c, new_g, new_c) and \
           st.session_state.env.is_figure_available(st.session_state.game_state, new_g, new_c):
            if new_g != curr_g or new_c != curr_c:
                st.session_state.game_state[r, c] = [new_g, new_c]
                st.session_state.env.state = st.session_state.game_state.copy()
                st.session_state.env.invalidate_action_mask()
                
                if FigureSudokuEnv.is_done(st.session_state.game_state):
                    st.session_state.status = "Gelöst! Glückwunsch!"
                    st.balloons()
                else:
                    st.session_state.status = "Bereit"
                st.rerun()
        else:
            st.session_state.status = "Ungültiger Zug!"
            st.toast("Dieser Zug verstößt gegen die Regeln!", icon="⚠️")
            st.rerun()

# Wenn KI am Lösen ist, nächsten Schritt ausführen
if st.session_state.is_solving:
    solve_step()

# Hilfe Sektion
if st.session_state.get('show_help', False):
    st.divider()
    h_col1, h_col2, h_col3 = st.columns(3)
    
    with h_col1:
        st.markdown("### DAS SPIELPRINZIP")
        st.write("Figure-Sudoku ist eine Variante des klassischen Sudokus, bei der anstelle von Zahlen eine Kombination aus Form und Farbe verwendet wird.")
        st.write("Ziel ist es, das 4x4 Gitter so zu füllen, dass jede Figur (Form + Farbe) genau einmal vorkommt und die Sudoku-Regeln eingehalten werden.")
        
        st.markdown("### DIE REGELN")
        st.markdown("- Jedes Feld muss am Ende eine eindeutige Figur enthalten.")
        st.markdown("- Jede Form darf pro Zeile/Spalte nur einmal vorkommen.")
        st.markdown("- Jede Farbe darf pro Zeile/Spalte nur einmal vorkommen.")
        st.markdown("- Jede Kombination ist im gesamten Gitter einzigartig.")

    with h_col2:
        st.markdown("### FORMEN")
        f_cols = st.columns(4)
        help_size = 60
        with f_cols[0]:
            st.markdown(render_cell_content(Geometry.CIRCLE.value, Color.EMPTY.value, size=help_size), unsafe_allow_html=True)
        with f_cols[1]:
            st.markdown(render_cell_content(Geometry.QUADRAT.value, Color.EMPTY.value, size=help_size), unsafe_allow_html=True)
        with f_cols[2]:
            st.markdown(render_cell_content(Geometry.TRIANGLE.value, Color.EMPTY.value, size=help_size), unsafe_allow_html=True)
        with f_cols[3]:
            st.markdown(render_cell_content(Geometry.HEXAGON.value, Color.EMPTY.value, size=help_size), unsafe_allow_html=True)

        st.markdown("### FARBEN")
        c_cols = st.columns(4)
        with c_cols[0]:
            st.markdown(render_cell_content(Geometry.CIRCLE.value, Color.RED.value, size=help_size), unsafe_allow_html=True)
        with c_cols[1]:
            st.markdown(render_cell_content(Geometry.CIRCLE.value, Color.GREEN.value, size=help_size), unsafe_allow_html=True)
        with c_cols[2]:
            st.markdown(render_cell_content(Geometry.CIRCLE.value, Color.BLUE.value, size=help_size), unsafe_allow_html=True)
        with c_cols[3]:
            st.markdown(render_cell_content(Geometry.CIRCLE.value, Color.YELLOW.value, size=help_size), unsafe_allow_html=True)

    with h_col3:
        st.markdown("### TEILBELEGUNGEN")
        st.write("Ab Level 11 sind Felder teilweise, d.h. nur Form oder nur Farbe, vorgegeben:")
        t_cols = st.columns(4)
        with t_cols[0]:
            st.markdown(render_cell_content(Geometry.CIRCLE.value, Color.EMPTY.value, size=help_size), unsafe_allow_html=True)
        with t_cols[1]:
            st.markdown(render_cell_content(Geometry.EMPTY.value, Color.RED.value, size=help_size), unsafe_allow_html=True)
            
        st.markdown("### STEUERUNG")
        st.write("- **Neues Spiel**: Startet eine neue Runde.")
        st.write("- **Lösen**: Lässt die KI das Rätsel lösen.")
        st.write("- **Werkzeugleiste**: Ziehe eine Form oder Farbe auf das Gitter.")
        st.write("- **Radiergummi**: Ziehe ihn auf ein Feld zum Löschen.")
