import streamlit as st
import numpy as np
import math
import time
import webbrowser
from sb3_contrib import MaskablePPO
import config
from figure_sudoku_env import FigureSudokuEnv
from shapes import Geometry, Color

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

def render_cell_content(geometry, color_val):
    if geometry == Geometry.EMPTY.value and color_val == Color.EMPTY.value:
        return ""
    
    size = 80
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
        return f'<div class="partial-color" style="border-color: {c};"></div>'
    
    return f'<svg class="shape-svg" viewBox="0 0 {size} {size}">{svg_content}</svg>'

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
    st.subheader("Manuelles Setzen")
    st.caption("Wähle ein Symbol/Farbe und klicke auf das Gitter")
    
    # Formen Auswahl
    st.write("**Formen**")
    cols = st.columns(4)
    geometries = [Geometry.CIRCLE, Geometry.QUADRAT, Geometry.TRIANGLE, Geometry.HEXAGON]
    for i, g in enumerate(geometries):
        with cols[i]:
            svg = render_cell_content(g.value, Color.EMPTY.value)
            is_selected = st.session_state.selected_tool == ('geometry', g.value)
            bg_color = "#ffe6e6" if is_selected else "transparent"
            border = "2px solid red" if is_selected else "1px solid #ddd"
            
            st.markdown(f'<div style="border: {border}; background-color: {bg_color}; padding: 5px; border-radius: 5px; display: flex; justify-content: center;">{svg}</div>', unsafe_allow_html=True)
            if st.button("Wählen", key=f"geom_{g.value}", use_container_width=True, disabled=controls_disabled):
                st.session_state.selected_tool = ('geometry', g.value)
                st.rerun()

    # Farben Auswahl
    st.write("**Farben**")
    cols = st.columns(4)
    colors = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW]
    for i, c in enumerate(colors):
        with cols[i]:
            svg = render_cell_content(Geometry.CIRCLE.value, c.value)
            is_selected = st.session_state.selected_tool == ('color', c.value)
            bg_color = "#ffe6e6" if is_selected else "transparent"
            border = "2px solid red" if is_selected else "1px solid #ddd"
            
            st.markdown(f'<div style="border: {border}; background-color: {bg_color}; padding: 5px; border-radius: 5px; display: flex; justify-content: center;">{svg}</div>', unsafe_allow_html=True)
            if st.button("Wählen", key=f"color_{c.value}", use_container_width=True, disabled=controls_disabled):
                st.session_state.selected_tool = ('color', c.value)
                st.rerun()
            
    if st.button("Auswahl aufheben", use_container_width=True, key="clear_selection", disabled=controls_disabled):
        st.session_state.selected_tool = None
        st.rerun()
        
    if st.button("Feld löschen (Radiergummi)", use_container_width=True, key="eraser_tool", disabled=controls_disabled):
        st.session_state.selected_tool = ('delete', 0)
        st.rerun()

    st.divider()
    if st.button("Hilfe anzeigen / ausblenden", use_container_width=True, key="help_toggle"):
        if 'show_help' not in st.session_state: st.session_state.show_help = False
        st.session_state.show_help = not st.session_state.show_help
        st.rerun()

# Hauptbereich
st.subheader(f"Status: {st.session_state.status}")

# Gitter-Darstellung
# Wir nutzen ein CSS Grid direkt für eine stabilere Darstellung
grid_html = '<div class="sudoku-grid">'
for r in range(4):
    for c in range(4):
        cell_data = st.session_state.game_state[r, c]
        content = render_cell_content(cell_data[0], cell_data[1])
        bg_color = "#f9f9f9" if (r+c)%2==0 else "white"
        # Klick nur erlauben, wenn nicht gelöst wird
        click_action = f"document.getElementById('cell_btn_{r}_{c}').click();" if not controls_disabled else ""
        grid_html += f'<div class="sudoku-cell" style="background-color: {bg_color};" onclick="{click_action}">{content}</div>'
grid_html += '</div>'

st.markdown(grid_html, unsafe_allow_html=True)

# Unsichtbare Buttons für die Interaktion
for r in range(4):
    cols = st.columns(4) # Wir platzieren sie in Spalten, damit sie nicht zu viel Platz wegnehmen
    for c in range(4):
        with cols[c]:
            if st.button(f"Feld {r+1},{c+1}", key=f"cell_btn_{r}_{c}", help=f"Klicke auf das Gitter oben", disabled=controls_disabled):
                if st.session_state.selected_tool == ('delete', 0):
                    st.session_state.game_state[r, c] = [Geometry.EMPTY.value, Color.EMPTY.value]
                    st.session_state.env.state = st.session_state.game_state.copy()
                    st.rerun()
                else:
                    handle_cell_click(r, c)
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
        f_cols = st.columns(2)
        with f_cols[0]:
            st.markdown(render_cell_content(Geometry.CIRCLE.value, Color.EMPTY.value), unsafe_allow_html=True)
            st.caption("Kreis")
            st.markdown(render_cell_content(Geometry.TRIANGLE.value, Color.EMPTY.value), unsafe_allow_html=True)
            st.caption("Dreieck")
        with f_cols[1]:
            st.markdown(render_cell_content(Geometry.QUADRAT.value, Color.EMPTY.value), unsafe_allow_html=True)
            st.caption("Quadrat")
            st.markdown(render_cell_content(Geometry.HEXAGON.value, Color.EMPTY.value), unsafe_allow_html=True)
            st.caption("Hexagon")

        st.markdown("### FARBEN")
        c_cols = st.columns(2)
        with c_cols[0]:
            st.markdown(render_cell_content(Geometry.CIRCLE.value, Color.RED.value), unsafe_allow_html=True)
            st.caption("Rot")
            st.markdown(render_cell_content(Geometry.CIRCLE.value, Color.BLUE.value), unsafe_allow_html=True)
            st.caption("Blau")
        with c_cols[1]:
            st.markdown(render_cell_content(Geometry.CIRCLE.value, Color.GREEN.value), unsafe_allow_html=True)
            st.caption("Grün")
            st.markdown(render_cell_content(Geometry.CIRCLE.value, Color.YELLOW.value), unsafe_allow_html=True)
            st.caption("Gelb")

    with h_col3:
        st.markdown("### TEILBELEGUNGEN")
        st.write("Ab Level 11 sind Felder teilweise vorgegeben:")
        t_cols = st.columns(2)
        with t_cols[0]:
            st.markdown(render_cell_content(Geometry.CIRCLE.value, Color.EMPTY.value), unsafe_allow_html=True)
            st.caption("Farbe fehlt")
        with t_cols[1]:
            st.markdown(render_cell_content(Geometry.EMPTY.value, Color.RED.value), unsafe_allow_html=True)
            st.caption("Form fehlt")
            
        st.markdown("### STEUERUNG")
        st.write("- **Neues Spiel**: Startet eine neue Runde.")
        st.write("- **Lösen**: Lässt die KI das Rätsel lösen.")
        st.write("- **Wähle Form/Farbe**: Dann klicke auf das Gitter.")
        st.write("- **Radiergummi**: Klicke auf ein Feld zum Löschen.")
        
        st.markdown("Autor: Andreas Börzel")
        st.markdown("[GitHub: Figure-Sudoku](https://github.com/aboerzel/FigureSudoku)")
        st.markdown("Lizenz: MIT License")
