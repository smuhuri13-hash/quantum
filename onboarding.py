import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from scipy.linalg import expm
import random
import math
import streamlit.components.v1 as components
import time

# =================================================================
# 1. GLOBAL HELPER SIMULATION FUNCTIONS
# =================================================================

def simulate_quantum_walk(start_node, time_val):
    """
    Simulates a Continuous-Time Quantum Walk (CTQW) on a 5-node cycle graph.
    Evolution operator: U = exp(-i * A * t)
    """
    A = np.zeros((5, 5))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
        
    psi0 = np.zeros(5, dtype=complex)
    psi0[start_node] = 1.0
    
    # Continuous-time quantum walk: U = exp(-i * A * t)
    U = expm(-1j * A * time_val)
    psit = U @ psi0
    probs = np.abs(psit)**2
    return probs

def simulate_classical_walk(start_node, steps):
    """
    Simulates a Discrete-Time Classical Random Walk on a 5-node cycle graph.
    """
    A = np.zeros((5, 5))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
        
    deg = np.sum(A, axis=0)
    M = A / deg
    
    p0 = np.zeros(5)
    p0[start_node] = 1.0
    
    pt = p0
    for _ in range(steps):
        pt = M @ pt
    return pt

def simulate_grover(N, target, iterations):
    """
    Simulates Grover's algorithm state amplitudes and probabilities step-by-step.
    """
    v = np.ones(N, dtype=complex) / np.sqrt(N)
    for _ in range(iterations):
        # Oracle: flip target amplitude
        v[target] = -v[target]
        # Diffusion: reflect about average
        mean_val = np.mean(v)
        v = 2 * mean_val - v
    probs = np.abs(v)**2
    return v, probs

# =================================================================
# 2. PLOT GENERATION HELPERS
# =================================================================

def make_grover_bars_plot(probs, target):
    colors = ["#38bdf8"] * len(probs)
    colors[target] = "#10b981"
    fig = px.bar(
        x=[f"Node {i}" for i in range(len(probs))],
        y=probs,
        labels={'x': 'User Node', 'y': 'Possibility Probability'},
        title="Possibility Probabilities Vector (|α|²)"
    )
    fig.update_traces(marker_color=colors)
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, b=10, t=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def make_grover_hist_plot(probs, target):
    outcomes = random.choices(range(len(probs)), weights=probs, k=100)
    counts = [outcomes.count(i) for i in range(len(probs))]
    colors = ["#818cf8"] * len(probs)
    colors[target] = "#ec4899"
    fig = px.bar(
        x=[f"Node {i}" for i in range(len(probs))],
        y=counts,
        labels={'x': 'State Measured', 'y': 'Shots count'},
        title="Simulated Measurement Histogram (100 Shots)"
    )
    fig.update_traces(marker_color=colors)
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, b=10, t=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def make_walk_graph_plot(q_probs):
    theta_pos = np.linspace(0, 2*np.pi, 5, endpoint=False)
    x_pos = np.cos(theta_pos)
    y_pos = np.sin(theta_pos)
    
    edges_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    fig = go.Figure()
    
    # Draw edges
    for u, v in edges_list:
        fig.add_trace(go.Scatter(
            x=[x_pos[u], x_pos[v]], y=[y_pos[u], y_pos[v]],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.15)', width=2),
            showlegend=False
        ))
        
    # Dynamic node sizes and colors mapped to probabilities
    node_sizes = [25 + 60 * p for p in q_probs]
    
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=q_probs,
            colorscale='Viridis',
            showscale=False,
            line=dict(color='white', width=1.5)
        ),
        text=[f"P{i}<br>({p:.1%})" for i, p in enumerate(q_probs)],
        textposition="top center",
        showlegend=False
    ))
    
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=35, r=35, b=35, t=35),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def make_walk_chart_plot(c_probs, q_probs):
    nodes_x = [f"Node {i}" for i in range(5)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=nodes_x,
        y=c_probs,
        name='Classical Random Walk',
        marker_color='#3b82f6'
    ))
    fig.add_trace(go.Bar(
        x=nodes_x,
        y=q_probs,
        name='Quantum Walk',
        marker_color='#10b981'
    ))
    fig.update_layout(
        barmode='group',
        height=250,
        margin=dict(l=10, r=10, b=10, t=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def get_walk_explanation_text(q_probs):
    max_node = np.argmax(q_probs)
    max_val = q_probs[max_node]
    
    if max_val > 0.3:
        txt = f"""
        <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 15px; margin-top: 15px;'>
            <b>🌊 Interference Explanation Panel:</b><br>
            - <b>Node {max_node}</b> gained higher probability ({max_val:.1%}) due to <b>interference effects</b>.<br>
            - Wave packets propagating clockwise and counter-clockwise meet at Node {max_node} in phase, constructively reinforcing the probability of finding the rumor here.
        </div>
        """
    else:
        txt = f"""
        <div style='background: rgba(56, 189, 248, 0.1); border: 1px solid #38bdf8; border-radius: 8px; padding: 15px; margin-top: 15px;'>
            <b>🌊 Wave Propagation Active:</b><br>
            - The quantum wave is diffusing smoothly across all paths. No single node dominates, showing balanced superposition.
        </div>
        """
    return txt

def make_grover_circuit_text(N, target):
    n_qubits = int(np.log2(N))
    circ = []
    for q in range(n_qubits):
        circ.append(f"q{q}: ──[ H ]──[   Oracle   ]──[  Diffusion  ]──[ M ]")
    circ.append(f"Oracle: Marks target index |{target}⟩ (sign inversion)")
    circ.append("Diffusion: Reflects amplitudes about the average (amplitude amplification)")
    return "\n".join(circ)

# =================================================================
# 3. ONBOARDING TERMINOLOGY & STYLING
# =================================================================

TERMINOLOGY_MAPPING = {
    "node": {"tech": "Node", "easy": "Person (Node)"},
    "edge": {"tech": "Edge", "easy": "Relationship (Edge)"},
    "state": {"tech": "Quantum State Vector", "easy": "Possibility Profile (Quantum State)"},
    "amplitude": {"tech": "Amplitude Encoding", "easy": "Social Density Mapping (Amplitude Encoding)"},
    "centrality": {"tech": "Centrality Metrics", "easy": "Influence Scores (Centrality)"},
    "quantum_walk": {"tech": "Quantum Walk", "easy": "Simultaneous Path Exploration (Quantum Walk)"},
    "entanglement": {"tech": "Quantum Entanglement", "easy": "Tight Peer Coupling (Entanglement)"},
}

def translate(term_key, beginner_mode):
    if term_key in TERMINOLOGY_MAPPING:
        return TERMINOLOGY_MAPPING[term_key]["easy"] if beginner_mode else TERMINOLOGY_MAPPING[term_key]["tech"]
    return term_key

def inject_onboarding_styles():
    st.markdown("""
    <style>
    .mesh-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -2;
        overflow: hidden;
        background: #0f172a;
    }
    .orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0.12;
        animation: float 20s infinite alternate ease-in-out;
    }
    .orb-1 {
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, #38bdf8 0%, rgba(56, 189, 248, 0) 70%);
        top: -10%;
        left: -10%;
        animation-delay: 0s;
    }
    .orb-2 {
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, #818cf8 0%, rgba(129, 140, 248, 0) 70%);
        bottom: -15%;
        right: -10%;
        animation: float 25s infinite alternate-reverse ease-in-out;
    }
    .orb-3 {
        width: 450px;
        height: 450px;
        background: radial-gradient(circle, #ec4899 0%, rgba(236, 72, 153, 0) 70%);
        top: 40%;
        left: 45%;
        animation: float 18s infinite alternate ease-in-out;
    }
    @keyframes float {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(5%, 8%) scale(1.05); }
        100% { transform: translate(-5%, -5%) scale(0.95); }
    }
    .onboard-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .onboard-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 40px rgba(56, 189, 248, 0.15);
        background: rgba(255, 255, 255, 0.05);
    }
    .pulse-glow {
        animation: pulse 2s infinite alternate ease-in-out;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 5px rgba(56, 189, 248, 0.4); }
        100% { box-shadow: 0 0 20px rgba(129, 140, 248, 0.8); }
    }
    .coin-container {
        width: 100px;
        height: 100px;
        perspective: 1000px;
        margin: 20px auto;
    }
    .coin {
        width: 100%;
        height: 100%;
        position: relative;
        transform-style: preserve-3d;
        transition: transform 1s;
    }
    .coin.spin {
        animation: spinCoin 3s infinite linear;
    }
    .coin-face {
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.5rem;
        border: 2px solid #818cf8;
    }
    .coin-front {
        background: radial-gradient(circle, #38bdf8 0%, #1e3a8a 100%);
        color: white;
    }
    .coin-back {
        background: radial-gradient(circle, #ec4899 0%, #701a75 100%);
        color: white;
        transform: rotateY(180deg);
    }
    @keyframes spinCoin {
        0% { transform: rotateY(0deg); }
        100% { transform: rotateY(360deg); }
    }
    .flip-card-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }
    .flip-card {
        background-color: transparent;
        width: 250px;
        height: 250px;
        perspective: 1000px;
    }
    .flip-card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.6s;
        transform-style: preserve-3d;
        cursor: pointer;
    }
    .flip-card:hover .flip-card-inner {
        transform: rotateY(180deg);
    }
    .flip-card-front, .flip-card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        -webkit-backface-visibility: hidden;
        backface-visibility: hidden;
        border-radius: 12px;
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .flip-card-front {
        background: rgba(255, 255, 255, 0.04);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .flip-card-back {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        color: white;
        transform: rotateY(180deg);
        border: 1px solid #38bdf8;
    }
    </style>
    
    <div class="mesh-bg">
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>
    </div>
    """, unsafe_allow_html=True)

# =================================================================
# 4. PAGE RENDERERS
# =================================================================

def show_landing_page():
    inject_onboarding_styles()
    st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-bottom: 0.5rem;'>Quantum Social Network Analysis Studio</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #94a3b8; margin-bottom: 2.5rem; max-width: 800px; margin-left: auto; margin-right: auto;'>Unlock complex network insights using quantum walks, superposition encodings, and hybrid quantum neural networks.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="onboard-card">
            <div>
                <h3 style="margin-top: 0; color: #38bdf8;">🏫 Guided Workshop Mode</h3>
                <p style="color: #cbd5e1; font-size: 0.95rem; min-height: 80px;">Take the linear step-by-step structured curriculum. Learn myths, run search race animations, and explore walk propagation maps.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Workshop Mode", key="btn_workshop", use_container_width=True):
            st.session_state.page = "workshop"
            st.session_state.workshop_step = 1
            st.rerun()
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="onboard-card">
            <div>
                <h3 style="margin-top: 0; color: #10b981;">📂 Real-World Application Cards</h3>
                <p style="color: #cbd5e1; font-size: 0.95rem; min-height: 80px;">Browse specific domain networks: Brain Connectivity, Financial Fraud, Misinformation spread, Twitter influence, and load them directly.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Browse Application Cards", key="btn_demo", use_container_width=True):
            st.session_state.page = "demo_datasets"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="onboard-card">
            <div>
                <h3 style="margin-top: 0; color: #f59e0b;">📤 Upload Your Network</h3>
                <p style="color: #cbd5e1; font-size: 0.95rem; min-height: 80px;">Import your own custom network dataset (CSV adjacency lists, GraphML, or JSON) and perform state encoding immediately.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Upload Dataset", key="btn_upload", use_container_width=True):
            st.session_state.page = "dashboard"
            st.session_state.dashboard_step = "1. Data Input"
            st.rerun()
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="onboard-card">
            <div>
                <h3 style="margin-top: 0; color: #8b5cf6;">🔬 Advanced Research Mode</h3>
                <p style="color: #cbd5e1; font-size: 0.95rem; min-height: 80px;">Access the complete technical dashboard: customize quantum circuits, compare ML runtimes, and explore matrices.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Dashboard", key="btn_dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
            
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin-top: 3rem;'>", unsafe_allow_html=True)
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("<span style='color: #94a3b8;'>Toggle Beginner Mode globally to hide equations and use relatable social network analogies.</span>", unsafe_allow_html=True)
    with col_right:
        st.session_state.beginner_mode = st.checkbox("Enable Beginner Explanations", value=st.session_state.beginner_mode)

def show_learn_page():
    inject_onboarding_styles()
    if st.button("← Return to Landing Page", key="btn_back_l"):
        st.session_state.page = "landing"
        st.rerun()
        
    step = st.session_state.learn_step
    
    # Progress Indicators
    step_titles = [
        "1. Bit vs Qubit",
        "2. Superposition",
        "3. Entanglement",
        "4. Search Speedup",
        "5. Tiny Quantum Walk"
    ]
    
    cols = st.columns(5)
    for idx, name in enumerate(step_titles):
        is_active = (idx + 1 == step)
        is_done = (idx + 1 < step)
        color = "#38bdf8" if is_active else ("#10b981" if is_done else "#475569")
        font_weight = "bold" if is_active else "normal"
        cols[idx].markdown(
            f"<div style='text-align: center; border-bottom: 4px solid {color}; padding-bottom: 8px; color: {color}; font-weight: {font_weight}; font-size: 0.9rem;'>{name}</div>", 
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if step == 1:
        st.header("Demo 1: Classical Bit vs Qubit")
        st.write("In classical networks, a person is either connected (1) or disconnected (0). In quantum representations, a node's state can exist as a qubit.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Classical State")
            bit_val = st.radio("Choose classical Bit state:", [0, 1], horizontal=True)
            st.markdown(f"<div style='font-size: 4rem; text-align: center; font-weight: 800; color: #38bdf8; margin: 20px 0;'>{bit_val}</div>", unsafe_allow_html=True)
            st.write("Classical systems exist in one state. A bit is strictly `0` or `1`. There is no uncertainty before measurement.")
            
        with col2:
            st.subheader("Quantum State")
            p0 = st.slider("Probability of |0⟩ (P(|0⟩))", 0.0, 1.0, 0.5, 0.05)
            p1 = 1.0 - p0
            phase_deg = st.slider("Phase Angle (degrees)", 0, 360, 0, 15)
            phase_rad = math.radians(phase_deg)
            alpha_mag = math.sqrt(p0)
            beta_mag = math.sqrt(p1)
            
            if st.session_state.beginner_mode:
                st.markdown(f"<div style='font-size: 1.5rem; text-align: center; font-weight: bold; color: #a855f7; margin: 20px 0;'>Possibility 0: {p0:.0%} | Possibility 1: {p1:.0%}</div>", unsafe_allow_html=True)
            else:
                phase_exp = f"e^(i*{phase_deg}°)" if phase_deg != 0 else ""
                st.markdown(f"<div style='font-size: 1.8rem; text-align: center; font-weight: bold; color: #a855f7; margin: 20px 0;'>{alpha_mag:.3f}|0⟩ + {beta_mag:.3f}{phase_exp}|1⟩</div>", unsafe_allow_html=True)
                
            theta = 2 * math.acos(alpha_mag)
            x = math.sin(theta) * math.cos(phase_rad)
            y = math.sin(theta) * math.sin(phase_rad)
            z = math.cos(theta)
            
            fig = go.Figure()
            u_grid = np.linspace(0, 2 * np.pi, 30)
            v_grid = np.linspace(0, np.pi, 15)
            for u in u_grid:
                fig.add_trace(go.Scatter3d(x=np.sin(v_grid)*np.cos(u), y=np.sin(v_grid)*np.sin(u), z=np.cos(v_grid), mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1), showlegend=False))
            for v in v_grid:
                fig.add_trace(go.Scatter3d(x=np.sin(v)*np.cos(u_grid), y=np.sin(v)*np.sin(u_grid), z=np.cos(v)*np.ones_like(u_grid), mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1), showlegend=False))
                
            fig.add_trace(go.Scatter3d(x=[-1, 1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='#475569', width=2), name="X"))
            fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1, 1], z=[0, 0], mode='lines', line=dict(color='#475569', width=2), name="Y"))
            fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1, 1], mode='lines', line=dict(color='#475569', width=2), name="Z"))
            fig.add_trace(go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode='lines+markers', marker=dict(size=6, color='#a855f7'), line=dict(color='#a855f7', width=5), name="Qubit"))
            
            fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'), margin=dict(l=0, r=0, b=0, t=0), height=300, paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("---")
            if st.button("⚡ Measure Qubit"):
                outcome = 0 if random.random() < p0 else 1
                st.session_state.q_outcome = outcome
                
            if "q_outcome" in st.session_state:
                collapsed_symbol = "|0⟩" if st.session_state.q_outcome == 0 else "|1⟩" if st.session_state.q_outcome == 1 and not st.session_state.beginner_mode else f"Possibility {st.session_state.q_outcome}"
                st.markdown(f"<div class='pulse-glow' style='background: rgba(168, 85, 247, 0.1); border: 1px solid #a855f7; border-radius: 8px; padding: 15px; text-align: center; font-weight: bold;'>Collapsed State: {collapsed_symbol}!</div>", unsafe_allow_html=True)
                
            st.write("")
            if st.session_state.beginner_mode:
                st.info("💡 **Analogy:** Unlike a light switch that is either OFF (0) or ON (1), a quantum state is like a dimmer switch that can represent multiple levels at the same time until measured.")
            else:
                st.info("💡 **Quantum Note:** A qubit is in a linear superposition of computational basis states. Measuring collapses it to $|0\\rangle$ or $|1\\rangle$.")

    elif step == 2:
        st.header("Demo 2: Superposition Demo")
        st.write("Superposition allows a quantum state to represent multiple values simultaneously. We create superposition using a **Hadamard Gate**.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Circuit")
            st.markdown("<div style='background: rgba(255,255,255,0.02); padding: 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); text-align: center;'><span style='font-family: monospace; font-size: 1.2rem; color: #818cf8;'>|0⟩ ───[ H ]─── 1/√2(|0⟩ + |1⟩)</span></div>", unsafe_allow_html=True)
            st.subheader("Coin Flip Analogy")
            st.write("Think of a coin. When it sits on the table, it is Heads (0) or Tails (1). When you spin it, it is a blur of BOTH states at once. This spinning blur is **Superposition**.")
            is_spinning = st.checkbox("Spin the Coin!", value=True)
            spin_class = "spin" if is_spinning else ""
            st.markdown(f"<div class='coin-container'><div class='coin {spin_class}'><div class='coin-face coin-front'>0</div><div class='coin-face coin-back'>1</div></div></div>", unsafe_allow_html=True)
            
        with col2:
            st.subheader("Interactive Simulator")
            applied = st.button("✨ Apply Hadamard Gate")
            if applied or "hadamard_applied" in st.session_state:
                st.session_state.hadamard_applied = True
                st.markdown("<div style='border: 1px solid #10b981; background: rgba(16, 185, 129, 0.05); padding: 15px; border-radius: 8px; margin-bottom: 20px;'><h4 style='color: #10b981; margin: 0;'>Hadamard Applied!</h4>State transformed from <b>|0⟩</b> to <b>1/√2(|0⟩ + |1⟩)</b></div>", unsafe_allow_html=True)
                
                fig_probs = px.bar(x=["|0⟩", "|1⟩"], y=[0.5, 0.5], labels={'x': 'State', 'y': 'Probability'}, color=["|0⟩", "|1⟩"], color_discrete_sequence=["#38bdf8", "#ec4899"])
                fig_probs.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig_probs, use_container_width=True)
                
                fig_sv = go.Figure()
                fig_sv.add_trace(go.Scatter3d(x=[-1, 1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='#475569', width=1)))
                fig_sv.add_trace(go.Scatter3d(x=[0, 0], y=[-1, 1], z=[0, 0], mode='lines', line=dict(color='#475569', width=1)))
                fig_sv.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1, 1], mode='lines', line=dict(color='#475569', width=1)))
                fig_sv.add_trace(go.Scatter3d(x=[0, 1], y=[0, 0], z=[0, 0], mode='lines+markers', marker=dict(size=6, color='#10b981'), line=dict(color='#10b981', width=5)))
                fig_sv.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'), margin=dict(l=0, r=0, b=0, t=0), height=200, paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig_sv, use_container_width=True)
            else:
                st.info("Click 'Apply Hadamard Gate' to observe the transition.")

        st.write("")
        if st.session_state.beginner_mode:
            st.info("💡 **Social Network Context:** Superposition is like a person who belongs to multiple circles of friends at once. Until they pick one group to hang out with tonight (measurement), they exist in both paths.")
        else:
            st.info("💡 **Quantum Note:** The Hadamard gate $H$ maps basis state $|0\\rangle$ to equal superposition.")

    elif step == 3:
        st.header("Demo 3: Entanglement Demo")
        st.write("Entanglement couples two qubits so that the state of one instantly dictates the state of the other, no matter how far apart they are.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Bell State Creation")
            entangle_clicked = st.button("🔗 Create Bell Pair (Entangle)")
            if entangle_clicked or "entangled" in st.session_state:
                st.session_state.entangled = True
                if st.session_state.beginner_mode:
                    st.markdown("<div style='font-size: 1.2rem; font-weight: bold; color: #818cf8; text-align: center; margin: 20px 0;'>Entangled Connection: 50% both 0, 50% both 1</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='font-size: 1.5rem; font-weight: bold; color: #818cf8; text-align: center; margin: 20px 0;'>|Φ⁺⟩ = 1/√2(|00⟩ + |11⟩)</div>", unsafe_allow_html=True)
                    
                st.markdown("""
                <div style='display: flex; justify-content: center; align-items: center; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); padding: 30px; margin-top: 10px;'>
                    <div style='text-align: center;'><div class="pulse-glow" style='width: 40px; height: 40px; border-radius: 50%; background: #38bdf8; display: inline-block;'></div><p style='color: #38bdf8; font-weight: bold;'>Node A</p></div>
                    <div style='width: 120px; height: 2px; background: linear-gradient(90deg, #38bdf8 0%, #ec4899 100%); margin: 0 20px; position: relative;'>
                        <div style='position: absolute; width: 8px; height: 8px; border-radius: 50%; background: white; top: -3px; left: 0%; animation: particleMove 2s infinite linear;'></div>
                    </div>
                    <div style='text-align: center;'><div class="pulse-glow" style='width: 40px; height: 40px; border-radius: 50%; background: #ec4899; display: inline-block;'></div><p style='color: #ec4899; font-weight: bold;'>Node B</p></div>
                </div>
                <style>@keyframes particleMove { 0% { left: 0%; } 100% { left: 100%; } }</style>
                """, unsafe_allow_html=True)
            else:
                st.info("Click 'Create Bell Pair' to entangle Node A and Node B.")
                
        with col2:
            st.subheader("Measurement Correlations")
            if "entangled" in st.session_state:
                st.write("Measuring Node A instantly collapses Node B. Result statistics over 100 runs:")
                hist_data = pd.DataFrame({"State": ["00", "01", "10", "11"], "Probability": [0.5, 0.0, 0.0, 0.5]})
                fig_hist = px.bar(hist_data, x="State", y="Probability", color="State", color_discrete_sequence=["#38bdf8", "#64748b", "#64748b", "#ec4899"])
                fig_hist.update_layout(height=220, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
                st.write("Perfect coordination (00 or 11, never 01 or 10).")
            else:
                st.info("Awaiting entanglement.")
                
        st.write("")
        if st.session_state.beginner_mode:
            st.info("💡 **Social Network Context:** Entanglement represents a perfect correlation. If Person A buys a product (1), Person B is guaranteed to buy it (1).")
        else:
            st.info("💡 **Quantum Note:** A Bell state is a maximally entangled two-qubit state.")

    elif step == 4:
        show_grover_search_interactive_demo()

    elif step == 5:
        show_quantum_walk_explorer_demo()

    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    col_prev, col_spacer, col_next = st.columns([1, 3, 1])
    
    with col_prev:
        if step > 1:
            if st.button("← Previous Demo", key="w_prev_demo"):
                st.session_state.learn_step -= 1
                st.rerun()
    with col_next:
        if step < 5:
            if st.button("Next Demo →", key="w_next_demo"):
                st.session_state.learn_step += 1
                st.rerun()
        else:
            if st.button("Finish & Continue →", key="w_finish_demo"):
                st.session_state.page = "transition"
                st.rerun()

# ----------------- DEMO 4 & 5 IMPLEMENTATIONS -----------------

def show_grover_search_interactive_demo():
    st.header("Demo 4: Classical Search vs Quantum Search (Grover Algorithm)")
    
    st.markdown("""
    💡 **Workshop Analogy:**
    - **Classical search:** Imagine checking every user in a directory one-by-one to find a specific influencer.
    - **Quantum search:** We increase the probability of useful candidates globally using wave superposition and reflection.
    """)
    
    if "grover_target" not in st.session_state:
        st.session_state.grover_target = 3
    if "grover_n_users" not in st.session_state:
        st.session_state.grover_n_users = 16
    if "grover_iterations" not in st.session_state:
        st.session_state.grover_iterations = 0
    if "classical_step_run" not in st.session_state:
        st.session_state.classical_step_run = -1
        
    col_ctrl, col_stats = st.columns([1.2, 2.8])
    
    with col_ctrl:
        st.subheader("Controls")
        n_users = st.selectbox("Number of Users (N):", [4, 8, 16, 32], index=2, key="g_users_sel")
        if n_users != st.session_state.grover_n_users:
            st.session_state.grover_n_users = n_users
            st.session_state.grover_target = min(st.session_state.grover_target, n_users - 1)
            st.session_state.grover_iterations = 0
            st.session_state.classical_step_run = -1
            st.rerun()
            
        target = st.selectbox("Select Target Influencer Node:", list(range(n_users)), index=min(3, n_users-1), key="g_target_sel")
        if target != st.session_state.grover_target:
            st.session_state.grover_target = target
            st.session_state.grover_iterations = 0
            st.session_state.classical_step_run = -1
            st.rerun()
            
        optimal_iter = int(np.round(np.pi / 4.0 * np.sqrt(n_users)))
        st.markdown(f"**Optimal Grover Iterations:** {optimal_iter}")
        
        g_iter = st.slider("Grover Iteration Step:", 0, optimal_iter, st.session_state.grover_iterations, key="g_iter_slider")
        st.session_state.grover_iterations = g_iter
        
        col_run1, col_run2 = st.columns(2)
        if col_run1.button("🏃 Run Classical Search", key="g_run_classical_btn"):
            st.session_state.classical_step_run = 0
            
        if col_run2.button("🌀 Run Grover Search", key="g_run_grover_btn"):
            # Animate Grover steps
            placeholder_g_bar = st.empty()
            placeholder_g_hist = st.empty()
            for step in range(optimal_iter + 1):
                st.session_state.grover_iterations = step
                v, probs = simulate_grover(n_users, target, step)
                placeholder_g_bar.plotly_chart(make_grover_bars_plot(probs, target), use_container_width=True)
                placeholder_g_hist.plotly_chart(make_grover_hist_plot(probs, target), use_container_width=True)
                time.sleep(0.4)
            st.session_state.grover_iterations = optimal_iter
            st.rerun()
            
        st.write("---")
        show_circuit = st.button("🔌 Show Circuit Details", key="g_show_circuit_btn")
        if show_circuit:
            st.markdown("**Grover Search Schematic Circuit:**")
            circ_text = make_grover_circuit_text(n_users, target)
            st.code(circ_text, language="text")

    with col_stats:
        c_panel, q_panel = st.columns(2)
        with c_panel:
            st.subheader("Classical Search")
            st.markdown("<span style='font-size: 0.8rem; color: #ef4444; font-weight: bold;'>Complexity: O(N)</span>", unsafe_allow_html=True)
            
            if st.session_state.classical_step_run >= 0:
                placeholder_c = st.empty()
                for step in range(target + 1):
                    grid_html = "<div style='display: grid; grid-template-columns: repeat(4, 30px); gap: 4px;'>"
                    for i in range(n_users):
                        if i < step:
                            grid_html += f"<div style='width: 30px; height: 30px; line-height: 30px; text-align: center; border: 1px solid #ef4444; background: rgba(239, 68, 68, 0.15); font-size: 0.8rem;'>❌</div>"
                        elif i == step and i == target:
                            grid_html += f"<div style='width: 30px; height: 30px; line-height: 30px; text-align: center; border: 2px solid #10b981; background: rgba(16, 185, 129, 0.3); color: white; font-weight: bold; font-size: 0.8rem;'>🎯</div>"
                        elif i == step:
                            grid_html += f"<div style='width: 30px; height: 30px; line-height: 30px; text-align: center; border: 1px solid #ef4444; background: rgba(239, 68, 68, 0.2); font-size: 0.8rem;'>🔍</div>"
                        else:
                            grid_html += f"<div style='width: 30px; height: 30px; line-height: 30px; text-align: center; border: 1px solid #475569; opacity: 0.3; font-size: 0.8rem;'>{i}</div>"
                    grid_html += "</div>"
                    placeholder_c.markdown(grid_html, unsafe_allow_html=True)
                    time.sleep(0.12)
                st.session_state.classical_step_run = -1
                st.success(f"Classical Search found influencer in {target + 1} steps!")
            else:
                grid_html = "<div style='display: grid; grid-template-columns: repeat(4, 30px); gap: 4px;'>"
                for i in range(n_users):
                    grid_html += f"<div style='width: 30px; height: 30px; line-height: 30px; text-align: center; border: 1px solid #475569; opacity: 0.5; font-size: 0.8rem;'>{i}</div>"
                grid_html += "</div>"
                st.markdown(grid_html, unsafe_allow_html=True)
                
            st.markdown(f"**Classical Operations:** {target + 1}\n**Runtime Estimate:** {(target+1)*1.5:.1f} ms")
            
        with q_panel:
            st.subheader("Grover Search")
            st.markdown("<span style='font-size: 0.8rem; color: #10b981; font-weight: bold;'>Complexity: O(√N)</span>", unsafe_allow_html=True)
            v, probs = simulate_grover(n_users, target, st.session_state.grover_iterations)
            
            st.plotly_chart(make_grover_bars_plot(probs, target), use_container_width=True)
            st.plotly_chart(make_grover_hist_plot(probs, target), use_container_width=True)
            
            st.markdown(f"**Grover Iteration:** {st.session_state.grover_iterations}\n**Target Node P({target}):** {probs[target]:.1%}")
            
    st.info("💡 **Explanation:** Grover amplifies the probability of desired nodes constructively rather than checking one-by-one. In social network context, it finds key influencers rapidly.")

def show_quantum_walk_explorer_demo():
    st.header("5-Node Quantum Walk Explorer")
    
    st.markdown("""
    💡 **Workshop Analogy:**
    - **Classical walk:** Explores paths one by one (slow diffusion, like a rumor spreading locally).
    - **Quantum walk:** Explores multiple paths simultaneously (rumor spreading like waves, creating peaks via interference).
    """)
    
    if "walk_start_node" not in st.session_state:
        st.session_state.walk_start_node = 0
    if "walk_time" not in st.session_state:
        st.session_state.walk_time = 0.0
        
    col_ctrl, col_graph = st.columns([1.2, 2.8])
    with col_ctrl:
        st.subheader("Controls")
        start_node = st.selectbox("Select Start Node:", [0, 1, 2, 3, 4], index=st.session_state.walk_start_node, key="qw_start_sel")
        st.session_state.walk_start_node = start_node
        
        t_slider = st.slider("Continuous Time Slider (t):", 0.0, 10.0, st.session_state.walk_time, 0.1, key="qw_time_slider")
        st.session_state.walk_time = t_slider
        
        st.markdown(f"**Corresponding Classical Steps:** {int(st.session_state.walk_time)}")
        
        if st.button("▶️ Play Animation", key="qw_play_btn"):
            placeholder_g = st.empty()
            placeholder_c = st.empty()
            placeholder_e = st.empty()
            
            for t_val in np.linspace(0.0, 10.0, 40):
                st.session_state.walk_time = t_val
                q_probs = simulate_quantum_walk(start_node, t_val)
                c_probs = simulate_classical_walk(start_node, int(t_val))
                
                placeholder_g.plotly_chart(make_walk_graph_plot(q_probs), use_container_width=True)
                placeholder_c.plotly_chart(make_walk_chart_plot(c_probs, q_probs), use_container_width=True)
                placeholder_e.markdown(get_walk_explanation_text(q_probs), unsafe_allow_html=True)
                time.sleep(0.06)
            st.session_state.walk_time = 10.0
            st.rerun()

    with col_graph:
        t_val = st.session_state.walk_time
        q_probs = simulate_quantum_walk(start_node, t_val)
        c_probs = simulate_classical_walk(start_node, int(t_val))
        
        tab_g, tab_c = st.tabs(["Animated Graph View", "Side-by-Side Comparison Chart"])
        with tab_g:
            st.plotly_chart(make_walk_graph_plot(q_probs), use_container_width=True)
        with tab_c:
            st.plotly_chart(make_walk_chart_plot(c_probs, q_probs), use_container_width=True)
            
        st.markdown(get_walk_explanation_text(q_probs), unsafe_allow_html=True)

# ----------------- TRANSITION & CURRICULUM SCENES -----------------

def show_transition_page():
    inject_onboarding_styles()
    st.header("Transitioning to Social Network Analysis")
    st.write("Now that you understand quantum fundamentals, let's see how they directly map to social networks:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quantum Concept")
        st.markdown("- **Qubit**\n- **Superposition**\n- **Entanglement**\n- **Quantum Walk**\n- **Measurement**")
    with col2:
        st.subheader("Social Network Equivalent")
        st.markdown("- **Node Representation** (A single person)\n- **Multiple Network Paths** (Exploring all connections at once)\n- **Correlated Users** (Connected friends influencing each other)\n- **Efficient Network Exploration** (Scanning paths instantly)\n- **Extracting Insights** (Collapsing quantum states to get answers)")
        
    st.write("---")
    st.subheader("QSNA Visual Processing Pipeline")
    st.markdown("""
    <div style='display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 25px; border-radius: 12px;'>
        <div style='flex: 1; text-align: center; min-width: 100px; padding: 10px;'><div style='font-size: 2rem;'>📊</div><div style='font-weight: bold; color: #38bdf8;'>1. Social Graph</div><span style='font-size: 0.8rem; color: #94a3b8;'>Users & connections</span></div>
        <div style='color: #475569;'>➔</div>
        <div style='flex: 1; text-align: center; min-width: 100px; padding: 10px;'><div style='font-size: 2rem;'>🔲</div><div style='font-weight: bold; color: #818cf8;'>2. Matrix</div><span style='font-size: 0.8rem; color: #94a3b8;'>Adjacency mapping</span></div>
        <div style='color: #475569;'>➔</div>
        <div style='flex: 1; text-align: center; min-width: 100px; padding: 10px;'><div style='font-size: 2rem;'>⚛️</div><div style='font-weight: bold; color: #a855f7;'>3. Encoding</div><span style='font-size: 0.8rem; color: #94a3b8;'>Convert to Qubits</span></div>
        <div style='color: #475569;'>➔</div>
        <div style='flex: 1; text-align: center; min-width: 100px; padding: 10px;'><div style='font-size: 2rem;'>🎛️</div><div style='font-weight: bold; color: #ec4899;'>4. Circuit</div><span style='font-size: 0.8rem; color: #94a3b8;'>Quantum processing</span></div>
        <div style='color: #475569;'>➔</div>
        <div style='flex: 1; text-align: center; min-width: 100px; padding: 10px;'><div style='font-size: 2rem;'>📈</div><div style='font-weight: bold; color: #10b981;'>5. Analysis</div><span style='font-size: 0.8rem; color: #94a3b8;'>Centrality & Communities</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("<br><br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1])
    with col_left:
        if st.button("🏠 Return to Landing Page", key="btn_trans_home"):
            st.session_state.page = "landing"
            st.rerun()
    with col_right:
        if st.button("🚀 Continue to QSNA Dashboard", type="primary", key="btn_trans_dash", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()

def show_challenges_page():
    inject_onboarding_styles()
    st.header("🏆 Quantum Network Challenges")
    st.write("Put your intuition to the test! Try to solve these network problems before revealing how quantum algorithms help.")
    
    tab1, tab2, tab3 = st.tabs(["Challenge 1: Find Influencer", "Challenge 2: Detect Communities", "Challenge 3: Rumor Defense"])
    
    with tab1:
        st.subheader("Identify the Most Influential Node")
        st.write("Below is a small social network of 6 people. One person connects separate sub-groups and holds the network together. Who is it?")
        edges = [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)]
        pos_x = [0.0, -0.8, -0.6, 1.0, 1.6, 1.4]
        pos_y = [0.0, 0.5, -0.6, 0.0, 0.6, -0.6]
        
        fig = go.Figure()
        for u, v in edges:
            fig.add_trace(go.Scatter(x=[pos_x[u], pos_x[v]], y=[pos_y[u], pos_y[v]], mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=pos_x, y=pos_y, mode='markers+text', marker=dict(size=30, color='#818cf8', line=dict(color='white', width=1)), text=[f"Person {i}" for i in range(6)], textposition="top center", showlegend=False))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=250, margin=dict(l=10,r=10,b=10,t=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        guess = st.selectbox("Guess the most influential person:", [f"Person {i}" for i in range(6)])
        if st.button("Check Guess", key="check_1"):
            if guess == "Person 0" or guess == "Person 3":
                st.success(f"Correct! **{guess}** acts as the crucial bridge (high betweenness and degree centrality).")
                st.info("💡 **Quantum Advantage:** Grover-based centrality search and continuous quantum walks can find these bridge nodes in $O(\\sqrt{N})$ time.")
            else:
                st.error(f"Not quite! Try checking Person 0 or Person 3.")
                
    with tab2:
        st.subheader("Detect Social Communities")
        st.write("How many distinct close-knit communities (clusters) do you see in the network below?")
        edges_2 = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (6, 7), (7, 8), (8, 6), (2, 3), (5, 6)]
        px_2 = [0, -0.5, 0.5,  2, 1.5, 2.5,  4, 3.5, 4.5]
        py_2 = [1, 0, 0,      1, 0, 0,      1, 0, 0]
        
        fig2 = go.Figure()
        for u, v in edges_2:
            fig2.add_trace(go.Scatter(x=[px_2[u], px_2[v]], y=[py_2[u], py_2[v]], mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=2), showlegend=False))
        fig2.add_trace(go.Scatter(x=px_2, y=py_2, mode='markers+text', marker=dict(size=25, color='#ec4899', line=dict(color='white', width=1)), text=[str(i) for i in range(9)], textposition="top center", showlegend=False))
        fig2.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=250, margin=dict(l=10,r=10,b=10,t=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)
        
        comm_guess = st.slider("Number of Communities:", 1, 5, 1)
        if st.button("Check Community Guess", key="check_2"):
            if comm_guess == 3:
                st.success("Correct! There are 3 distinct triangles (communities) connected by weak bridges.")
                st.info("💡 **Quantum Advantage:** Quantum modularity optimization allows quantum processors to find optimal clusterings simultaneously.")
            else:
                st.error(f"Incorrect. You guessed {comm_guess}. Look at the triangles.")
                
    with tab3:
        st.subheader("Predict & Block Misinformation Spread")
        st.write("A rumor starts at Node 0. To stop it from reaching Node 4, you can block (vaccinate) one node. Who should you block?")
        edges_3 = [(0, 1), (1, 2), (2, 4), (0, 3), (3, 4)]
        px_3 = [0, 1, 2, 1, 3]
        py_3 = [0, 1, 1, -1, 0]
        
        fig3 = go.Figure()
        for u, v in edges_3:
            fig3.add_trace(go.Scatter(x=[px_3[u], px_3[v]], y=[py_3[u], py_3[v]], mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=2), showlegend=False))
        fig3.add_trace(go.Scatter(x=px_3, y=py_3, mode='markers+text', marker=dict(size=30, color='#10b981', line=dict(color='white', width=1)), text=[f"Node {i}" for i in range(5)], textposition="top center", showlegend=False))
        fig3.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=250, margin=dict(l=10,r=10,b=10,t=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)
        
        vaccine_guess = st.selectbox("Select node to block:", [1, 2, 3])
        if st.button("Check Misinformation Guess", key="check_3"):
            if vaccine_guess == 3:
                st.success("Excellent! Blocking Node 3 cuts off the shortest path directly.")
            else:
                st.warning(f"Blocking Node {vaccine_guess} helps, but Node 3 is the most critical pipeline.")

    st.write("<br><br>", unsafe_allow_html=True)
    if st.button("🏠 Back to Home", key="btn_challenges_home"):
        st.session_state.page = "landing"
        st.rerun()

# Metadata of improved demo datasets (Real-world Application Cards)
DEMO_DATASETS_META = {
    "Misinformation Detection": {
        "desc": "A scale-free propagation network. Model rumor spreading from central 'hubs' that amplify incoming misinformation to the group.",
        "nodes": 50, "edges": 95, "density": 0.078, "clustering": 0.22, "avg_degree": 3.8
    },
    "Friendship Networks": {
        "desc": "A social network representing modular friendship circles. Showcases high local clustering coefficient and short average communication paths.",
        "nodes": 34, "edges": 78, "density": 0.139, "clustering": 0.57, "avg_degree": 4.58
    },
    "Twitter Interactions": {
        "desc": "A directed follow-and-retweet network exhibiting asymmetric relationship structures and follower-leader patterns.",
        "nodes": 45, "edges": 120, "density": 0.061, "clustering": 0.31, "avg_degree": 5.33
    },
    "Citation Networks": {
        "desc": "A directed acyclic graph mapping paper references, illustrating research authority pipeline dynamics.",
        "nodes": 40, "edges": 85, "density": 0.054, "clustering": 0.18, "avg_degree": 4.25
    },
    "Brain Connectivity": {
        "desc": "A dense structural map representing neural signals between different brain sectors. Very high density, fast information synchronization.",
        "nodes": 28, "edges": 142, "density": 0.376, "clustering": 0.61, "avg_degree": 10.14
    },
    "Financial Fraud": {
        "desc": "A transaction connection graph featuring circular patterns, indicating suspected money laundering behavior.",
        "nodes": 32, "edges": 64, "density": 0.129, "clustering": 0.05, "avg_degree": 4.0
    },
    "Disease Spread": {
        "desc": "A lattice-structured network mapping spatial physical proximity, suitable for tracing epidemic SIR transmissions.",
        "nodes": 40, "edges": 75, "density": 0.096, "clustering": 0.42, "avg_degree": 3.75
    },
    "Influencer Discovery": {
        "desc": "A star-like hub network where central social leaders hold dominant centrality scores.",
        "nodes": 35, "edges": 80, "density": 0.134, "clustering": 0.15, "avg_degree": 4.57
    }
}

def show_demo_datasets_page():
    inject_onboarding_styles()
    st.header("📂 Real-World Application Cards")
    st.write("Browse pre-loaded network datasets. Select a dataset to view its profile, load the graph, and analyze it.")
    
    keys = list(DEMO_DATASETS_META.keys())
    for i in range(0, len(keys), 2):
        col1, col2 = st.columns(2)
        with col1:
            k1 = keys[i]
            meta1 = DEMO_DATASETS_META[k1]
            st.markdown(f"""
            <div class="glass-card" style='border-left: 5px solid #38bdf8;'>
                <h4>{k1}</h4>
                <p style='font-size: 0.9rem; color: #cbd5e1; min-height: 50px;'>{meta1["desc"]}</p>
                <span style='font-size: 0.8rem; color: #94a3b8;'>Nodes: {meta1["nodes"]} | Edges: {meta1["edges"]} | Density: {meta1["density"]:.3f}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Launch {k1}", key=f"launch_{k1}"):
                load_and_run_dataset(k1, meta1)
        with col2:
            if i + 1 < len(keys):
                k2 = keys[i+1]
                meta2 = DEMO_DATASETS_META[k2]
                st.markdown(f"""
                <div class="glass-card" style='border-left: 5px solid #a855f7;'>
                    <h4>{k2}</h4>
                    <p style='font-size: 0.9rem; color: #cbd5e1; min-height: 50px;'>{meta2["desc"]}</p>
                    <span style='font-size: 0.8rem; color: #94a3b8;'>Nodes: {meta2["nodes"]} | Edges: {meta2["edges"]} | Density: {meta2["density"]:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Launch {k2}", key=f"launch_{k2}"):
                    load_and_run_dataset(k2, meta2)
                    
    st.write("<br><br>", unsafe_allow_html=True)
    if st.button("🏠 Back to Home", key="btn_app_cards_home"):
        st.session_state.page = "landing"
        st.rerun()

def load_and_run_dataset(name, meta):
    if name == "Friendship Networks":
        st.session_state.G = nx.karate_club_graph()
    elif name == "Financial Fraud":
        G = nx.cycle_graph(8)
        G.add_edges_from([(8,9), (9,10), (10,11), (11,8), (0,8), (4,10)])
        st.session_state.G = G
    elif name == "Brain Connectivity":
        st.session_state.G = nx.erdos_renyi_graph(28, 0.35)
    elif name == "Misinformation Detection":
        st.session_state.G = nx.barabasi_albert_graph(50, 2)
    else:
        st.session_state.G = nx.watts_strogatz_graph(meta["nodes"], 4, 0.1)
        
    st.success(f"Successfully loaded '{name}' network!")
    st.session_state.page = "dashboard"
    st.session_state.dashboard_step = "2. Graph Dashboard"
    st.rerun()

# =================================================================
# 5. WORKSHOP, STORY MODE & ADVANCED SUB-DEMOS
# =================================================================

def show_workshop_mode():
    inject_onboarding_styles()
    
    st.markdown("<h1 style='text-align: center;'>🏫 QSNA Guided Workshop</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem;'>Learn how quantum physics transforms network analysis from the ground up.</p>", unsafe_allow_html=True)
    
    # Progress indicator
    steps = [
        "1. Debunk Myths",
        "2. Learn Fundamentals",
        "3. Search Race",
        "4. Network Stories",
        "5. Graduation"
    ]
    
    cols = st.columns(len(steps))
    current_step = st.session_state.workshop_step
    
    for idx, name in enumerate(steps):
        is_active = (idx + 1 == current_step)
        is_done = (idx + 1 < current_step)
        color = "#38bdf8" if is_active else ("#10b981" if is_done else "#475569")
        font_weight = "bold" if is_active else "normal"
        cols[idx].markdown(
            f"<div style='text-align: center; border-bottom: 4px solid {color}; padding-bottom: 8px; color: {color}; font-weight: {font_weight}; font-size: 0.9rem;'>{name}</div>", 
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render Step Content
    if current_step == 1:
        st.subheader("Step 1: Debunking Quantum Myths")
        st.write("Before we start building circuits, let's clear up some common misconceptions about quantum computing in social networks.")
        show_myths_panel()
        
    elif current_step == 2:
        st.subheader("Step 2: Interactive Fundamentals")
        st.write("Interact with the core quantum building blocks. Step through all 5 interactive modules:")
        
        sub_step = st.session_state.learn_step
        st.markdown(f"**Current Concept: Demo {sub_step} of 5**")
        
        # Render the specific step content
        if sub_step == 1:
            st.markdown("### Demo 1: Classical Bit vs Qubit")
            st.write("In classical networks, a person is either connected (1) or disconnected (0). In quantum representations, a node's state can exist as a qubit.")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Classical State")
                bit_val = st.radio("Choose classical Bit state:", [0, 1], horizontal=True, key="ws_bit_val")
                st.markdown(f"<div style='font-size: 4rem; text-align: center; font-weight: 800; color: #38bdf8; margin: 20px 0;'>{bit_val}</div>", unsafe_allow_html=True)
                st.write("Classical systems exist in one state. A bit is strictly `0` or `1`. There is no uncertainty before measurement.")
            with col2:
                st.subheader("Quantum State")
                p0 = st.slider("Probability of |0⟩ (P(|0⟩))", 0.0, 1.0, 0.5, 0.05, key="ws_p0")
                p1 = 1.0 - p0
                phase_deg = st.slider("Phase Angle (degrees)", 0, 360, 0, 15, key="ws_phase")
                phase_rad = math.radians(phase_deg)
                alpha_mag = math.sqrt(p0)
                beta_mag = math.sqrt(p1)
                if st.session_state.beginner_mode:
                    st.markdown(f"<div style='font-size: 1.5rem; text-align: center; font-weight: bold; color: #a855f7; margin: 20px 0;'>Possibility 0: {p0:.0%} | Possibility 1: {p1:.0%}</div>", unsafe_allow_html=True)
                else:
                    phase_exp = f"e^(i*{phase_deg}°)" if phase_deg != 0 else ""
                    st.markdown(f"<div style='font-size: 1.8rem; text-align: center; font-weight: bold; color: #a855f7; margin: 20px 0;'>{alpha_mag:.3f}|0⟩ + {beta_mag:.3f}{phase_exp}|1⟩</div>", unsafe_allow_html=True)
                theta = 2 * math.acos(alpha_mag)
                x = math.sin(theta) * math.cos(phase_rad)
                y = math.sin(theta) * math.sin(phase_rad)
                z = math.cos(theta)
                fig = go.Figure()
                u_grid = np.linspace(0, 2 * np.pi, 30)
                v_grid = np.linspace(0, np.pi, 15)
                for u in u_grid:
                    fig.add_trace(go.Scatter3d(x=np.sin(v_grid)*np.cos(u), y=np.sin(v_grid)*np.sin(u), z=np.cos(v_grid), mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1), showlegend=False))
                for v in v_grid:
                    fig.add_trace(go.Scatter3d(x=np.sin(v)*np.cos(u_grid), y=np.sin(v)*np.sin(u_grid), z=np.cos(v)*np.ones_like(u_grid), mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=1), showlegend=False))
                fig.add_trace(go.Scatter3d(x=[-1, 1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='#475569', width=2), name="X"))
                fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1, 1], z=[0, 0], mode='lines', line=dict(color='#475569', width=2), name="Y"))
                fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1, 1], mode='lines', line=dict(color='#475569', width=2), name="Z"))
                fig.add_trace(go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode='lines+markers', marker=dict(size=6, color='#a855f7'), line=dict(color='#a855f7', width=5), name="Qubit"))
                fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'), margin=dict(l=0, r=0, b=0, t=0), height=300, paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.write("---")
                if st.button("⚡ Measure Qubit", key="ws_measure_btn"):
                    st.session_state.ws_q_outcome = 0 if random.random() < p0 else 1
                if "ws_q_outcome" in st.session_state:
                    collapsed_symbol = "|0⟩" if st.session_state.ws_q_outcome == 0 else "|1⟩" if st.session_state.ws_q_outcome == 1 and not st.session_state.beginner_mode else f"Possibility {st.session_state.ws_q_outcome}"
                    st.markdown(f"<div class='pulse-glow' style='background: rgba(168, 85, 247, 0.1); border: 1px solid #a855f7; border-radius: 8px; padding: 15px; text-align: center; font-weight: bold;'>Collapsed State: {collapsed_symbol}!</div>", unsafe_allow_html=True)
                st.write("")
                if st.session_state.beginner_mode:
                    st.info("💡 **Analogy:** Unlike a light switch that is either OFF (0) or ON (1), a quantum state is like a dimmer switch that can represent multiple levels at the same time until measured.")
                else:
                    st.info("💡 **Quantum Note:** A qubit is in a linear superposition of basis states.")
        elif sub_step == 2:
            st.markdown("### Demo 2: Superposition")
            st.write("Superposition allows a quantum state to represent multiple values simultaneously. We create superposition using a **Hadamard Gate**.")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Circuit")
                st.markdown("<div style='background: rgba(255,255,255,0.02); padding: 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); text-align: center;'><span style='font-family: monospace; font-size: 1.2rem; color: #818cf8;'>|0⟩ ───[ H ]─── 1/√2(|0⟩ + |1⟩)</span></div>", unsafe_allow_html=True)
                st.subheader("Coin Flip Analogy")
                st.write("Think of a coin. When it sits on the table, it is Heads (0) or Tails (1). When you spin it, it is a blur of BOTH states at once. This spinning blur is **Superposition**.")
                is_spinning = st.checkbox("Spin the Coin!", value=True, key="ws_spin")
                spin_class = "spin" if is_spinning else ""
                st.markdown(f"<div class='coin-container'><div class='coin {spin_class}'><div class='coin-face coin-front'>0</div><div class='coin-face coin-back'>1</div></div></div>", unsafe_allow_html=True)
            with col2:
                st.subheader("Interactive Simulator")
                applied = st.button("✨ Apply Hadamard Gate", key="ws_h_btn")
                if applied or "ws_hadamard_applied" in st.session_state:
                    st.session_state.ws_hadamard_applied = True
                    st.markdown("<div style='border: 1px solid #10b981; background: rgba(16, 185, 129, 0.05); padding: 15px; border-radius: 8px; margin-bottom: 20px;'><h4 style='color: #10b981; margin: 0;'>Hadamard Applied!</h4>State transformed from <b>|0⟩</b> to <b>1/√2(|0⟩ + |1⟩)</b></div>", unsafe_allow_html=True)
                    fig_probs = px.bar(x=["|0⟩", "|1⟩"], y=[0.5, 0.5], labels={'x': 'State', 'y': 'Probability'}, color=["|0⟩", "|1⟩"], color_discrete_sequence=["#38bdf8", "#ec4899"])
                    fig_probs.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                    st.plotly_chart(fig_probs, use_container_width=True)
                else:
                    st.info("Click 'Apply Hadamard Gate' to observe the transition.")
        elif sub_step == 3:
            st.markdown("### Demo 3: Entanglement")
            st.write("Entanglement couples two qubits so that the state of one instantly dictates the state of the other, no matter how far apart they are.")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Bell State Creation")
                entangle_clicked = st.button("🔗 Create Bell Pair (Entangle)", key="ws_ent_btn")
                if entangle_clicked or "ws_entangled" in st.session_state:
                    st.session_state.ws_entangled = True
                    if st.session_state.beginner_mode:
                        st.markdown("<div style='font-size: 1.2rem; font-weight: bold; color: #818cf8; text-align: center; margin: 20px 0;'>Entangled Connection: 50% both 0, 50% both 1</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='font-size: 1.5rem; font-weight: bold; color: #818cf8; text-align: center; margin: 20px 0;'>|Φ⁺⟩ = 1/√2(|00⟩ + |11⟩)</div>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style='display: flex; justify-content: center; align-items: center; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); padding: 30px; margin-top: 10px;'>
                        <div style='text-align: center;'><div class="pulse-glow" style='width: 40px; height: 40px; border-radius: 50%; background: #38bdf8; display: inline-block;'></div><p style='color: #38bdf8; font-weight: bold;'>Node A</p></div>
                        <div style='width: 120px; height: 2px; background: linear-gradient(90deg, #38bdf8 0%, #ec4899 100%); margin: 0 20px; position: relative;'>
                            <div style='position: absolute; width: 8px; height: 8px; border-radius: 50%; background: white; top: -3px; left: 0%; animation: particleMove 2s infinite linear;'></div>
                        </div>
                        <div style='text-align: center;'><div class="pulse-glow" style='width: 40px; height: 40px; border-radius: 50%; background: #ec4899; display: inline-block;'></div><p style='color: #ec4899; font-weight: bold;'>Node B</p></div>
                    </div>
                    """, unsafe_allow_html=True)
            with col2:
                st.subheader("Measurement Correlations")
                if "ws_entangled" in st.session_state:
                    st.write("Measuring Node A instantly collapses Node B. Result statistics over 100 runs:")
                    hist_data = pd.DataFrame({"State": ["00", "01", "10", "11"], "Probability": [0.5, 0.0, 0.0, 0.5]})
                    fig_hist = px.bar(hist_data, x="State", y="Probability", color="State", color_discrete_sequence=["#38bdf8", "#64748b", "#64748b", "#ec4899"])
                    fig_hist.update_layout(height=220, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("Awaiting entanglement.")
        elif sub_step == 4:
            show_grover_search_interactive_demo()
        elif sub_step == 5:
            show_quantum_walk_explorer_demo()
            
        # Stepper buttons for fundamentals sub-demos
        st.write("")
        col_ws_prev, col_ws_spacer, col_ws_next = st.columns([1, 3, 1])
        with col_ws_prev:
            if sub_step > 1:
                if st.button("← Previous Demo", key="ws_sub_prev"):
                    st.session_state.learn_step -= 1
                    st.rerun()
        with col_ws_next:
            if sub_step < 5:
                if st.button("Next Demo →", key="ws_sub_next"):
                    st.session_state.learn_step += 1
                    st.rerun()
            else:
                st.success("You've completed all 5 fundamentals demos! Proceed to the next main step.")
                
    elif current_step == 3:
        st.subheader("Step 3: Classical vs Quantum Search Race")
        st.write("Witness a direct execution speed comparison between classical sequential searching and quantum Grover search.")
        show_classical_quantum_race()
        
    elif current_step == 4:
        st.subheader("Step 4: Social Network Story Mode")
        st.write("Explore narrative scenarios illustrating how quantum states and entanglement map onto social networking dynamics.")
        show_story_mode()
        
    elif current_step == 5:
        st.subheader("Step 5: Workshop Graduation")
        st.markdown("""
        ### Congratulations! 🎉
        You have successfully completed the QSNA Quantum Workshop. You have learned:
        1. **Quantum Superposition** lets us explore multiple social network paths at the same time.
        2. **Quantum Entanglement** maps tightly coupled social relationships and coordinated actions.
        3. **Grover's Algorithm** speeds up key influencer search from $O(N)$ to $O(\\sqrt{N})$.
        4. **Quantum Walks** propagate information as a probability wave, utilizing constructive interference to pinpoint central influencers.
        """)
        
        st.write("---")
        st.subheader("🔍 Estimating Hardware Resources for Your Growth")
        n_est = st.slider("Estimate Qubits for a Social Graph of Size (N):", 5, 200, 30, key="ws_est_slider")
        show_resource_estimator(n_est)
        
        st.write("<br>", unsafe_allow_html=True)
        if st.button("🚀 Enter Advanced Research Mode", type="primary", key="ws_grad_btn", use_container_width=True):
            st.session_state.page = "dashboard"
            st.session_state.dashboard_step = "1. Data Input"
            st.rerun()
            
    # Bottom Navigation for Main Steps
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    col_prev, col_spacer, col_next = st.columns([1.2, 3, 1.2])
    with col_prev:
        if current_step > 1:
            if st.button("← Previous Step", key="ws_prev_step", use_container_width=True):
                st.session_state.workshop_step -= 1
                st.rerun()
        else:
            if st.button("🏠 Exit Workshop", key="ws_exit_btn", use_container_width=True):
                st.session_state.page = "landing"
                st.rerun()
    with col_next:
        if current_step < 5:
            if st.button("Next Step →", key="ws_next_step", use_container_width=True):
                st.session_state.workshop_step += 1
                st.rerun()
        else:
            if st.button("🏠 Exit Workshop", key="ws_exit_btn_grad", use_container_width=True):
                st.session_state.page = "landing"
                st.rerun()

def show_myths_panel():
    st.subheader("💡 Debunking Quantum Myths (Hover or Click to Flip)")
    myths_html = """
    <div class="flip-card-container">
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <span style="font-size: 2.5rem;">🔮</span>
                    <h5 style="margin-top: 10px; color: #f59e0b;">Myth #1</h5>
                    <p style="font-size: 0.9rem;">"Quantum computers are just super-fast regular computers."</p>
                </div>
                <div class="flip-card-back">
                    <h5 style="color: #38bdf8;">The Reality</h5>
                    <p style="font-size: 0.85rem; line-height: 1.4;">Quantum computers aren't just faster; they run completely different types of algorithms using qubits, allowing them to solve certain complex network routing and optimization problems that are impossible for classical machines.</p>
                </div>
            </div>
        </div>
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <span style="font-size: 2.5rem;">🚶‍♂️</span>
                    <h5 style="margin-top: 10px; color: #f59e0b;">Myth #2</h5>
                    <p style="font-size: 0.9rem;">"A quantum walk is just a normal random walk."</p>
                </div>
                <div class="flip-card-back">
                    <h5 style="color: #38bdf8;">The Reality</h5>
                    <p style="font-size: 0.85rem; line-height: 1.4;">Classical walks hop from node to node randomly. Quantum walks propagate as a probability wave, enabling constructive wave interference that makes them find target connections quadratically faster.</p>
                </div>
            </div>
        </div>
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <span style="font-size: 2.5rem;">🌐</span>
                    <h5 style="margin-top: 10px; color: #f59e0b;">Myth #3</h5>
                    <p style="font-size: 0.9rem;">"We can't map social networks to quantum systems."</p>
                </div>
                <div class="flip-card-back">
                    <h5 style="color: #38bdf8;">The Reality</h5>
                    <p style="font-size: 0.85rem; line-height: 1.4;">By representing the network adjacency matrix as a Hamiltonian operator, or mapping nodes to qubit basis states, we can run network analytics like centrality directly on quantum hardware!</p>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(myths_html, unsafe_allow_html=True)

def show_classical_quantum_race():
    st.subheader("🏁 The Race: Classical Sequential vs Grover Quantum Wave")
    st.write("Click 'Start Race' to watch classical search find the target node vs Grover's wave amplification.")
    
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body {
        background: transparent;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
    }
    .race-box {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 20px;
        justify-content: center;
    }
    .lane {
        flex: 1;
        min-width: 280px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.03);
    }
    .lane-title {
        font-weight: bold;
        margin-bottom: 15px;
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(4, 35px);
        gap: 6px;
        justify-content: center;
        margin-bottom: 15px;
    }
    .dot {
        width: 35px;
        height: 35px;
        line-height: 35px;
        font-size: 0.75rem;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    .dot.target {
        border-color: #f59e0b;
        color: #f59e0b;
    }
    .dot.scan {
        background: rgba(239, 68, 68, 0.3) !important;
        border-color: #ef4444 !important;
        color: white;
        transform: scale(1.1);
    }
    .dot.found {
        background: rgba(16, 185, 129, 0.4) !important;
        border-color: #10b981 !important;
        color: white;
        font-weight: bold;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
    }
    .dot.quantum-active {
        background: rgba(56, 189, 248, 0.15);
        border-color: #38bdf8;
        transform: scale(1.05);
    }
    .dot.quantum-target-grow {
        background: rgba(16, 185, 129, 0.4) !important;
        border-color: #10b981 !important;
        color: white;
        transform: scale(1.2);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.8);
    }
    button.btn {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        border: none;
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-top: 10px;
    }
    button.btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(139, 92, 246, 0.4);
    }
    .stats {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    </style>
    </head>
    <body>
    <div class="race-box">
        <div class="lane" id="c-lane">
            <div class="lane-title" style="color: #ef4444;">Classical Sequential Search</div>
            <div class="grid" id="c-grid"></div>
            <div class="stats" id="c-stats">Steps: 0 | Time: 0s</div>
        </div>
        <div class="lane" id="q-lane">
            <div class="lane-title" style="color: #38bdf8;">Grover Quantum Search</div>
            <div class="grid" id="q-grid"></div>
            <div class="stats" id="q-stats">Steps: 0 | Time: 0s</div>
        </div>
    </div>
    <div style="text-align: center; margin-top: 15px;">
        <button class="btn" id="start-btn">Start Race</button>
    </div>
    
    <script>
    const n = 16;
    const target = 13;
    const cGrid = document.getElementById('c-grid');
    const qGrid = document.getElementById('q-grid');
    const startBtn = document.getElementById('start-btn');
    
    for (let i = 0; i < n; i++) {
        const cDot = document.createElement('div');
        cDot.className = 'dot' + (i === target ? ' target' : '');
        cDot.innerText = i;
        cDot.id = 'c-' + i;
        cGrid.appendChild(cDot);
        
        const qDot = document.createElement('div');
        qDot.className = 'dot' + (i === target ? ' target' : '');
        qDot.innerText = i;
        qDot.id = 'q-' + i;
        qGrid.appendChild(qDot);
    }
    
    startBtn.addEventListener('click', () => {
        startBtn.disabled = true;
        runRace();
    });
    
    function runRace() {
        for (let i = 0; i < n; i++) {
            document.getElementById('c-' + i).className = 'dot' + (i === target ? ' target' : '');
            document.getElementById('q-' + i).className = 'dot' + (i === target ? ' target' : '');
            document.getElementById('q-' + i).style.opacity = '1';
            document.getElementById('q-' + i).style.borderColor = '';
        }
        
        let cStep = 0;
        
        const cInterval = setInterval(() => {
            if (cStep < n) {
                const prev = document.getElementById('c-' + (cStep - 1));
                if (prev) prev.classList.remove('scan');
                
                const cur = document.getElementById('c-' + cStep);
                if (cStep === target) {
                    cur.className = 'dot found';
                    document.getElementById('c-stats').innerHTML = `Steps: ${cStep + 1} | Found!`;
                    clearInterval(cInterval);
                } else {
                    cur.classList.add('scan');
                    document.getElementById('c-stats').innerHTML = `Steps: ${cStep + 1} | Searching...`;
                    cStep++;
                }
            }
        }, 200);
        
        setTimeout(() => {
            for (let i = 0; i < n; i++) {
                document.getElementById('q-' + i).classList.add('quantum-active');
            }
            document.getElementById('q-stats').innerHTML = `Steps: 1 (Superposition) | Time: 0.1s`;
        }, 150);
        
        setTimeout(() => {
            document.getElementById('q-' + target).style.borderColor = '#ec4899';
            document.getElementById('q-stats').innerHTML = `Steps: 2 (Oracle flip) | Time: 0.2s`;
        }, 400);
        
        setTimeout(() => {
            for (let i = 0; i < n; i++) {
                if (i !== target) {
                    document.getElementById('q-' + i).style.opacity = '0.4';
                }
            }
            document.getElementById('q-' + target).className = 'dot quantum-target-grow';
            document.getElementById('q-stats').innerHTML = `Steps: 3 (Diffusion) | Time: 0.3s`;
        }, 700);
        
        setTimeout(() => {
            document.getElementById('q-' + target).className = 'dot found';
            document.getElementById('q-stats').innerHTML = `Steps: 3 (Optimal) | Found in 3 iterations!`;
            startBtn.disabled = false;
        }, 1000);
    }
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=350)

def show_story_mode():
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.02); border-radius: 12px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.05);'>
        <p style='color: #94a3b8; font-size: 0.95rem;'>Social Network Story Mode maps complex quantum concepts to everyday social interactions. Explore these real-world scenarios:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card" style='border-top: 4px solid #38bdf8; height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>
            <div>
                <h4 style='color: #38bdf8; margin-top: 0;'>Scenario 1: The Viral Rumor</h4>
                <p style='font-size: 0.85rem; color: #cbd5e1;'>A rumor starts in a high school network. Classic models track it crawling friend-to-friend sequentially. A <b>Quantum Walk</b> model propagates information as a probability wave, simulating how rumor waves meet and amplify or cancel out in phase.</p>
            </div>
            <div style='background: rgba(56, 189, 248, 0.1); padding: 8px; border-radius: 4px; font-size: 0.75rem; text-align: center; color: #38bdf8; font-weight: bold;'>
                Mapped Concept: Quantum Walks
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="glass-card" style='border-top: 4px solid #ec4899; height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>
            <div>
                <h4 style='color: #ec4899; margin-top: 0;'>Scenario 2: The Coordinated Trend</h4>
                <p style='font-size: 0.85rem; color: #cbd5e1;'>Two friends post identically timed photos across the globe. No physical message was sent, but they act in perfect coordination. This models <b>Quantum Entanglement</b>, where two nodes are coupled such that measuring one instantly dictates the other's state.</p>
            </div>
            <div style='background: rgba(236, 72, 153, 0.1); padding: 8px; border-radius: 4px; font-size: 0.75rem; text-align: center; color: #ec4899; font-weight: bold;'>
                Mapped Concept: Entanglement
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="glass-card" style='border-top: 4px solid #10b981; height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>
            <div>
                <h4 style='color: #10b981; margin-top: 0;'>Scenario 3: Influencer Discovery</h4>
                <p style='font-size: 0.85rem; color: #cbd5e1;'>Finding a hidden broker among 1,000 users. A classical engine checks database rows one-by-one. <b>Grover's Search</b> runs a possibility superposition over all entries, reflecting amplitudes toward the target to find them in a fraction of the time.</p>
            </div>
            <div style='background: rgba(16, 185, 129, 0.1); padding: 8px; border-radius: 4px; font-size: 0.75rem; text-align: center; color: #10b981; font-weight: bold;'>
                Mapped Concept: Grover's Search
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_interactive_visualizer():
    st.subheader("🔲 Graph-to-Matrix Coordinate Mapper")
    st.write("Hover over matrix cells or nodes to see the mapping of graph edges to adjacency matrix entries.")
    
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body {
        background: transparent;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
    }
    .viz-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
    }
    .panel {
        flex: 1;
        min-width: 250px;
        text-align: center;
    }
    .matrix-grid {
        display: grid;
        grid-template-columns: repeat(6, 32px);
        gap: 4px;
        justify-content: center;
        margin: 15px auto;
    }
    .cell {
        width: 32px;
        height: 32px;
        line-height: 32px;
        text-align: center;
        font-size: 0.75rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .cell.header {
        background: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        font-weight: bold;
        border-color: rgba(56, 189, 248, 0.2);
        cursor: default;
    }
    .cell.edge-active {
        background: rgba(16, 185, 129, 0.4);
        border-color: #10b981;
        color: white;
        font-weight: bold;
    }
    .cell.self-loop {
        background: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
    .cell.hovered {
        background: #8b5cf6 !important;
        border-color: #a855f7 !important;
        color: white !important;
        transform: scale(1.1);
        box-shadow: 0 0 8px rgba(139, 92, 246, 0.6);
    }
    svg {
        background: transparent;
        overflow: visible;
    }
    .node {
        fill: #3b82f6;
        stroke: #fff;
        stroke-width: 2px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .node:hover, .node.active {
        fill: #a855f7;
        stroke: #38bdf8;
        stroke-width: 3px;
        filter: drop-shadow(0 0 6px rgba(56, 189, 248, 0.8));
    }
    .edge {
        stroke: rgba(255, 255, 255, 0.2);
        stroke-width: 2px;
        transition: all 0.2s ease;
    }
    .edge.active {
        stroke: #10b981;
        stroke-width: 4px;
        filter: drop-shadow(0 0 4px rgba(16, 185, 129, 0.8));
    }
    .node-text {
        fill: #cbd5e1;
        font-size: 11px;
        font-weight: bold;
        pointer-events: none;
        text-anchor: middle;
        dominant-baseline: middle;
    }
    .info-bar {
        margin-top: 15px;
        font-size: 0.85rem;
        color: #94a3b8;
        min-height: 20px;
    }
    </style>
    </head>
    <body>
    <div class="viz-container">
        <div class="panel">
            <h4 style="margin: 0 0 10px 0; color: #38bdf8;">Graph View</h4>
            <svg width="220" height="220" id="graph-svg">
                <line id="e01" class="edge" x1="110" y1="30" x2="190" y2="90" />
                <line id="e12" class="edge" x1="190" y2="90" x2="160" y2="180" />
                <line id="e23" class="edge" x1="160" y2="180" x2="60" y2="180" />
                <line id="e34" class="edge" x1="60" y2="180" x2="30" y2="90" />
                <line id="e40" class="edge" x1="30" y2="90" x2="110" y2="30" />
                
                <circle id="n0" class="node" cx="110" cy="30" r="16" />
                <circle id="n1" class="node" cx="190" cy="90" r="16" />
                <circle id="n2" class="node" cx="160" cy="180" r="16" />
                <circle id="n3" class="node" cx="60" cy="180" r="16" />
                <circle id="n4" class="node" cx="30" cy="90" r="16" />
                
                <text x="110" y="30" class="node-text">P0</text>
                <text x="190" y="90" class="node-text">P1</text>
                <text x="160" y="180" class="node-text">P2</text>
                <text x="60" y="180" class="node-text">P3</text>
                <text x="30" y="90" class="node-text">P4</text>
            </svg>
        </div>
        <div class="panel">
            <h4 style="margin: 0 0 10px 0; color: #a855f7;">Adjacency Matrix</h4>
            <div class="matrix-grid" id="matrix">
                <div class="cell header">/</div>
                <div class="cell header">P0</div>
                <div class="cell header">P1</div>
                <div class="cell header">P2</div>
                <div class="cell header">P3</div>
                <div class="cell header">P4</div>
                
                <div class="cell header">P0</div>
                <div id="c00" class="cell self-loop" data-r="0" data-c="0">0</div>
                <div id="c01" class="cell edge-active" data-r="0" data-c="1">1</div>
                <div id="c02" class="cell" data-r="0" data-c="2">0</div>
                <div id="c03" class="cell" data-r="0" data-c="3">0</div>
                <div id="c04" class="cell edge-active" data-r="0" data-c="4">1</div>
                
                <div class="cell header">P1</div>
                <div id="c10" class="cell edge-active" data-r="1" data-c="0">1</div>
                <div id="c11" class="cell self-loop" data-r="1" data-c="1">0</div>
                <div id="c12" class="cell edge-active" data-r="1" data-c="2">1</div>
                <div id="c13" class="cell" data-r="1" data-c="3">0</div>
                <div id="c14" class="cell" data-r="1" data-c="4">0</div>
                
                <div class="cell header">P2</div>
                <div id="c20" class="cell" data-r="2" data-c="0">0</div>
                <div id="c21" class="cell edge-active" data-r="2" data-c="1">1</div>
                <div id="c22" class="cell self-loop" data-r="2" data-c="2">0</div>
                <div id="c23" class="cell edge-active" data-r="2" data-c="3">1</div>
                <div id="c24" class="cell" data-r="2" data-c="4">0</div>
                
                <div class="cell header">P3</div>
                <div id="c30" class="cell" data-r="3" data-c="0">0</div>
                <div id="c31" class="cell" data-r="3" data-c="1">0</div>
                <div id="c32" class="cell edge-active" data-r="3" data-c="2">1</div>
                <div id="c33" class="cell self-loop" data-r="3" data-c="3">0</div>
                <div id="c34" class="cell edge-active" data-r="3" data-c="4">1</div>
                
                <div class="cell header">P4</div>
                <div id="c40" class="cell edge-active" data-r="4" data-c="0">1</div>
                <div id="c41" class="cell" data-r="4" data-c="1">0</div>
                <div id="c42" class="cell" data-r="4" data-c="2">0</div>
                <div id="c43" class="cell edge-active" data-r="4" data-c="3">1</div>
                <div id="c44" class="cell self-loop" data-r="4" data-c="4">0</div>
            </div>
        </div>
    </div>
    <div id="info" class="info-bar">Hover over a cell or node to explore coordinates.</div>
    
    <script>
    const cells = document.querySelectorAll('.cell[data-r]');
    const nodes = document.querySelectorAll('.node');
    const info = document.getElementById('info');
    
    cells.forEach(cell => {
        cell.addEventListener('mouseenter', () => {
            const r = cell.getAttribute('data-r');
            const c = cell.getAttribute('data-c');
            cell.classList.add('hovered');
            
            document.getElementById('n' + r).classList.add('active');
            document.getElementById('n' + c).classList.add('active');
            
            const edgeId = getEdgeId(r, c);
            if (edgeId) {
                const edge = document.getElementById(edgeId);
                if (edge) edge.classList.add('active');
                info.innerHTML = `Matrix element A[${r}][${c}] = 1 → Relationship exists between Person ${r} and Person ${c}.`;
            } else if (r === c) {
                info.innerHTML = `Matrix diagonal A[${r}][${c}] = 0 → Self-loops are disabled.`;
            } else {
                info.innerHTML = `Matrix element A[${r}][${c}] = 0 → No direct relationship between Person ${r} and Person ${c}.`;
            }
        });
        
        cell.addEventListener('mouseleave', () => {
            const r = cell.getAttribute('data-r');
            const c = cell.getAttribute('data-c');
            cell.classList.remove('hovered');
            document.getElementById('n' + r).classList.remove('active');
            document.getElementById('n' + c).classList.remove('active');
            const edgeId = getEdgeId(r, c);
            if (edgeId) {
                const edge = document.getElementById(edgeId);
                if (edge) edge.classList.remove('active');
            }
            info.innerHTML = "Hover over a cell or node to explore coordinates.";
        });
    });
    
    nodes.forEach(node => {
        node.addEventListener('mouseenter', () => {
            const id = node.getAttribute('id').replace('n', '');
            node.classList.add('active');
            
            cells.forEach(cell => {
                if (cell.getAttribute('data-r') === id || cell.getAttribute('data-c') === id) {
                    cell.style.background = 'rgba(139, 92, 246, 0.2)';
                }
            });
            info.innerHTML = `Person ${id}: Connected to neighbors via Row ${id} and Column ${id} in the matrix.`;
        });
        node.addEventListener('mouseleave', () => {
            const id = node.getAttribute('id').replace('n', '');
            node.classList.remove('active');
            cells.forEach(cell => {
                cell.style.background = '';
            });
            info.innerHTML = "Hover over a cell or node to explore coordinates.";
        });
    });
    
    function getEdgeId(r, c) {
        const u = Math.min(r, c);
        const v = Math.max(r, c);
        if ((u === 0 && v === 1) || (u === 1 && v === 2) || (u === 2 && v === 3) || (u === 3 && v === 4) || (u === 0 && v === 4)) {
            return `e${u}${v}`;
        }
        return null;
    }
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=360)

def show_bell_state_sandbox():
    st.subheader("💡 2-Qubit Bell State Builder Sandbox")
    st.write("Assemble a simple quantum circuit to entangle two nodes. See how gate options shift state vectors and create tight couplings.")
    
    col_gates, col_res = st.columns([1.5, 2.5])
    
    with col_gates:
        st.markdown("**Circuit Controls**")
        gate_q0 = st.selectbox("Gate on Node A (q0):", ["None", "H", "X"], key="bell_q0")
        gate_q1 = st.selectbox("Gate on Node B (q1):", ["None", "H", "X"], key="bell_q1")
        gate_cnot = st.selectbox("CNOT Connection:", ["None", "CNOT (q0 -> q1)", "CNOT (q1 -> q0)"], key="bell_cnot")
        
        st.markdown("**Circuit Diagram:**")
        diagram = []
        g0_str = f"[{gate_q0}]" if gate_q0 != "None" else "───"
        g1_str = f"[{gate_q1}]" if gate_q1 != "None" else "───"
        
        if gate_cnot == "CNOT (q0 -> q1)":
            diagram.append(f"q0: ──{g0_str}───●───")
            diagram.append(f"q1: ──{g1_str}───X───")
        elif gate_cnot == "CNOT (q1 -> q0)":
            diagram.append(f"q0: ──{g0_str}───X───")
            diagram.append(f"q1: ──{g1_str}───●───")
        else:
            diagram.append(f"q0: ──{g0_str}───────")
            diagram.append(f"q1: ──{g1_str}───────")
            
        st.code("\n".join(diagram), language="text")
        
    with col_res:
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        
        u0 = H if gate_q0 == "H" else X if gate_q0 == "X" else I
        u1 = H if gate_q1 == "H" else X if gate_q1 == "X" else I
        
        u_single = np.kron(u0, u1)
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
        psi1 = u_single @ psi0
        
        if gate_cnot == "CNOT (q0 -> q1)":
            cnot01 = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=complex)
            psit = cnot01 @ psi1
        elif gate_cnot == "CNOT (q1 -> q0)":
            cnot10 = np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]
            ], dtype=complex)
            psit = cnot10 @ psi1
        else:
            psit = psi1
            
        probs = np.abs(psit)**2
        
        labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
        parts = []
        for i, val in enumerate(psit):
            if np.abs(val) > 0.01:
                real = val.real
                imag = val.imag
                if np.abs(imag) < 0.001:
                    parts.append(f"{real:.3f}{labels[i]}")
                elif np.abs(real) < 0.001:
                    parts.append(f"{imag:.3f}i{labels[i]}")
                else:
                    parts.append(f"({real:.3f} + {imag:.3f}i){labels[i]}")
                    
        stvector_str = " + ".join(parts) if parts else "0"
        
        st.subheader("Simulated Output")
        st.markdown(f"**Statevector:** `{stvector_str}`")
        
        fig = px.bar(x=labels, y=probs, labels={'x': 'State', 'y': 'Probability'}, range_y=[0.0, 1.05])
        fig.update_layout(
            height=160,
            margin=dict(l=10, r=10, b=10, t=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        det = psit[0] * psit[3] - psit[1] * psit[2]
        if np.abs(det) > 1e-4:
            st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 6px; padding: 10px; font-size: 0.85rem; color: #10b981; font-weight: bold;'>
                🔗 Quantum Entanglement Detected!
            </div>
            <span style='font-size: 0.8rem; color: #94a3b8;'>Measuring Node A collapses Node B immediately. This state cannot be factored into two independent qubits!</span>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(100, 116, 139, 0.1); border: 1px solid #64748b; border-radius: 6px; padding: 10px; font-size: 0.85rem; color: #cbd5e1;'>
                ⚪ Separate Qubits (Unentangled)
            </div>
            <span style='font-size: 0.8rem; color: #94a3b8;'>These nodes are independent. Measuring Node A gives no information about Node B's state.</span>
            """, unsafe_allow_html=True)

def show_resource_estimator(node_count):
    st.subheader("🔬 Quantum Hardware Resource Estimator")
    st.write(f"Calculating hardware requirements for running calculations on a network of size **{node_count} nodes**:")
    
    n_sq = node_count ** 2
    amp_qubits = int(np.ceil(np.log2(n_sq))) if n_sq > 0 else 0
    basis_qubits = node_count
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="glass-card" style='border-left: 4px solid #38bdf8;'>
            <h5 style='color: #38bdf8; margin: 0;'>Amplitude Encoding (Compressed)</h5>
            <div style='font-size: 2rem; font-weight: 800; margin: 10px 0;'>{amp_qubits} Qubits</div>
            <span style='font-size: 0.8rem; color: #cbd5e1;'>Requires log₂(N²) qubits. Best for large, dense adjacency matrix representations.</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="glass-card" style='border-left: 4px solid #a855f7;'>
            <h5 style='color: #a855f7; margin: 0;'>Basis / One-Hot Encoding</h5>
            <div style='font-size: 2rem; font-weight: 800; margin: 10px 0;'>{basis_qubits} Qubits</div>
            <span style='font-size: 0.8rem; color: #cbd5e1;'>Requires N qubits (one per person). Best for walks and simple circuit structures.</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
    **Estimated Execution Specifications:**
    - **Logical gate count (Quantum Walk step):** ~{int(node_count * 12)} CNOT gates
    - **Coherence time requirement:** ~{(node_count * 1.5):.1f} μs
    - **Recommended Quantum Computer backend:** superconducting or trapped-ion (> {basis_qubits} qubits)
    """)

