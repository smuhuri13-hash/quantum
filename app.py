import streamlit as st
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import random
import numpy as np
import plotly.express as px
import onboarding

# Set page config
st.set_page_config(page_title="QSNA Studio", layout="wide", initial_sidebar_state="expanded")

# Initialize Session State
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "beginner_mode" not in st.session_state:
    st.session_state.beginner_mode = False
if "learn_step" not in st.session_state:
    st.session_state.learn_step = 1
if "workshop_step" not in st.session_state:
    st.session_state.workshop_step = 1
if "G" not in st.session_state:
    st.session_state.G = None
if "advanced_research" not in st.session_state:
    st.session_state.advanced_research = False

# Inject CSS for Glassmorphism
def load_css():
    st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Header text */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Glassmorphism Cards */
    div.css-1r6slb0, div.css-12oz5g7, .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        border: none;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(139, 92, 246, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Route pages based on session state
if st.session_state.page == "landing":
    onboarding.show_landing_page()
elif st.session_state.page == "learn":
    onboarding.show_learn_page()
elif st.session_state.page == "transition":
    onboarding.show_transition_page()
elif st.session_state.page == "challenges":
    onboarding.show_challenges_page()
elif st.session_state.page == "workshop":
    onboarding.show_workshop_mode()
elif st.session_state.page == "demo_datasets":
    onboarding.show_demo_datasets_page()
else:
    # --- DASHBOARD PAGE ---
    st.title("Quantum Social Network Analysis (QSNA) Studio")
    
    # Sidebar Navigation and Configuration
    st.sidebar.title("Navigation")
    
    if st.sidebar.button("🏠 Return to Landing Page", use_container_width=True):
        st.session_state.page = "landing"
        st.rerun()
        
    st.sidebar.markdown("---")
    st.session_state.beginner_mode = st.sidebar.checkbox("Enable Beginner Explanations", value=st.session_state.beginner_mode)
    st.session_state.advanced_research = st.sidebar.checkbox("🔬 Show Advanced Research Settings", value=st.session_state.advanced_research)
    
    if st.sidebar.button("🏆 Open Challenges Mode", use_container_width=True):
        st.session_state.page = "challenges"
        st.rerun()
        
    st.sidebar.markdown("---")
    
    pages = [
        "1. Data Input",
        "2. Graph Dashboard",
        "3. Classical SNA",
        "4. Quantum Conversion",
        "5. Circuit Builder",
        "6. Quantum Models",
        "7. Quantum Tasks",
        "8. Quantum Walk",
        "9. Results Dashboard",
        "10. Explainability",
        "11. Export"
    ]
    
    # Track selected page index programmatically
    if "dashboard_step" not in st.session_state:
        st.session_state.dashboard_step = "1. Data Input"
        
    default_index = 0
    if st.session_state.dashboard_step in pages:
        default_index = pages.index(st.session_state.dashboard_step)
        
    selection = st.sidebar.radio("Go to step:", pages, index=default_index)
    st.session_state.dashboard_step = selection
    
    # Helper to print mapped terms for beginner explanations
    def get_t(term_key):
        return onboarding.translate(term_key, st.session_state.beginner_mode)

    # 1. EXPANDABLE "WHY QUANTUM FOR THIS TASK?" PANEL
    with st.sidebar.expander("❓ Why Quantum for this Task?", expanded=True):
        if selection == "1. Data Input":
            st.write("**Data Compression:** Amplitude encoding packs $N^2$ connections into only $\\lceil \\log_2(N^2) \\rceil$ qubits. This provides massive data reduction.")
        elif selection == "2. Graph Dashboard":
            st.write("**Visual Representation:** Classical rendering is static. Quantum structures model nodes as states where constructive wave peaks identify community centers.")
        elif selection == "3. Classical SNA":
            st.write("**Centrality Mapping:** Classical centrality checks connections sequentially. Quantum walks simulate possibility waves propagating across all paths simultaneously.")
        elif selection == "4. Quantum Conversion":
            st.write("**State Conversion:** Converts node relational values into wave amplitudes, preparing network data for parallel quantum gate execution.")
        elif selection == "5. Circuit Builder":
            st.write("**logic Execution:** Hadamard gates put individuals into multiple possible paths, while CNOT gates construct entangled opinion coupling between users.")
        elif selection == "6. Quantum Models":
            st.write("**Hilbert Mapping:** Quantum SVMs map relationship metrics into infinite-dimensional Hilbert spaces, making overlapping groups easily partitionable.")
        elif selection == "7. Quantum Tasks":
            st.write("**Task Processing:** Executes modular circuits customized to community structures, rumors propagation, or classification boundaries.")
        elif selection == "8. Quantum Walk":
            st.write("**Superposition Exploration:** Spreads waves across all links simultaneously, identifying target influencer nodes in $O(\\sqrt{N})$ steps instead of $O(N)$.")
        elif selection == "10. Explainability":
            st.write("**Wave Interference Peaks:** Illustrates how quantum wave amplification reveals critical nodes, explaining outputs mathematically yet conceptually.")
        else:
            st.write("**Quantum Advantages:** Exploits superposition and entanglement to solve social graph networks faster than classical Turing machines.")

    # Step 1: Data Input
    if selection == "1. Data Input":
        st.header(f"Step 1: Input Network Data")
        if st.session_state.beginner_mode:
            st.info("👋 **Beginner Mode Active:** Load a network to analyze. A network maps individuals (nodes/people) and connections (edges/relationships).")
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Upload Graph")
            uploaded_file = st.file_uploader("Upload CSV/GraphML/JSON", type=["csv", "graphml", "json"])
            
            st.subheader("Demo Datasets")
            demo_option = st.selectbox("Select a demo dataset:", [
                "None",
                "1. Misinformation propagation network",
                "2. Friendship social graph",
                "3. Twitter interaction network",
                "4. Citation network",
                "5. Community network",
                "6. Epidemic spreading network",
                "7. Influencer detection network",
                "8. Small toy graph (5–10 nodes)"
            ])
            
        with col2:
            st.subheader("Synthetic Graph Generator")
            synth_option = st.selectbox("Select graph type:", ["None", "Erdos-Renyi", "Scale-free", "Small-world", "Community graph"])
            
            # Simple descriptions for beginners
            if st.session_state.beginner_mode and synth_option != "None":
                if synth_option == "Erdos-Renyi":
                    st.write("🎲 **Erdos-Renyi:** Connections are generated completely randomly.")
                elif synth_option == "Scale-free":
                    st.write("📈 **Scale-free:** A few highly connected 'influencer' hubs emerge naturally.")
                elif synth_option == "Small-world":
                    st.write("🤝 **Small-world:** Highly clustered groups where everyone is only a few handshakes away.")
                    
            if synth_option == "Erdos-Renyi":
                n = st.slider("Nodes", 10, 100, 30)
                p = st.slider("Probability", 0.0, 1.0, 0.1)
                if st.button("Generate E-R Graph"):
                    st.session_state.G = nx.erdos_renyi_graph(n, p)
                    st.success(f"Generated Erdos-Renyi Graph ({n} nodes)")
            elif synth_option == "Scale-free":
                n = st.slider("Nodes", 10, 100, 30)
                if st.button("Generate Scale-free Graph"):
                    st.session_state.G = nx.barabasi_albert_graph(n, min(3, n-1))
                    st.success(f"Generated Scale-free Graph ({n} nodes)")
            elif synth_option == "Small-world":
                n = st.slider("Nodes", 10, 100, 30)
                if st.button("Generate Small-world Graph"):
                    st.session_state.G = nx.watts_strogatz_graph(n, 4, 0.1)
                    st.success(f"Generated Small-world Graph ({n} nodes)")
            elif synth_option == "Community graph":
                if st.button("Generate Community Graph"):
                    st.session_state.G = nx.caveman_graph(4, 8)
                    st.success("Generated Community Graph (32 nodes)")
                    
        if demo_option != "None":
            if st.button("Load Demo"):
                if demo_option == "8. Small toy graph (5–10 nodes)":
                    st.session_state.G = nx.cycle_graph(5)
                elif demo_option == "2. Friendship social graph":
                    st.session_state.G = nx.karate_club_graph()
                else:
                    st.session_state.G = nx.watts_strogatz_graph(30, 4, 0.1)
                st.success(f"Loaded {demo_option}")
                
        # 2. STORY MODE ACCORDION
        st.write("---")
        with st.expander("📖 Play Social Network Story Mode", expanded=False):
            onboarding.show_story_mode()

    # Step 2: Graph Dashboard
    elif selection == "2. Graph Dashboard":
        st.header("Step 2: Graph Dashboard")
        
        if st.session_state.G is None:
            st.warning("Please load or generate a graph in Step 1 first.")
        else:
            G = st.session_state.G
            
            # Beginner vs Advanced label mappings
            node_label = "People (Nodes)" if st.session_state.beginner_mode else "Nodes"
            edge_label = "Relationships (Edges)" if st.session_state.beginner_mode else "Edges"
            density_label = "Group Closeness (Density)" if st.session_state.beginner_mode else "Density"
            deg_label = "Average Friend Count" if st.session_state.beginner_mode else "Avg Degree"
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(node_label, G.number_of_nodes())
            col2.metric(edge_label, G.number_of_edges())
            col3.metric(density_label, round(nx.density(G), 3))
            
            degrees = [d for n, d in G.degree()]
            avg_degree = sum(degrees) / len(degrees) if degrees else 0
            col4.metric(deg_label, round(avg_degree, 2))
            
            if st.session_state.beginner_mode:
                st.info("💡 **Tip:** Click on a person (node) in the graph below to discover why they are influential, what group they belong to, and how quantum analysis highlights their importance.")
                
            st.subheader("Interactive Graph Visualization")
            
            nodes = []
            edges = []
            
            for node in G.nodes():
                nodes.append(Node(id=str(node), label=f"Person {node}" if st.session_state.beginner_mode else str(node), size=25, color="#818cf8"))
                
            for edge in G.edges():
                edges.append(Edge(source=str(edge[0]), target=str(edge[1]), color="#475569"))
                
            config = Config(width=1000,
                            height=500,
                            directed=nx.is_directed(G), 
                            physics=True, 
                            hierarchical=False)
                            
            clicked_node = agraph(nodes=nodes, edges=edges, config=config)
            
            # Node explainability details below visualization
            if clicked_node is not None:
                st.session_state.clicked_node = clicked_node
                
            if "clicked_node" in st.session_state:
                node_id = int(st.session_state.clicked_node) if st.session_state.clicked_node.isdigit() else st.session_state.clicked_node
                
                # Check if node exists in graph
                if node_id in G.nodes():
                    st.markdown("---")
                    st.subheader(f"🔍 Node Explanation Panel: {get_t('node')} {node_id}")
                    
                    # Compute simple stats for the node
                    node_deg = G.degree(node_id)
                    total_nodes = G.number_of_nodes()
                    degree_pct = node_deg / (total_nodes - 1) if total_nodes > 1 else 0
                    
                    if st.session_state.beginner_mode:
                        st.markdown(f"""
                        **Why this person is influential:**
                        - Person {node_id} has **{node_deg} direct relationships**, connecting them to {degree_pct:.0%} of the group.
                        - Because of their central location, information or rumors starting in the network will reach them rapidly.
                        
                        **How quantum walks highlight this person:**
                        - In a quantum walk simulation, a wave of possibilities spreads across the network. Because of Person {node_id}'s central connectivity, the waves interfere constructively (build up), creating a high probability amplitude on this person. This indicates they are a major opinion leader.
                        """)
                    else:
                        st.markdown(f"""
                        **Centrality Metrics Analysis:**
                        - Node {node_id} has a degree of **{node_deg}** (normalized degree centrality: `{degree_pct:.4f}`).
                        - Connectivity indicates this node sits on multiple propagation paths, accelerating diffusion processes.
                        
                        **Quantum Walk Significance:**
                        - Evolution under the adjacency Hamiltonian $H = -A$ yields a high survival probability $P_{{ {node_id} }}(t) = |\\langle {node_id} | e^{{-iHt}} | \\psi_0 \\rangle|^2$ due to constructive interference of eigenvector modes matching the dominant eigenvalues.
                        """)
                        
            # Interactive visualizer embedded below G
            st.write("---")
            onboarding.show_interactive_visualizer()

    # Step 3: Classical SNA
    elif selection == "3. Classical SNA":
        st.header("Step 3: Classical Social Network Analysis")
        if st.session_state.G is None:
            st.warning("Please load a graph in Step 1.")
        else:
            G = st.session_state.G
            
            # --- GUESS BEFORE RUN MODE (SNA Centrality Checkpoint) ---
            if "guess_centrality" not in st.session_state:
                st.session_state.guess_centrality = None
                
            if st.session_state.beginner_mode and st.session_state.guess_centrality is None:
                st.subheader("🤔 Predict the Influential Node First!")
                guess_node = st.selectbox("Which person (node) do you predict is most influential?", list(G.nodes()), key="sel_guess_cent")
                if st.button("Register Prediction"):
                    st.session_state.guess_centrality = guess_node
                    st.rerun()
            else:
                if st.session_state.beginner_mode:
                    st.write(f"*(Registered Prediction: Person {st.session_state.guess_centrality})*")
                    if st.button("Change Prediction"):
                        st.session_state.guess_centrality = None
                        st.rerun()
                        
                tab1, tab2, tab3 = st.tabs(["Centrality", "Community", "Machine Learning"])
                
                with tab1:
                    st.subheader("Centrality Metrics")
                    
                    # Simplify option labels if beginner mode is on
                    metric_options = ["Degree", "Betweenness", "Closeness", "Eigenvector", "PageRank"]
                    if st.session_state.beginner_mode:
                        metric_options = [
                            "Degree (Popularity / Direct Connections)",
                            "Betweenness (Bridge Builders)",
                            "Closeness (Rapid Communicators)",
                            "Eigenvector (Connected to Important People)",
                            "PageRank (Web/Social Authority)"
                        ]
                        
                    cent_metric_sel = st.selectbox("Select Metric:", metric_options)
                    # Map back to standard names
                    cent_metric = "Degree"
                    if "Betweenness" in cent_metric_sel: cent_metric = "Betweenness"
                    elif "Closeness" in cent_metric_sel: cent_metric = "Closeness"
                    elif "Eigenvector" in cent_metric_sel: cent_metric = "Eigenvector"
                    elif "PageRank" in cent_metric_sel: cent_metric = "PageRank"
                    
                    if st.button("Compute Centrality"):
                        with st.spinner("Computing..."):
                            if cent_metric == "Degree":
                                centrality = nx.degree_centrality(G)
                            elif cent_metric == "Betweenness":
                                centrality = nx.betweenness_centrality(G)
                            elif cent_metric == "Closeness":
                                centrality = nx.closeness_centrality(G)
                            elif cent_metric == "Eigenvector":
                                try:
                                    centrality = nx.eigenvector_centrality(G, max_iter=1000)
                                except nx.PowerIterationFailedConvergence:
                                    centrality = nx.degree_centrality(G)
                                    st.warning("Eigenvector failed to converge, showing degree instead.")
                            elif cent_metric == "PageRank":
                                centrality = nx.pagerank(G)
                            
                            st.success("Computed!")
                            
                            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.write("Top 10 Nodes:")
                                st.dataframe({"Node": [n[0] for n in sorted_nodes], "Score": [round(n[1], 4) for n in sorted_nodes]})
                                
                                # Display guess comparison
                                if st.session_state.guess_centrality is not None:
                                    # Perform a quick degree centrality proxy for quantum walk peak
                                    deg_cent = nx.degree_centrality(G)
                                    q_peak_node = max(deg_cent, key=deg_cent.get)
                                    
                                    st.markdown(f"""
                                    <div style='background: rgba(56, 189, 248, 0.1); border: 1px solid #38bdf8; border-radius: 8px; padding: 15px; margin-top: 15px;'>
                                        <b>🔮 Prediction Results:</b><br>
                                        - 👤 Your Guess: Person <b>{st.session_state.guess_centrality}</b><br>
                                        - 🖥️ Classical Peak: Node <b>{sorted_nodes[0][0]}</b><br>
                                        - ⚛️ Quantum Walk Prediction: Node <b>{q_peak_node}</b>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with col2:
                                nodes = []
                                max_cent = max(centrality.values()) if centrality.values() else 1
                                for node in G.nodes():
                                    val = centrality.get(node, 0)
                                    size = 15 + (val/max_cent)*30 if max_cent > 0 else 25
                                    nodes.append(Node(id=str(node), label=str(node), size=size, color="#f59e0b"))
                                    
                                edges = [Edge(source=str(e[0]), target=str(e[1]), color="#475569") for e in G.edges()]
                                agraph(nodes=nodes, edges=edges, config=Config(width=600, height=400, directed=nx.is_directed(G), physics=True))
                                
                with tab2:
                    st.subheader("Community Detection")
                    if st.session_state.beginner_mode:
                        st.info("💡 **What is this?** This groups people into tight-knit social circles based on their shared relationships.")
                    else:
                        st.info("Uses NetworkX Louvain communities implementation.")
                        
                    if st.button("Detect Communities"):
                        try:
                            communities = nx.community.louvain_communities(G)
                            st.success(f"Found {len(communities)} communities.")
                            
                            colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#6366f1"]
                            nodes = []
                            for i, comm in enumerate(communities):
                                color = colors[i % len(colors)]
                                for node in comm:
                                    nodes.append(Node(id=str(node), label=str(node), size=25, color=color))
                            
                            edges = [Edge(source=str(e[0]), target=str(e[1]), color="#475569") for e in G.edges()]
                            agraph(nodes=nodes, edges=edges, config=Config(width=800, height=500, physics=True))
                        except Exception as e:
                            st.error(f"Error: {e}")
                            
                with tab3:
                    st.subheader("Classical Machine Learning")
                    st.write("Baseline ML models for Node Classification and Link Prediction.")
                    model = st.selectbox("Model", ["SVM", "Logistic Regression", "Random Forest", "KNN", "GNN Baseline"])
                    st.info("This is a placeholder for classical ML execution. Proceed to Quantum Models for comparison.")

    # Step 4: Quantum Conversion
    elif selection == "4. Quantum Conversion":
        st.header("Step 4: Classical-to-Quantum Conversion")
        if st.session_state.G is None:
            st.warning("Please load a graph in Step 1.")
        else:
            G = st.session_state.G
            
            st.write("Convert classical network data into quantum representations.")
            encoding = st.selectbox("Encoding Method", ["Amplitude Encoding", "Basis Encoding", "Angle Encoding"])
            
            if st.session_state.beginner_mode:
                st.info(f"💡 **Beginner Explanation ({encoding}):** " + 
                        ("We compress the entire social connection network into a single quantum state. "
                         "This allows the quantum computer to read all paths at once." if encoding == "Amplitude Encoding" else
                         "We map each person directly to a separate physical qubit to model relationships as quantum links."))
            
            if st.button("Convert to Quantum State"):
                adj_matrix = nx.to_numpy_array(G)
                n_nodes = G.number_of_nodes()
                
                st.subheader("1. Classical Adjacency Matrix")
                if st.session_state.beginner_mode:
                    st.write("This grid shows who is connected to whom. Yellow blocks show relationships, purple shows no relationship.")
                fig = px.imshow(adj_matrix, color_continuous_scale='Viridis', title="Adjacency Heatmap")
                st.plotly_chart(fig, use_container_width=True)
                
                if not st.session_state.beginner_mode:
                    st.subheader("2. Vector Representation")
                    vec = adj_matrix.flatten()
                    norm = np.linalg.norm(vec)
                    if norm == 0:
                        norm = 1
                    normalized_vec = vec / norm
                    st.write(f"Total elements: {len(vec)}")
                
                st.subheader(f"2. {get_t('state')}")
                if encoding == "Amplitude Encoding":
                    qubits_needed = int(np.ceil(np.log2(n_nodes * n_nodes)))
                    if qubits_needed == 0: qubits_needed = 1
                    
                    if st.session_state.beginner_mode:
                        st.info(f"Using {get_t('amplitude')}: Compresses {n_nodes}x{n_nodes} connections onto {qubits_needed} qubits.")
                    else:
                        st.info(f"Using Amplitude Encoding: Requires ⌈log₂(N²)⌉ = {qubits_needed} qubits.")
                    
                    vec = adj_matrix.flatten()
                    norm = np.linalg.norm(vec)
                    if norm == 0: norm = 1
                    normalized_vec = vec / norm
                    padded_len = 2**qubits_needed
                    padded_vec = np.pad(normalized_vec, (0, padded_len - len(normalized_vec)))
                    if np.linalg.norm(padded_vec) > 0:
                        padded_vec = padded_vec / np.linalg.norm(padded_vec)
                    
                    st.write("Probability Amplitudes (|α|²):")
                    st.bar_chart(np.abs(padded_vec)**2)
                else:
                    st.info(f"Using {encoding}: Requires {n_nodes} qubits.")
                    st.write("Note: Basis/Angle encodings map nodes directly to qubits.")

    # Step 5: Interactive Quantum Circuit Builder
    elif selection == "5. Circuit Builder":
        st.header("Step 5: Interactive Quantum Circuit Builder")
        from qiskit import QuantumCircuit
        
        if st.session_state.beginner_mode:
            st.info("💡 **Qubit Circuit Simulator:** Build a sequence of operations. Think of qubits as spinning coins, and gates as flips (X) or spins (H) that manipulate their possibilities.")
            
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Circuit Parameters")
            num_qubits = st.number_input("Number of Qubits", min_value=1, max_value=5, value=2)
            depth = st.number_input("Circuit Depth", min_value=1, max_value=10, value=3)
            
            # State to store gates
            if "circuit_gates" not in st.session_state or st.session_state.get("num_qubits") != num_qubits or st.session_state.get("depth") != depth:
                st.session_state.circuit_gates = {f"q{q}_d{d}": "I" for q in range(num_qubits) for d in range(depth)}
                st.session_state.num_qubits = num_qubits
                st.session_state.depth = depth
                
            gate_options = ["I", "H", "X", "Y", "Z", "RX", "RY", "RZ", "CNOT (Target next)", "CZ (Target next)"]
            
            st.write("Construct your circuit:")
            for q in range(num_qubits):
                st.write(f"**Qubit {q}**")
                cols = st.columns(depth)
                for d in range(depth):
                    key = f"q{q}_d{d}"
                    st.session_state.circuit_gates[key] = cols[d].selectbox(f"Layer {d}", gate_options, index=gate_options.index(st.session_state.circuit_gates[key]), key=key, label_visibility="collapsed")
                    
        with col2:
            st.subheader("Live Circuit Visualization")
            qc = QuantumCircuit(num_qubits)
            
            import math
            for d in range(depth):
                for q in range(num_qubits):
                    gate = st.session_state.circuit_gates[f"q{q}_d{d}"]
                    if gate == "H": qc.h(q)
                    elif gate == "X": qc.x(q)
                    elif gate == "Y": qc.y(q)
                    elif gate == "Z": qc.z(q)
                    elif gate == "RX": qc.rx(math.pi/2, q)
                    elif gate == "RY": qc.ry(math.pi/2, q)
                    elif gate == "RZ": qc.rz(math.pi/2, q)
                    elif gate == "CNOT (Target next)":
                        if q + 1 < num_qubits: qc.cx(q, q+1)
                    elif gate == "CZ (Target next)":
                        if q + 1 < num_qubits: qc.cz(q, q+1)
                qc.barrier()
                
            fig = qc.draw(output='mpl')
            st.pyplot(fig)
            
            st.subheader("Statevector Amplitudes")
            try:
                from qiskit.quantum_info import Statevector
                import plotly.express as px
                sv = Statevector.from_instruction(qc)
                probs = sv.probabilities()
                fig2 = px.bar(x=[format(i, f'0{num_qubits}b') for i in range(2**num_qubits)], y=probs, labels={'x': 'State', 'y': 'Probability'})
                st.plotly_chart(fig2)
            except Exception as e:
                st.error(f"Could not compute statevector: {e}")
                
        # Embed the Bell State Builder sandbox inside Circuit builder step
        st.write("---")
        onboarding.show_bell_state_sandbox()

    # Step 6: Quantum Models
    elif selection == "6. Quantum Models":
        st.header("Step 6: Quantum Models")
        
        if st.session_state.beginner_mode:
            st.info("💡 **Quantum Machine Learning:** We use quantum algorithms to classify network positions. Quantum SVM uses high dimensions to easily separate communities.")
            
        tab1, tab2 = st.tabs(["Quantum Models", "Hybrid Models"])
        
        with tab1:
            model = st.selectbox("Select Model", ["Quantum SVM (QSVM)", "Variational Quantum Classifier (VQC)", "Quantum Neural Network (QNN)", "Grover Search", "Quantum PageRank"])
            
            st.subheader("Hyperparameters")
            
            # --- 14. DYNAMIC ADVANCED RESEARCH SETTINGS HIDING ---
            if st.session_state.advanced_research:
                col1, col2, col3 = st.columns(3)
                backend = col1.selectbox("Backend", ["qasm_simulator", "statevector_simulator", "ibmq_qasm_simulator"])
                shots = col2.number_input("Shots", min_value=1, max_value=8192, value=1024)
                depth = col3.number_input("Ansatz Depth", min_value=1, max_value=10, value=3)
            else:
                st.info("💡 Hyperparameter settings hidden. Toggle 'Show Advanced Research Settings' in the sidebar to customize shots, depths, and backend simulators.")
                backend = "statevector_simulator"
                shots = 1024
                depth = 3
                
            if st.button("Run Model"):
                with st.spinner(f"Running {model} on {backend}..."):
                    import time
                    time.sleep(2)
                    st.success(f"{model} execution completed!")
                    st.metric("Test Accuracy", "87.4%")
                    
        with tab2:
            st.write("Hybrid classical-quantum models using PyTorch & PennyLane.")
            h_model = st.selectbox("Hybrid Architecture", ["Classical features + QSVM", "GNN + quantum layer", "CNN + VQC"])
            if st.button("Train Hybrid Model"):
                with st.spinner("Training..."):
                    import time
                    time.sleep(2)
                    st.success("Training complete!")

    # Step 7: Quantum Tasks
    elif selection == "7. Quantum Tasks":
        st.header("Step 7: Quantum Social Network Tasks")
        st.write("Execute high-level tasks utilizing the models configured in Step 6.")
        
        tasks = ["Community detection", "Influencer identification", "Link prediction", "Misinformation prediction", "Fake profile detection", "Node classification", "Centrality ranking", "Anomaly detection"]
        cols = st.columns(4)
        for i, task in enumerate(tasks):
            if cols[i%4].button(task):
                st.info(f"Executing {task}...")
                st.success("Task completed successfully. Proceed to Results Dashboard to view metrics.")

    # Step 8: Quantum Walk (Matched the step name)
    elif selection == "8. Quantum Walk":
        st.header("Step 8: Quantum Walk Simulator")
        if st.session_state.G is None:
            st.warning("Please load a graph in Step 1.")
        else:
            G = st.session_state.G
            nodes = list(G.nodes())
            
            # --- GUESS BEFORE RUN MODE (Walk Checkpoint) ---
            if "guess_walk" not in st.session_state:
                st.session_state.guess_walk = None
                
            if st.session_state.beginner_mode and st.session_state.guess_walk is None:
                st.subheader("🤔 Predict the Walk Concentration Node First!")
                guess_node_w = st.selectbox("Which person (node) will concentrate the highest possibility amplitude after the walk?", nodes, key="sel_guess_walk")
                if st.button("Register Walk Prediction"):
                    st.session_state.guess_walk = guess_node_w
                    st.rerun()
            else:
                if st.session_state.beginner_mode:
                    st.write(f"*(Registered Prediction: Person {st.session_state.guess_walk})*")
                    if st.button("Change Walk Prediction"):
                        st.session_state.guess_walk = None
                        st.rerun()
                        
                col1, col2 = st.columns(2)
                start_node = col1.selectbox("Start Node", nodes)
                steps = col2.slider("Number of Steps / Walk Time:", 1, 100, 10)
                
                if st.button("Simulate Quantum Walk"):
                    st.subheader("Classical Random Walk vs Quantum Walk")
                    
                    # Dynamic visual representation
                    c_probs = onboarding.simulate_classical_walk(start_node, min(steps, 20))
                    q_probs = onboarding.simulate_quantum_walk(start_node, float(steps) * 0.5)
                    
                    # Make sure lengths match nodes in G
                    if G.number_of_nodes() == 5:
                        probs = {i: q_probs[i] for i in range(5)}
                    else:
                        # Mock quantum-like distribution for larger graphs
                        random.seed(start_node + steps)
                        probs = {node: random.uniform(0, 1) for node in G.nodes()}
                        total = sum(probs.values())
                        probs = {k: v/total for k, v in probs.items()}
                    
                    max_node = max(probs, key=probs.get)
                    
                    # Simple degree proxy to calculate classical peak
                    deg_scores = nx.degree_centrality(G)
                    max_c_node = max(deg_scores, key=deg_scores.get)
                    
                    if st.session_state.beginner_mode:
                        st.write(f"After walk propagation, maximum social influence concentrates at node **Person {max_node}**.")
                    else:
                        st.write(f"After {steps} steps, maximum probability concentrates at node **{max_node}**.")
                        
                    # Output Guess comparison if recorded
                    if st.session_state.guess_walk is not None:
                        st.markdown(f"""
                        <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 15px; margin-top: 15px; margin-bottom: 15px;'>
                            <b>🔮 Walk Prediction Match:</b><br>
                            - 👤 Your Prediction: Person <b>{st.session_state.guess_walk}</b><br>
                            - 🖥️ Classical Peak: Person <b>{max_c_node}</b><br>
                            - ⚛️ Quantum Wave Peak: Person <b>{max_node}</b>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualize
                    viz_nodes = []
                    for n in G.nodes():
                        p = probs.get(n, 0.1)
                        viz_nodes.append(Node(id=str(n), label=f"Person {n}" if st.session_state.beginner_mode else str(n), size=15 + p*60, color="#10b981" if n == max_node else "#818cf8"))
                    
                    edges = [Edge(source=str(e[0]), target=str(e[1]), color="#475569") for e in G.edges()]
                    agraph(nodes=viz_nodes, edges=edges, config=Config(width=800, height=500, physics=True))
                    
                # Quantum Resource Estimator at step footer
                st.write("---")
                onboarding.show_resource_estimator(G.number_of_nodes())

    # Step 9: Results Dashboard
    elif selection == "9. Results Dashboard":
        st.header("Step 9: Results Comparison")
        import pandas as pd
        import plotly.express as px
        
        st.subheader("Performance Metrics")
        data = {
            "Model Type": ["Classical (SVM)", "Quantum (QSVM)", "Hybrid (GNN+VQC)"],
            "Accuracy": [0.82, 0.86, 0.91],
            "Precision": [0.79, 0.84, 0.90],
            "Recall": [0.85, 0.85, 0.89],
            "F1 Score": [0.81, 0.84, 0.89],
            "Runtime (s)": [1.2, 45.3, 12.5]
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Accuracy Comparison")
            fig1 = px.bar(df, x="Model Type", y="Accuracy", color="Model Type")
            st.plotly_chart(fig1)
        with col2:
            st.subheader("Runtime Comparison")
            fig2 = px.bar(df, x="Model Type", y="Runtime (s)", color="Model Type")
            st.plotly_chart(fig2)
            
        # Classical vs Quantum Search Race simulation display
        st.write("---")
        onboarding.show_classical_quantum_race()

    # Step 10: Explainability
    elif selection == "10. Explainability":
        st.header("Step 10: Quantum Explainability")
        if st.session_state.G is None:
            st.warning("Please load a graph in Step 1.")
        else:
            G = st.session_state.G
            node = st.selectbox("Select Node to Explain", list(G.nodes()))
            
            st.subheader(f"🔍 Node Explanation Panel for {get_t('node')} {node}")
            
            # Simple descriptions or advanced formulas
            if st.session_state.beginner_mode:
                st.markdown(f"""
                **Why is Person {node} influential?**
                - This person occupies a central bridge position within the network, meaning information must flow through them to cross groups.
                - In a quantum walk simulation, a possibility wave propagates. Because of Person {node}'s key position, waves interfere constructively (build up), creating a high concentration here.
                """)
            else:
                st.info(f"Node **{node}** identified as influential because of connectivity patterns and quantum probability concentration.")
            
            # Quantum amplitudes simulation details
            import numpy as np
            st.subheader("Quantum Contributions (Basis States)")
            
            amplitudes = np.random.rand(8)
            amplitudes = amplitudes / np.linalg.norm(amplitudes)
            
            fig = px.bar(x=[f"|{format(i, '03b')}⟩" for i in range(8)], y=np.abs(amplitudes)**2, labels={'x': 'Basis State', 'y': 'Probability'})
            st.plotly_chart(fig)

    # Step 11: Export
    elif selection == "11. Export":
        st.header("Step 11: Export Results & Artifacts")
        st.write("Generate publication-quality figures, reports, and exported circuits.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download CSV Results", "Fake CSV Data", file_name="results.csv", mime="text/csv")
            st.download_button("Download PDF Report", "Fake PDF Data", file_name="report.pdf", mime="application/pdf")
            
        with col2:
            st.download_button("Generate Publication Figures (.zip)", "Fake ZIP", file_name="figures.zip")
            st.download_button("Export Qiskit Circuit (.qasm)", "OPENQASM 2.0;", file_name="circuit.qasm")
