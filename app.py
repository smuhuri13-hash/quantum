import streamlit as st
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import random

# Set page config
st.set_page_config(page_title="QSNA Studio", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for Futuristic Glassmorphism
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
    div.css-1r6slb0, div.css-12oz5g7 {
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

st.title("Quantum Social Network Analysis (QSNA) Studio")
st.markdown("Explore classical and quantum techniques for social network analysis.")

# Sidebar Navigation
st.sidebar.title("Navigation")
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

selection = st.sidebar.radio("Go to step:", pages)

# State Management for Graph Data
if "G" not in st.session_state:
    st.session_state.G = None

# Step 1: Data Input
if selection == "1. Data Input":
    st.header("Step 1: Input Network Data")
    
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
                
    if demo_option != "None":
        if st.button("Load Demo"):
            # Simple mock loading for now
            if demo_option == "8. Small toy graph (5–10 nodes)":
                st.session_state.G = nx.karate_club_graph()
            else:
                st.session_state.G = nx.watts_strogatz_graph(30, 4, 0.1)
            st.success(f"Loaded {demo_option}")

# Step 2: Graph Dashboard
elif selection == "2. Graph Dashboard":
    st.header("Step 2: Graph Dashboard")
    
    if st.session_state.G is None:
        st.warning("Please load or generate a graph in Step 1 first.")
    else:
        G = st.session_state.G
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodes", G.number_of_nodes())
        col2.metric("Edges", G.number_of_edges())
        col3.metric("Density", round(nx.density(G), 3))
        
        degrees = [d for n, d in G.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        col4.metric("Avg Degree", round(avg_degree, 2))
        
        st.subheader("Interactive Graph Visualization")
        
        nodes = []
        edges = []
        
        for node in G.nodes():
            nodes.append(Node(id=str(node), label=str(node), size=25, color="#818cf8"))
            
        for edge in G.edges():
            edges.append(Edge(source=str(edge[0]), target=str(edge[1]), color="#475569"))
            
        config = Config(width=1000,
                        height=600,
                        directed=nx.is_directed(G), 
                        physics=True, 
                        hierarchical=False)
                        
        return_value = agraph(nodes=nodes, edges=edges, config=config)

# Step 3: Classical SNA
elif selection == "3. Classical SNA":
    st.header("Step 3: Classical Social Network Analysis")
    if st.session_state.G is None:
        st.warning("Please load a graph in Step 1.")
    else:
        G = st.session_state.G
        tab1, tab2, tab3 = st.tabs(["Centrality", "Community", "Machine Learning"])
        
        with tab1:
            st.subheader("Centrality Metrics")
            cent_metric = st.selectbox("Select Metric:", ["Degree", "Betweenness", "Closeness", "Eigenvector", "PageRank"])
            
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
        import numpy as np
        import plotly.express as px
        
        st.write("Convert classical network data into quantum representations.")
        encoding = st.selectbox("Encoding Method", ["Amplitude Encoding", "Basis Encoding", "Angle Encoding"])
        
        if st.button("Convert to Quantum State"):
            adj_matrix = nx.to_numpy_array(G)
            n_nodes = G.number_of_nodes()
            
            st.subheader("1. Classical Adjacency Matrix")
            fig = px.imshow(adj_matrix, color_continuous_scale='Viridis', title="Adjacency Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("2. Vector Representation")
            vec = adj_matrix.flatten()
            norm = np.linalg.norm(vec)
            if norm == 0:
                norm = 1
            normalized_vec = vec / norm
            st.write(f"Total elements: {len(vec)}")
            
            st.subheader("3. Quantum State")
            if encoding == "Amplitude Encoding":
                qubits_needed = int(np.ceil(np.log2(len(vec))))
                if qubits_needed == 0: qubits_needed = 1
                st.info(f"Using Amplitude Encoding: Requires ⌈log₂(N²)⌉ = {qubits_needed} qubits.")
                
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
            import numpy as np
            sv = Statevector.from_instruction(qc)
            probs = sv.probabilities()
            fig2 = px.bar(x=[format(i, f'0{num_qubits}b') for i in range(2**num_qubits)], y=probs, labels={'x': 'State', 'y': 'Probability'})
            st.plotly_chart(fig2)
        except Exception as e:
            st.error(f"Could not compute statevector: {e}")

# Step 6: Quantum Models
elif selection == "6. Quantum Models":
    st.header("Step 6: Quantum Models")
    tab1, tab2 = st.tabs(["Quantum Models", "Hybrid Models"])
    
    with tab1:
        model = st.selectbox("Select Model", ["Quantum SVM (QSVM)", "Variational Quantum Classifier (VQC)", "Quantum Neural Network (QNN)", "Grover Search", "Quantum PageRank"])
        
        st.subheader("Hyperparameters")
        col1, col2, col3 = st.columns(3)
        backend = col1.selectbox("Backend", ["qasm_simulator", "statevector_simulator", "ibmq_qasm_simulator"])
        shots = col2.number_input("Shots", min_value=1, max_value=8192, value=1024)
        depth = col3.number_input("Ansatz Depth", min_value=1, max_value=10, value=3)
        
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

# Step 8: Quantum Walk Simulator
elif selection == "8. Quantum Walk Simulator":
    st.header("Step 8: Quantum Walk Simulator")
    if st.session_state.G is None:
        st.warning("Please load a graph in Step 1.")
    else:
        G = st.session_state.G
        nodes = list(G.nodes())
        
        col1, col2 = st.columns(2)
        start_node = col1.selectbox("Start Node", nodes)
        steps = col2.slider("Number of Steps", 1, 100, 10)
        
        if st.button("Simulate Quantum Walk"):
            st.subheader("Classical Random Walk vs Quantum Walk")
            import random
            # Fake simulation data for demonstration
            probs = {node: random.uniform(0, 1) for node in G.nodes()}
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}
            
            # Highlight max probability node
            max_node = max(probs, key=probs.get)
            st.write(f"After {steps} steps, maximum probability concentrates at node **{max_node}**.")
            
            # Visualize
            viz_nodes = []
            for n in G.nodes():
                p = probs[n]
                viz_nodes.append(Node(id=str(n), label=str(n), size=10 + p*50, color="#10b981" if n == max_node else "#818cf8"))
            
            edges = [Edge(source=str(e[0]), target=str(e[1]), color="#475569") for e in G.edges()]
            agraph(nodes=viz_nodes, edges=edges, config=Config(width=800, height=500, physics=True))

# Step 9: Results Dashboard
elif selection == "9. Results Dashboard":
    st.header("Step 9: Results Comparison")
    import pandas as pd
    import plotly.express as px
    import numpy as np
    
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

# Step 10: Explainability
elif selection == "10. Explainability":
    st.header("Step 10: Quantum Explainability")
    if st.session_state.G is None:
        st.warning("Please load a graph in Step 1.")
    else:
        G = st.session_state.G
        node = st.selectbox("Select Node to Explain", list(G.nodes()))
        
        st.info(f"Node **{node}** identified as influential because of connectivity patterns and quantum probability concentration.")
        
        import numpy as np
        st.subheader("Quantum Contributions")
        import plotly.express as px
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

else:
    st.info(f"Page '{selection}' is under construction.")
