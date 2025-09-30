""" 
Quantum Cryptography Workshop Platform
Standard Chartered GBS Bangalore
Python + Streamlit + MongoDB + IBM Qiskit

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import List, Tuple, Dict
import json

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
from qiskit_ibm_runtime import QiskitRuntimeService

# MongoDB imports
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Configuration
st.set_page_config(
    page_title="Quantum Cryptography Workshop - Standard Chartered",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #0066cc 0%, #00ccff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .quantum-state {
        font-family: 'Courier New', monospace;
        background: #1e1e1e;
        color: #00ff00;
        padding: 1rem;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    .security-alert {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .secure {
        background: #d4edda;
        border-left: 4px solid #28a745;
    }
    .compromised {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


class QuantumCryptoEngine:
    """Quantum Cryptography Engine with Qiskit"""
    
    def __init__(self):
        self.simulator = AerSimulator()
        self.ibm_token = st.secrets.get("IBM_QUANTUM_TOKEN", "")
        self.service = None
        
        if self.ibm_token:
            try:
                self.service = QiskitRuntimeService(
                    channel="ibm_quantum",
                    token=self.ibm_token
                )
            except Exception as e:
                st.warning(f"IBM Quantum connection failed: {e}. Using simulator.")
    
    def run_bb84_protocol(self, n_bits: int = 100, with_eavesdropper: bool = False) -> Dict:
        """Execute BB84 QKD Protocol"""
        
        # Step 1: Alice generates random bits and bases
        alice_bits = np.random.randint(0, 2, n_bits).tolist()
        alice_bases = np.random.randint(0, 2, n_bits).tolist()
        
        # Step 2: Bob chooses random measurement bases
        bob_bases = np.random.randint(0, 2, n_bits).tolist()
        
        # Step 3: Quantum transmission and measurement
        bob_measurements = []
        
        for i in range(n_bits):
            # Create quantum circuit
            qc = QuantumCircuit(1, 1)
            
            # Alice's state preparation
            if alice_bits[i] == 1:
                qc.x(0)
            if alice_bases[i] == 1:
                qc.h(0)
            
            # Bob's measurement
            if bob_bases[i] == 1:
                qc.h(0)
            qc.measure(0, 0)
            
            # Execute
            job = self.simulator.run(transpile(qc, self.simulator), shots=1)
            result = job.result()
            counts = result.get_counts()
            measurement = int(list(counts.keys())[0])
            
            # Simulate eavesdropper
            if with_eavesdropper and np.random.random() < 0.25:
                measurement = 1 - measurement
            
            bob_measurements.append(measurement)
        
        # Step 4: Key sifting
        shared_key = []
        matching_indices = []
        
        for i in range(n_bits):
            if alice_bases[i] == bob_bases[i]:
                shared_key.append(alice_bits[i])
                matching_indices.append(i)
        
        # Step 5: Error detection
        if matching_indices:
            test_size = max(1, len(matching_indices) // 10)
            test_indices = np.random.choice(matching_indices, size=test_size, replace=False)
            
            errors = sum(1 for idx in test_indices if alice_bits[idx] != bob_measurements[idx])
            error_rate = errors / test_size
        else:
            error_rate = 0
        
        # Security analysis
        threshold = 0.11
        security_status = "SECURE" if error_rate < threshold else "COMPROMISED"
        eavesdropping_detected = error_rate > threshold
        
        return {
            "alice_bits": alice_bits,
            "alice_bases": alice_bases,
            "bob_bases": bob_bases,
            "bob_measurements": bob_measurements,
            "shared_key": shared_key,
            "matching_indices": matching_indices,
            "error_rate": error_rate,
            "security_status": security_status,
            "eavesdropping_detected": eavesdropping_detected,
            "sifting_efficiency": len(matching_indices) / n_bits,
            "key_length": len(shared_key)
        }
    
    def create_quantum_circuit_diagram(self, bit: int, basis: int) -> QuantumCircuit:
        """Create quantum circuit for visualization"""
        qc = QuantumCircuit(1, 1, name=f"BB84: bit={bit}, basis={'X' if basis else 'Z'}")
        
        if bit == 1:
            qc.x(0)
        if basis == 1:
            qc.h(0)
        qc.measure(0, 0)
        
        return qc
    
    def calculate_key_rate(self, distance_km: float, qber: float = 0.02) -> float:
        """Calculate secret key rate"""
        alpha = 0.2 # dB/km attenuation
        eta = 10 ** (-alpha * distance_km / 10)
        
        q_total = qber + (1 - eta) / 2
        
        if q_total > 0.11:
            return 0
        
        h = lambda x: -x * np.log2(x) - (1-x) * np.log2(1-x) if 0 < x < 1 else 0
        r = eta * (1 - 2 * h(q_total))
        
        return max(0, r * 1e6) # Convert to bps (assuming 1 MHz clock)


class MongoDBHandler:
    """MongoDB Handler for session and results storage"""
    
    def __init__(self):
        try:
            # Use MongoDB Atlas or local instance
            mongo_uri = st.secrets.get("MONGODB_URI", "mongodb://localhost:27017/")
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client['quantum_crypto']
            self.sessions = self.db['sessions']
            self.results = self.db['results']
            st.success("✅ Connected to MongoDB")
        except ConnectionFailure:
            st.warning("⚠️ MongoDB not available. Running in demo mode.")
            self.client = None
    
    def save_session(self, session_data: Dict) -> str:
        """Save session to MongoDB"""
        if self.client:
            session_data['timestamp'] = datetime.now()
            result = self.sessions.insert_one(session_data)
            return str(result.inserted_id)
        return "demo_session"
    
    def save_results(self, results_data: Dict) -> str:
        """Save BB84 results to MongoDB"""
        if self.client:
            results_data['timestamp'] = datetime.now()
            result = self.results.insert_one(results_data)
            return str(result.inserted_id)
        return "demo_result"
    
    def get_statistics(self) -> Dict:
        """Get aggregate statistics"""
        if self.client:
            total_sessions = self.sessions.count_documents({})
            total_results = self.results.count_documents({})
            
            avg_error_rate = list(self.results.aggregate([
                {"$group": {"_id": None, "avg_error": {"$avg": "$error_rate"}}}
            ]))
            
            return {
                "total_sessions": total_sessions,
                "total_results": total_results,
                "avg_error_rate": avg_error_rate[0]['avg_error'] if avg_error_rate else 0
            }
        return {"total_sessions": 0, "total_results": 0, "avg_error_rate": 0}


def render_sidebar():
    """Render sidebar with configuration"""
    st.sidebar.title("🔐 Quantum Crypto Workshop")
    st.sidebar.markdown("---")
    
    st.sidebar.header("Configuration")
    
    page = st.sidebar.radio(
        "Select Module",
        ["🏠 Home", "🔬 BB84 Demo", "📊 Performance Analysis", 
         "🛡️ Security Analysis", "📈 Statistics", "📚 Technical Docs"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("BB84 Parameters")
    
    n_bits = st.sidebar.slider("Number of Bits", 10, 500, 100)
    with_eve = st.sidebar.checkbox("Enable Eavesdropper (Eve)", value=False)
    distance = st.sidebar.slider("Distance (km)", 10, 200, 50)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Standard Chartered GBS**  
    Cybersecurity Team  
    Bangalore, India
    """)
    
    return page, n_bits, with_eve, distance


def render_home():
    """Render home page"""
    st.markdown('<h1 class="main-header">Quantum Cryptography Workshop</h1>', unsafe_allow_html=True)
    st.markdown("### Standard Chartered GBS Bangalore - Cybersecurity Architecture")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔐 Quantum Security</h3>
            <p>Information-theoretic security guaranteed by quantum mechanics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🚀 BB84 Protocol</h3>
            <p>Industry-standard quantum key distribution implementation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ Eavesdrop Detection</h3>
            <p>Automatic detection of quantum eavesdropping attempts</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🎯 Workshop Objectives")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Technical Understanding:**
        - Quantum key distribution protocols (BB84, E91, SARG04)
        - Quantum channel engineering and physics
        - Error correction and privacy amplification
        - Integration with existing infrastructure
        
        **Hands-on Experience:**
        - Interactive BB84 simulation with IBM Qiskit
        - Real quantum circuit visualization
        - Performance analysis across distances
        - Security analysis and threat detection
        """)
    
    with col2:
        st.markdown("""
        **Business Impact:**
        - Risk quantification for quantum threats
        - ROI analysis for QKD deployment
        - Regulatory compliance strategies
        - Implementation roadmap for Standard Chartered
        
        **Strategic Planning:**
        - Vendor evaluation framework
        - Phased deployment approach
        - Team capability building
        - Industry collaboration opportunities
        """)
    
    st.markdown("---")
    
    st.subheader("📊 Quick Stats")
    
    # Get MongoDB statistics
    db_handler = MongoDBHandler()
    stats = db_handler.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Sessions", stats['total_sessions'])
    col2.metric("BB84 Runs", stats['total_results'])
    col3.metric("Avg Error Rate", f"{stats['avg_error_rate']*100:.2f}%")
    col4.metric("Security Status", "✅ SECURE")


def render_bb84_demo(n_bits: int, with_eve: bool, distance: float):
    """Render BB84 demonstration"""
    st.title("🔬 BB84 Quantum Key Distribution Demo")
    
    engine = QuantumCryptoEngine()
    db_handler = MongoDBHandler()
    
    # Control panel
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        **Protocol Parameters:**
        - Number of qubits: `{n_bits}`
        - Eavesdropper present: `{'Yes (Eve active)' if with_eve else 'No'}`
        - Transmission distance: `{distance} km`
        """)
    
    with col2:
        if st.button("🚀 Run BB84 Protocol", type="primary"):
            st.session_state['bb84_running'] = True
    
    if st.session_state.get('bb84_running', False):
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1
        status_text.text("Step 1/5: Alice generating random bits and bases...")
        progress_bar.progress(20)
        time.sleep(0.5)
        
        # Step 2
        status_text.text("Step 2/5: Bob choosing random measurement bases...")
        progress_bar.progress(40)
        time.sleep(0.5)
        
        # Step 3
        status_text.text("Step 3/5: Quantum transmission and measurement...")
        progress_bar.progress(60)
        
        # Run BB84
        result = engine.run_bb84_protocol(n_bits, with_eve)
        
        # Step 4
        status_text.text("Step 4/5: Performing key sifting...")
        progress_bar.progress(80)
        time.sleep(0.5)
        
        # Step 5
        status_text.text("Step 5/5: Analyzing security and errors...")
        progress_bar.progress(100)
        time.sleep(0.5)
        
        status_text.text("✅ BB84 Protocol Complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Save to MongoDB
        db_handler.save_results({
            "n_bits": n_bits,
            "with_eavesdropper": with_eve,
            "distance": distance,
            "error_rate": result['error_rate'],
            "key_length": result['key_length'],
            "security_status": result['security_status']
        })
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Results Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Shared Key Length", f"{result['key_length']} bits")
        col2.metric("Sifting Efficiency", f"{result['sifting_efficiency']*100:.1f}%")
        col3.metric("Error Rate (QBER)", f"{result['error_rate']*100:.2f}%")
        col4.metric("Security Status", result['security_status'])
        
        # Security alert
        if result['eavesdropping_detected']:
            st.markdown("""
            <div class="security-alert compromised">
                <strong>🚨 SECURITY ALERT: Eavesdropping Detected!</strong><br>
                High error rate indicates possible quantum eavesdropping attempt.
                The communication channel is compromised. Abort key generation.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="security-alert secure">
                <strong>✅ CHANNEL SECURE</strong><br>
                Low error rate indicates secure quantum channel.
                Proceed with key usage for encrypted communications.
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed data
        st.markdown("---")
        st.subheader("🔤 Quantum Data (First 20 bits)")
        
        df = pd.DataFrame({
            "Index": range(20),
            "Alice Bit": result['alice_bits'][:20],
            "Alice Basis": ["Z" if b == 0 else "X" for b in result['alice_bases'][:20]],
            "Bob Basis": ["Z" if b == 0 else "X" for b in result['bob_bases'][:20]],
            "Bob Measurement": result['bob_measurements'][:20],
            "Bases Match": ["✓" if result['alice_bases'][i] == result['bob_bases'][i] else "✗" for i in range(20)],
            "Used in Key": ["✓" if i in result['matching_indices'][:20] else "✗" for i in range(20)]
        })
        
        st.dataframe(df, use_container_width=True)
        
        # Quantum circuits
        st.markdown("---")
        st.subheader("⚛️ Quantum Circuit Visualization")
        
        cols = st.columns(4)
        for i in range(4):
            with cols[i]:
                qc = engine.create_quantum_circuit_diagram(
                    result['alice_bits'][i],
                    result['alice_bases'][i]
                )
                st.text(f"Qubit {i}")
                st.code(qc.draw(output='text'), language=None)
        
        # Shared key
        if result['shared_key']:
            st.markdown("---")
            st.subheader("🔑 Generated Shared Key")
            
            key_binary = ''.join(map(str, result['shared_key'][:128]))
            key_hex = hex(int(key_binary[:128] if len(key_binary) >= 128 else key_binary.ljust(128, '0'), 2))[2:].zfill(32)
            
            st.markdown(f"""
            <div class="quantum-state">
            <strong>Binary (first 128 bits):</strong><br>
            {key_binary}<br><br>
            <strong>Hexadecimal (32 bytes):</strong><br>
            {key_hex}
            </div>
            """, unsafe_allow_html=True)


def render_performance_analysis(distance: float):
    """Render performance analysis"""
    st.title("📊 QKD Performance Analysis")
    
    engine = QuantumCryptoEngine()
    
    st.markdown("### Key Rate vs Distance Analysis")
    
    # Calculate key rates for different distances
    distances = np.linspace(10, 200, 50)
    key_rates_clean = [engine.calculate_key_rate(d, 0.02) for d in distances]
    key_rates_noisy = [engine.calculate_key_rate(d, 0.05) for d in distances]
    
    # Plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=distances,
        y=key_rates_clean,
        mode='lines',
        name='Low Noise (QBER=2%)',
        line=dict(color='green', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=distances,
        y=key_rates_noisy,
        mode='lines',
        name='Higher Noise (QBER=5%)',
        line=dict(color='orange', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[distance],
        y=[engine.calculate_key_rate(distance, 0.02)],
        mode='markers',
        name='Current Configuration',
        marker=dict(color='red', size=15, symbol='star')
    ))
    
    fig.update_layout(
        title="QKD Secret Key Rate vs Transmission Distance",
        xaxis_title="Distance (km)",
        yaxis_title="Key Rate (bits/second)",
        yaxis_type="log",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Error rate analysis
    st.markdown("---")
    st.markdown("### Error Rate vs Distance")
    
    qber_distances = np.linspace(10, 200, 50)
    qber_values = [0.02 + (1 - 10**(-0.2*d/10))/2 for d in qber_distances]
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=qber_distances,
        y=[q*100 for q in qber_values],
        mode='lines',
        name='QBER',
        line=dict(color='blue', width=3),
        fill='tozeroy'
    ))
    
    fig2.add_hline(y=11, line_dash="dash", line_color="red", 
                   annotation_text="Security Threshold (11%)")
    
    fig2.update_layout(
        title="Quantum Bit Error Rate vs Distance",
        xaxis_title="Distance (km)",
        yaxis_title="QBER (%)",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Performance metrics table
    st.markdown("---")
    st.subheader("📈 Performance Metrics")
    
    performance_data = []
    for d in [25, 50, 75, 100, 150, 200]:
        rate = engine.calculate_key_rate(d, 0.02)
        qber = 0.02 + (1 - 10**(-0.2*d/10))
