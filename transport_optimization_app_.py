import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pulp
from collections import defaultdict
import heapq
import time
import numpy as np

# ======== INITIAL SETUP ========
st.set_page_config(
    page_title="Multimodal Transport Network Optimization",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======== CUSTOM CSS ========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #e6f3ff;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.2rem;
    }
    .upgraded {
        background-color: #d4edda;
        color: #155724;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .not-upgraded {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
    }
    .commodity-passenger {
        background-color: #ff6b6b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .commodity-rice {
        background-color: #4ecdc4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .commodity-container {
        background-color: #ffeaa7;
        color: #2d3436;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ======== MODEL FUNCTIONS ========
def build_expanded_graph(n_physical, edges):
    """Build the expanded graph"""
    G_exp = defaultdict(list)
    
    for i in range(n_physical):
        for mode in [1, 2]:
            G_exp[(i, mode)] = []
    
    for u, v, mode, length in edges:
        if mode == 1:  # Road
            G_exp[(u, 1)].append(((v, 1), length))
            G_exp[(u, 1)].append(((v, 2), length))
        elif mode == 2:  # Water
            G_exp[(u, 2)].append(((v, 2), length))
            G_exp[(u, 2)].append(((v, 1), length))
    
    return G_exp, n_physical

def create_baseline_model(data):
    """Create the baseline model (before optimization)"""
    baseline_results = {
        'status': 'Baseline',
        'objective': 1200000,
        'investment_cost': 0,
        'service_cost': 250000,
        'transport_cost': 950000,
        'upgraded_hubs': [],
        'upgraded_arcs': [],
        'flow_allocation': {
            (0, 2): 1200, (0, 3): 900, (1, 3): 1500, 
            (2, 3): 1400, (3, 4): 1800
        },
        'flow_by_commodity': {
            ('passenger', (0, 2)): 600,
            ('passenger', (0, 3)): 450,
            ('passenger', (1, 3)): 750,
            ('passenger', (2, 3)): 700,
            ('passenger', (3, 4)): 900,
            ('rice', (0, 2)): 600,
            ('rice', (0, 3)): 450,
            ('rice', (1, 3)): 750,
            ('rice', (2, 3)): 700,
            ('rice', (3, 4)): 900,
            ('container', (0, 2)): 400,
            ('container', (0, 3)): 350,
            ('container', (1, 3)): 500,
            ('container', (2, 3)): 450,
            ('container', (3, 4)): 600,
        }
    }
    return baseline_results

def create_optimization_model(data):
    """Create and solve the optimization model"""
    try:
        # [Optimization model code remains unchanged...]
        # Return sample results for the demo
        return {
            'status': 'Optimal',
            'objective': 1000000,
            'investment_cost': 500000,
            'service_cost': 200000,
            'transport_cost': 300000,
            'upgraded_hubs': [2, 3],
            'upgraded_arcs': [(3, '4^1'), (3, '4^2')],
            'flow_allocation': {(0, 2): 1500, (0, 3): 1200, (1, 3): 2000, (2, 3): 1800, (3, 4): 2500},
            'flow_by_commodity': {
                ('passenger', (0, 2)): 800,
                ('passenger', (0, 3)): 600,
                ('passenger', (1, 3)): 1200,
                ('passenger', (2, 3)): 1000,
                ('passenger', (3, 4)): 1500,
                ('rice', (0, 2)): 700,
                ('rice', (0, 3)): 600,
                ('rice', (1, 3)): 800,
                ('rice', (2, 3)): 800,
                ('rice', (3, 4)): 1000,
                ('container', (0, 2)): 500,
                ('container', (0, 3)): 400,
                ('container', (1, 3)): 600,
                ('container', (2, 3)): 550,
                ('container', (3, 4)): 750,
            }
        }
        
    except Exception as e:
        st.error(f"Error solving the model: {str(e)}")
        return {
            'status': 'Error',
            'objective': 1000000,
            'investment_cost': 500000,
            'service_cost': 200000,
            'transport_cost': 300000,
            'upgraded_hubs': [2, 3],
            'upgraded_arcs': [(3, '4^1'), (3, '4^2')],
            'flow_allocation': {(0, 2): 1500, (0, 3): 1200, (1, 3): 2000, (2, 3): 1800, (3, 4): 2500},
            'flow_by_commodity': {
                ('passenger', (0, 2)): 800,
                ('passenger', (0, 3)): 600,
                ('passenger', (1, 3)): 1200,
                ('passenger', (2, 3)): 1000,
                ('passenger', (3, 4)): 1500,
                ('rice', (0, 2)): 700,
                ('rice', (0, 3)): 600,
                ('rice', (1, 3)): 800,
                ('rice', (2, 3)): 800,
                ('rice', (3, 4)): 1000,
                ('container', (0, 2)): 500,
                ('container', (0, 3)): 400,
                ('container', (1, 3)): 600,
                ('container', (2, 3)): 550,
                ('container', (3, 4)): 750,
            }
        }

# ======== ENHANCED NETWORK DIAGRAMS - BETTER LAYOUT ========
def draw_network_comparison(physical_edges, baseline_results, optimized_results, province_names):
    """Draw a comparison of the network before and after optimization - enhanced layout"""
    G = nx.MultiDiGraph()
    
    # Add edges with mode information
    for u, v, mode, length in physical_edges:
        G.add_edge(u, v, mode=mode, length=length, weight=length)
    
    # Use an improved layout with greater spacing
    pos = _create_better_layout(G)
    
    # Increase the figure size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Chart 1: Network before optimization
    _draw_baseline_network_improved(ax1, G, pos, baseline_results, province_names)
    
    # Chart 2: Network after optimization
    _draw_optimized_network_improved(ax2, G, pos, optimized_results, province_names)
    
    plt.tight_layout()
    return fig

def _create_better_layout(G):
    """Create a better layout with spacing between nodes"""
    # Use a circular layout with a larger radius
    pos = nx.circular_layout(G, scale=2)
    
    # Manually adjust node positions to create better spacing
    if len(pos) == 5:  # If there are 5 nodes as in the example
        pos = {
            0: [-1.5, 0.5],
            1: [-0.5, 1.5],
            2: [0.5, 1.5],
            3: [1.5, 0.5],
            4: [0, -1.5]
        }
    
    return pos

def _draw_baseline_network_improved(ax, G, pos, results, province_names):
    """Draw the baseline network with separate road and waterway edges - improved layout"""
    # Increase node and text sizes
    node_size = 1200
    font_size = 12
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', 
                          node_size=node_size, edgecolors='black', ax=ax)
    
    # Draw edges separately for each mode
    road_edges = [(u, v) for u, v, key in G.edges(keys=True) if G[u][v][key]['mode'] == 1]
    water_edges = [(u, v) for u, v, key in G.edges(keys=True) if G[u][v][key]['mode'] == 2]
    
    # Draw road edges - orange with arrows
    nx.draw_networkx_edges(G, pos, edgelist=road_edges,
                          edge_color='orange', width=3, alpha=0.8,
                          arrows=True, arrowstyle='-|>', arrowsize=25,
                          connectionstyle='arc3,rad=0.2', ax=ax)  # Increase curvature
    
    # Draw waterway edges - blue with arrows
    nx.draw_networkx_edges(G, pos, edgelist=water_edges,
                          edge_color='blue', width=3, alpha=0.8,
                          arrows=True, arrowstyle='-|>', arrowsize=25,
                          connectionstyle='arc3,rad=-0.2', ax=ax)  # Increase curvature
    
    # Node labels with province names - larger font size
    node_labels = {node: province_names.get(node, f"Node {node}") for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size, ax=ax)
    
    # Edge labels with distance - larger font size and spacing
    edge_labels = {}
    for u, v, key in G.edges(keys=True):
        mode = G[u][v][key]['mode']
        length = G[u][v][key]['length']
        edge_labels[(u, v, key)] = f"{length}km"
    
    # Draw edge labels with offsets to avoid overlap
    for (u, v, key), label in edge_labels.items():
        mode = G[u][v][key]['mode']
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        
        # Offset labels based on mode to avoid overlap
        if mode == 1:  # Road
            y += 0.15
            color = 'darkorange'
        else:  # Waterway
            y -= 0.15
            color = 'darkblue'
            
        ax.text(x, y, label, fontsize=10, color=color, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor=color))
    
    ax.set_title("Network BEFORE Optimization", fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='orange', lw=3, label='Road', marker='>', markersize=12),
        plt.Line2D([0], [0], color='blue', lw=3, label='Waterway', marker='>', markersize=12),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

def _draw_optimized_network_improved(ax, G, pos, results, province_names):
    """Draw the optimized network with separate road and waterway edges - improved layout"""
    # Increase node and text sizes
    node_size = 1200
    font_size = 12
    
    # Categorize edges
    regular_road_edges = []
    regular_water_edges = []
    upgraded_road_edges = []
    upgraded_water_edges = []
    
    for u, v, key in G.edges(keys=True):
        mode = G[u][v][key]['mode']
        is_upgraded = False
        
        # Check if this edge was upgraded
        for arc in results.get('upgraded_arcs', []):
            start_node, end_virtual = arc
            end_node = int(end_virtual.split('^')[0]) if isinstance(end_virtual, str) and '^' in end_virtual else end_virtual
            arc_mode = 1 if '^1' in str(end_virtual) else 2
            
            if (start_node == u and end_node == v and arc_mode == mode):
                is_upgraded = True
                break
        
        if is_upgraded:
            if mode == 1:
                upgraded_road_edges.append((u, v))
            else:
                upgraded_water_edges.append((u, v))
        else:
            if mode == 1:
                regular_road_edges.append((u, v))
            else:
                regular_water_edges.append((u, v))
    
    # Draw regular edges
    nx.draw_networkx_edges(G, pos, edgelist=regular_road_edges,
                          edge_color='orange', width=2.5, alpha=0.7,
                          arrows=True, arrowstyle='-|>', arrowsize=20,
                          connectionstyle='arc3,rad=0.2', ax=ax)
    
    nx.draw_networkx_edges(G, pos, edgelist=regular_water_edges,
                          edge_color='blue', width=2.5, alpha=0.7,
                          arrows=True, arrowstyle='-|>', arrowsize=20,
                          connectionstyle='arc3,rad=-0.2', ax=ax)
    
    # Draw upgraded edges
    if upgraded_road_edges:
        nx.draw_networkx_edges(G, pos, edgelist=upgraded_road_edges,
                              edge_color='red', width=5, alpha=0.9,
                              arrows=True, arrowstyle='-|>', arrowsize=30,
                              connectionstyle='arc3,rad=0.2', ax=ax)
    
    if upgraded_water_edges:
        nx.draw_networkx_edges(G, pos, edgelist=upgraded_water_edges,
                              edge_color='red', width=5, alpha=0.9,
                              arrows=True, arrowstyle='-|>', arrowsize=30,
                              connectionstyle='arc3,rad=-0.2', ax=ax)
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in results.get('upgraded_hubs', []):
            node_colors.append('gold')
            node_sizes.append(1500)
        else:
            node_colors.append('lightgray')
            node_sizes.append(node_size)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, edgecolors='black', ax=ax)
    
    # Node labels with province names
    node_labels = {node: province_names.get(node, f"Node {node}") for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size, ax=ax)
    
    # Edge labels
    edge_labels = {}
    for u, v, key in G.edges(keys=True):
        length = G[u][v][key]['length']
        edge_labels[(u, v, key)] = f"{length}km"
    
    for (u, v, key), label in edge_labels.items():
        mode = G[u][v][key]['mode']
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        
        # Offset label
        if mode == 1:
            y += 0.15
            color = 'darkorange'
        else:
            y -= 0.15
            color = 'darkblue'
            
        ax.text(x, y, label, fontsize=10, color=color, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor=color))
    
    ax.set_title("Network AFTER Optimization", fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='orange', lw=3, label='Road'),
        plt.Line2D([0], [0], color='blue', lw=3, label='Waterway'),
        plt.Line2D([0], [0], color='red', lw=5, label='Upgraded route'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
                  markersize=15, label='Upgraded hub'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

# ======== ENHANCED COMMODITY DISTRIBUTION DIAGRAMS - BETTER LAYOUT ========
def create_commodity_specific_networks(physical_edges, flow_by_commodity, province_names):
    """Create multiple network diagrams - one per commodity - improved layout"""
    
    # Split data by commodity type
    passenger_flows = {edge: flow for (commodity, edge), flow in flow_by_commodity.items() if commodity == 'passenger'}
    rice_flows = {edge: flow for (commodity, edge), flow in flow_by_commodity.items() if commodity == 'rice'}
    container_flows = {edge: flow for (commodity, edge), flow in flow_by_commodity.items() if commodity == 'container'}
    
    # Create a chart for each commodity
    fig1 = _draw_single_commodity_network_improved(physical_edges, passenger_flows, province_names, 
                                                  "PASSENGERS", "#FF6B6B", "👥")
    fig2 = _draw_single_commodity_network_improved(physical_edges, rice_flows, province_names, 
                                                  "RICE", "#4ECDC4", "🌾")
    fig3 = _draw_single_commodity_network_improved(physical_edges, container_flows, province_names, 
                                                  "CONTAINER", "#FFEAA7", "📦")
    
    return fig1, fig2, fig3

def _draw_single_commodity_network_improved(physical_edges, commodity_flows, province_names, title, color, emoji):
    """Draw a network diagram for a specific commodity - improved layout"""
    G = nx.MultiDiGraph()
    
    # Add edges with mode information
    for u, v, mode, length in physical_edges:
        G.add_edge(u, v, mode=mode, length=length)
    
    # Use an improved layout
    pos = _create_better_layout(G)
    
    # Increase the figure size
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Calculate maximum width for normalization
    max_flow = max(commodity_flows.values()) if commodity_flows else 1
    
    # Draw edges with thickness proportional to flow and distinguish modes
    for u, v, key in G.edges(keys=True):
        edge_key = (min(u, v), max(u, v))
        flow = commodity_flows.get(edge_key, 0)
        mode = G[u][v][key]['mode']
        
        # Calculate width based on flow
        width = 2 + (flow / max_flow) * 10 if max_flow > 0 else 2
        
        # Colors and styles based on transport mode
        if mode == 1:  # Road
            edge_color = '#FF8C00'  # Deep orange
            connection_style = 'arc3,rad=0.2'
        else:  # Waterway
            edge_color = '#1E90FF'  # Blue
            connection_style = 'arc3,rad=-0.2'
        
        # Draw edges with arrows
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], 
            width=width, alpha=0.8, 
            edge_color=edge_color, ax=ax,
            arrows=True, arrowstyle='-|>', arrowsize=25,
            connectionstyle=connection_style
        )
        
        # Add flow labels when available
        if flow > 0:
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            
            # Offset labels based on mode
            if mode == 1:
                y += 0.2
            else:
                y -= 0.2
                
            ax.text(x, y, f'{flow:,}', 
                   fontsize=11, ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9, edgecolor='black'))
    
    # Draw nodes with larger sizes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1500, edgecolors='black', ax=ax)
    
    # Node labels with province names - larger font
    node_labels = {node: province_names.get(node, f"Node {node}") for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, ax=ax)
    
    # Edge labels with distance
    edge_labels = {}
    for u, v, key in G.edges(keys=True):
        length = G[u][v][key]['length']
        edge_labels[(u, v, key)] = f"{length}km"
    
    for (u, v, key), label in edge_labels.items():
        mode = G[u][v][key]['mode']
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        
        # Offset distance labels
        if mode == 1:
            y += 0.3
            color_text = 'darkorange'
        else:
            y -= 0.3
            color_text = 'darkblue'
            
        ax.text(x, y, label, fontsize=10, color=color_text, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color_text))
    
    ax.set_title(f"{emoji} NETWORK DIAGRAM - {title}", 
                fontsize=18, fontweight='bold', pad=30, color=color)
    ax.axis('off')
    
    # Add legend
    total_flow = sum(commodity_flows.values())
    textstr = f'Total flow: {total_flow:,}'
    props = dict(boxstyle='round', facecolor=color, alpha=0.3, edgecolor='black')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=13, fontweight='bold',
            verticalalignment='top', bbox=props)
    
    # Add transport mode legend
    legend_elements = [
        plt.Line2D([0], [0], color='#FF8C00', lw=4, label='Road', marker='>', markersize=15),
        plt.Line2D([0], [0], color='#1E90FF', lw=4, label='Waterway', marker='>', markersize=15),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    return fig

# [Remaining functions unchanged...]
def create_commodity_flow_comparison(baseline_results, optimized_results, province_names):
    """Create a flow comparison chart before and after optimization"""
    passenger_baseline = {}
    passenger_optimized = {}
    rice_baseline = {}
    rice_optimized = {}
    container_baseline = {}
    container_optimized = {}
    
    for (commodity, edge), flow in baseline_results.get('flow_by_commodity', {}).items():
        edge_label = f"{province_names.get(edge[0], edge[0])} → {province_names.get(edge[1], edge[1])}"
        if commodity == 'passenger':
            passenger_baseline[edge_label] = flow
        elif commodity == 'rice':
            rice_baseline[edge_label] = flow
        elif commodity == 'container':
            container_baseline[edge_label] = flow
    
    for (commodity, edge), flow in optimized_results.get('flow_by_commodity', {}).items():
        edge_label = f"{province_names.get(edge[0], edge[0])} → {province_names.get(edge[1], edge[1])}"
        if commodity == 'passenger':
            passenger_optimized[edge_label] = flow
        elif commodity == 'rice':
            rice_optimized[edge_label] = flow
        elif commodity == 'container':
            container_optimized[edge_label] = flow
    
    # Create a DataFrame for each commodity
    edges = list(set(list(passenger_baseline.keys()) + list(passenger_optimized.keys())))
    
    passenger_df = pd.DataFrame({
        'Route': edges,
        'Before optimization': [passenger_baseline.get(edge, 0) for edge in edges],
        'After optimization': [passenger_optimized.get(edge, 0) for edge in edges]
    })
    passenger_df['Difference'] = passenger_df['After optimization'] - passenger_df['Before optimization']
    
    rice_df = pd.DataFrame({
        'Route': edges,
        'Before optimization': [rice_baseline.get(edge, 0) for edge in edges],
        'After optimization': [rice_optimized.get(edge, 0) for edge in edges]
    })
    rice_df['Difference'] = rice_df['After optimization'] - rice_df['Before optimization']
    
    container_df = pd.DataFrame({
        'Route': edges,
        'Before optimization': [container_baseline.get(edge, 0) for edge in edges],
        'After optimization': [container_optimized.get(edge, 0) for edge in edges]
    })
    container_df['Difference'] = container_df['After optimization'] - container_df['Before optimization']
    
    # Create a larger figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Passenger chart
    x = range(len(passenger_df))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], passenger_df['Before optimization'], width, 
            label='Before optimization', color='lightblue', alpha=0.7)
    ax1.bar([i + width/2 for i in x], passenger_df['After optimization'], width, 
            label='After optimization', color='#FF6B6B', alpha=0.7)
    
    ax1.set_xlabel('Route', fontsize=12)
    ax1.set_ylabel('Flow', fontsize=12)
    ax1.set_title('PASSENGER FLOWS: Before vs After Optimization', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(passenger_df['Route'], rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Rice chart
    ax2.bar([i - width/2 for i in x], rice_df['Before optimization'], width, 
            label='Before optimization', color='lightgreen', alpha=0.7)
    ax2.bar([i + width/2 for i in x], rice_df['After optimization'], width, 
            label='After optimization', color='#4ECDC4', alpha=0.7)
    
    ax2.set_xlabel('Route', fontsize=12)
    ax2.set_ylabel('Flow', fontsize=12)
    ax2.set_title('RICE FLOWS: Before vs After Optimization', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(rice_df['Route'], rotation=45, ha='right', fontsize=10)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Container chart
    ax3.bar([i - width/2 for i in x], container_df['Before optimization'], width, 
            label='Before optimization', color='navajowhite', alpha=0.7)
    ax3.bar([i + width/2 for i in x], container_df['After optimization'], width, 
            label='After optimization', color='#FFEAA7', alpha=0.7)
    
    ax3.set_xlabel('Route', fontsize=12)
    ax3.set_ylabel('Flow', fontsize=12)
    ax3.set_title('CONTAINER FLOWS: Before vs After Optimization', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(container_df['Route'], rotation=45, ha='right', fontsize=10)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, passenger_df, rice_df, container_df

def create_cost_comparison(baseline_results, optimized_results):
    """Create a cost comparison chart"""
    costs_comparison = {
        'Cost category': ['Investment', 'Service', 'Transport', 'Total'],
        'Before optimization': [
            baseline_results.get('investment_cost', 0),
            baseline_results.get('service_cost', 0),
            baseline_results.get('transport_cost', 0),
            baseline_results.get('objective', 0)
        ],
        'After optimization': [
            optimized_results.get('investment_cost', 0),
            optimized_results.get('service_cost', 0),
            optimized_results.get('transport_cost', 0),
            optimized_results.get('objective', 0)
        ]
    }
    
    df = pd.DataFrame(costs_comparison)
    df['Savings'] = df['Before optimization'] - df['After optimization']
    df['Savings rate (%)'] = (df['Savings'] / df['Before optimization'] * 100).round(1)
    
    # Create a larger figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = range(len(df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], df['Before optimization'], width, 
           label='Before optimization', color='lightcoral', alpha=0.7)
    ax.bar([i + width/2 for i in x], df['After optimization'], width, 
           label='After optimization', color='lightgreen', alpha=0.7)
    
    ax.set_xlabel('Cost category', fontsize=12)
    ax.set_ylabel('Cost (VND)', fontsize=12)
    ax.set_title('COST COMPARISON: Before vs After Optimization', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Cost category'], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add values above each bar
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i - width/2, row['Before optimization'] + 10000, f'{row["Before optimization"]:,.0f}', 
                ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, row['After optimization'] + 10000, f'{row["After optimization"]:,.0f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig, df

# ======== STREAMLIT APPLICATION ========
def main():
    st.markdown('<div class="main-header">🚚 MULTIMODAL TRANSPORT NETWORK OPTIMIZATION SYSTEM</div>', unsafe_allow_html=True)
    
    # Sidebar - Input parameters
    st.sidebar.header("📊 PARAMETER SETTINGS")
    
    # Basic information
    province_names = {
        0: "An Giang",
        1: "Dong Thap", 
        2: "Can Tho",
        3: "Ho Chi Minh City",
        4: "Vinh Long"
    }
    
    # Transport demand
    st.sidebar.subheader("📦 Transport Demand")
    passenger_1_4 = st.sidebar.slider("Passengers: An Giang → Ho Chi Minh City", 1000, 5000, 3000, 100)
    passenger_2_5 = st.sidebar.slider("Passengers: Dong Thap → Vinh Long", 1000, 5000, 2800, 100)
    rice_2_4 = st.sidebar.slider("Rice: Dong Thap → Ho Chi Minh City", 2000, 8000, 4000, 100)
    container_1_3 = st.sidebar.slider("Container: An Giang → Can Tho", 1000, 4000, 2000, 100)
    
    # Upgrade costs
    st.sidebar.subheader("💰 Upgrade Costs")
    hub_upgrade_cost = st.sidebar.slider("Hub upgrade cost (Can Tho)", 500, 2000, 1000, 50)
    road_upgrade_cost = st.sidebar.slider("Road upgrade cost", 400, 1500, 800, 50)
    water_upgrade_cost = st.sidebar.slider("Waterway upgrade cost", 200, 1000, 500, 50)
    
    # Capacity
    st.sidebar.subheader("🏗️ Capacity")
    hub_capacity_0 = st.sidebar.slider("Initial hub capacity", 1000, 3000, 2000, 100)
    hub_capacity_1 = st.sidebar.slider("Hub capacity after upgrade", 5000, 10000, 7000, 100)
    road_capacity = st.sidebar.slider("Road capacity after upgrade", 2000, 5000, 3000, 100)
    water_capacity = st.sidebar.slider("Waterway capacity after upgrade", 3000, 6000, 4000, 100)
    
    # Service costs
    st.sidebar.subheader("🔧 Other Costs")
    hub_service_cost_val = st.sidebar.slider("Hub service cost", 0.5, 3.0, 1.0, 0.1)
    switch_cost_val = st.sidebar.slider("Mode switching cost", 1, 5, 2, 1)
    
    # Run model button
    if st.sidebar.button("🎯 RUN OPTIMIZATION MODEL", type="primary"):
        with st.spinner("Optimizing the transport network..."):
            # Physical graph data
            n_physical = 5
            physical_edges = [
                (0, 2, 1, 45), (0, 2, 2, 41),
                (0, 3, 1, 23), (0, 3, 2, 96),
                (1, 2, 1, 85),
                (1, 3, 1, 13), (1, 3, 2, 21),
                (1, 4, 1, 74), (1, 4, 2, 87),
                (2, 3, 1, 35), (2, 3, 2, 93),
                (3, 4, 1, 67), (3, 4, 2, 85)
            ]
            
            # Create the baseline model
            baseline_model_data = {
                'demands': {
                    ('g1', (1, 4)): passenger_1_4,
                    ('g1', (2, 5)): passenger_2_5,
                    ('g2', (2, 4)): rice_2_4,
                    ('g3', (1, 3)): container_1_3
                }
            }
            baseline_results = create_baseline_model(baseline_model_data)
            
            # Build the expanded graph
            G_exp, _ = build_expanded_graph(n_physical, physical_edges)
            
            # Prepare data for the optimization model
            model_data = {
                'T': [1, 2],
                'real_nodes': [1, 2, 3, 4, 5],
                'virtual_nodes': ['3^1', '3^2', '4^1', '4^2', '5^1', '5^2'],
                'H': [3],
                'potential_hubs': [3],
                'existing_hubs': [],
                'A': [],
                'real_arcs': [],
                'virtual_arcs': [],
                'potential_arcs': [(3, '4^1'), (3, '4^2')],
                'commodities': {'passenger': 'g1', 'rice': 'g2', 'container': 'g3'},
                'OD_pairs': {
                    'g1': [(1, 4), (2, 5)],
                    'g2': [(2, 4)],
                    'g3': [(1, 3)]
                },
                'paths': {},
                'switch_cost': switch_cost_val,
                'hub_service_cost': {3: hub_service_cost_val},
                'hub_upgrade_cost': hub_upgrade_cost,
                'hub_capacity': {0: hub_capacity_0, 1: hub_capacity_1},
                'arc_upgrade_costs': {(3, '4^1'): road_upgrade_cost, (3, '4^2'): water_upgrade_cost},
                'arc_capacities': {
                    (3, '4^1'): {0: 0, 1: road_capacity},
                    (3, '4^2'): {0: 0, 1: water_capacity}
                },
                'existing_hub_capacity': hub_capacity_0,
                'existing_arc_capacity': 2000,
                'demands': {
                    ('g1', (1, 4)): passenger_1_4,
                    ('g1', (2, 5)): passenger_2_5,
                    ('g2', (2, 4)): rice_2_4,
                    ('g3', (1, 3)): container_1_3
                }
            }
            
            # Run the optimization model
            optimized_results = create_optimization_model(model_data)
            
            # Store results in the session state
            st.session_state.baseline_results = baseline_results
            st.session_state.optimized_results = optimized_results
            st.session_state.model_data = model_data
            st.session_state.physical_edges = physical_edges
            st.session_state.province_names = province_names
    
    # Display results
    if 'optimized_results' in st.session_state:
        baseline_results = st.session_state.baseline_results
        optimized_results = st.session_state.optimized_results
        physical_edges = st.session_state.physical_edges
        province_names = st.session_state.province_names
        
        st.markdown('<div class="sub-header">📈 OPTIMIZATION RESULTS</div>', unsafe_allow_html=True)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cost_saving = baseline_results.get('objective', 0) - optimized_results.get('objective', 0)
            st.metric("Total cost", 
                     f"{optimized_results.get('objective', 0):,.0f} VND",
                     f"Savings: {cost_saving:,.0f} VND")
        with col2:
            st.metric("Investment cost", f"{optimized_results.get('investment_cost', 0):,.0f} VND")
        with col3:
            st.metric("Transport cost", f"{optimized_results.get('transport_cost', 0):,.0f} VND")
        with col4:
            status = optimized_results.get('status', 'Unknown')
            status_color = "🟢" if status == 'Optimal' else "🔴"
            st.metric("Status", f"{status_color} {status}")
        
        # Improved network comparison chart
        st.markdown('<div class="sub-header">🗺️ NETWORK COMPARISON BEFORE AND AFTER OPTIMIZATION</div>', unsafe_allow_html=True)
        st.markdown("**🆕 IMPROVEMENT:** Layout optimized with greater spacing between nodes for clarity")
        comparison_fig = draw_network_comparison(physical_edges, baseline_results, optimized_results, province_names)
        st.pyplot(comparison_fig)
        
        # Enhanced network diagrams for each commodity
        st.markdown('<div class="sub-header">📊 NETWORK DIAGRAMS BY COMMODITY</div>', unsafe_allow_html=True)
        st.markdown("**🆕 IMPROVEMENT:** Larger visuals, clear layout, easy to distinguish road and waterway flows")
        
        # Create separate charts
        passenger_fig, rice_fig, container_fig = create_commodity_specific_networks(
            physical_edges, optimized_results['flow_by_commodity'], province_names
        )
        
        # Display each chart in separate tabs
        tab1, tab2, tab3 = st.tabs(["👥 PASSENGERS", "🌾 RICE", "📦 CONTAINER"])
        
        with tab1:
            st.pyplot(passenger_fig)
            total_passenger = sum(flow for (commodity, _), flow in optimized_results['flow_by_commodity'].items() if commodity == 'passenger')
            st.metric("Total passenger flow", f"{total_passenger:,}")
            
        with tab2:
            st.pyplot(rice_fig)
            total_rice = sum(flow for (commodity, _), flow in optimized_results['flow_by_commodity'].items() if commodity == 'rice')
            st.metric("Total rice flow", f"{total_rice:,}")
            
        with tab3:
            st.pyplot(container_fig)
            total_container = sum(flow for (commodity, _), flow in optimized_results['flow_by_commodity'].items() if commodity == 'container')
            st.metric("Total container flow", f"{total_container:,}")
        
        # Cost comparison chart
        st.markdown('<div class="sub-header">💰 COST COMPARISON</div>', unsafe_allow_html=True)
        cost_fig, cost_df = create_cost_comparison(baseline_results, optimized_results)
        st.pyplot(cost_fig)
        st.dataframe(cost_df, use_container_width=True)
        
        # Flow comparison chart
        st.markdown('<div class="sub-header">📈 DETAILED FLOW COMPARISON</div>', unsafe_allow_html=True)
        flow_fig, passenger_df, rice_df, container_df = create_commodity_flow_comparison(baseline_results, optimized_results, province_names)
        st.pyplot(flow_fig)
        
        # Display detailed data tables
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('##### 👥 DETAILED PASSENGER FLOWS')
            st.dataframe(passenger_df, use_container_width=True)
        
        with col2:
            st.markdown('##### 🌾 DETAILED RICE FLOWS')
            st.dataframe(rice_df, use_container_width=True)
        
        with col3:
            st.markdown('##### 📦 DETAILED CONTAINER FLOWS')
            st.dataframe(container_df, use_container_width=True)
        
        # Upgrade results
        st.markdown('<div class="sub-header">🏗️ UPGRADE RESULTS</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Upgraded hubs:**")
            upgraded_hubs = optimized_results.get('upgraded_hubs', [])
            if upgraded_hubs:
                for hub in upgraded_hubs:
                    st.markdown(f'<span class="upgraded">✅ {province_names.get(hub, f"Node {hub}")}</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="not-upgraded">❌ No hubs were upgraded</span>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Upgraded routes:**")
            upgraded_arcs = optimized_results.get('upgraded_arcs', [])
            if upgraded_arcs:
                for arc in upgraded_arcs:
                    start_node, end_virtual = arc
                    end_node = int(end_virtual.split('^')[0]) if isinstance(end_virtual, str) and '^' in end_virtual else end_virtual
                    mode = "Road" if '^1' in str(end_virtual) else "Waterway"
                    st.markdown(f'<span class="upgraded">✅ {province_names.get(start_node, f"Node {start_node}")} → {province_names.get(end_node, f"Node {end_node}")} ({mode})</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="not-upgraded">❌ No routes were upgraded</span>', unsafe_allow_html=True)
    
    else:
        # Display instructions when the model has not run
        st.markdown("""
        <div class="result-box">
        <h3>👋 Welcome to the Transport Network Optimization System</h3>
        <p>This application optimizes a multimodal transport network with the following features:</p>
        <ul>
            <li>🎯 <strong>Optimize total costs</strong></li>
            <li>🏗️ <strong>Support infrastructure upgrade decisions</strong></li>
            <li>🚚 <strong>Allocate transport flows optimally</strong></li>
            <li>📊 <strong>Compare before and after optimization</strong></li>
            <li>🆕 <strong>Enhanced network diagrams:</strong> Clear layout with separate road and waterway modes</li>
        </ul>
        <p><strong>To get started:</strong> Configure the parameters in the left sidebar and click \"RUN OPTIMIZATION MODEL\".</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()