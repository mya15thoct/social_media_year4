# Figure 1.1: Fraud as Supply Chain Operational Risk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_figure_1_1():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define positions
    boxes = [
        {"pos": (1, 4), "width": 1.5, "height": 0.8, "text": "Fraud\nTransactions", "color": "#ff6b6b"},
        {"pos": (3.5, 4), "width": 1.5, "height": 0.8, "text": "Information\nDistortion", "color": "#feca57"},
        {"pos": (6, 4), "width": 1.5, "height": 0.8, "text": "Bullwhip\nEffect", "color": "#48dbfb"},
        {"pos": (8.5, 4), "width": 1.5, "height": 0.8, "text": "Supply Chain\nPerformance ↓", "color": "#ff9ff3"},
    ]
    
    # Draw boxes
    for box in boxes:
        fancy_box = FancyBboxPatch(
            box["pos"], box["width"], box["height"],
            boxstyle="round,pad=0.1", 
            edgecolor='black', 
            facecolor=box["color"],
            linewidth=2
        )
        ax.add_patch(fancy_box)
        ax.text(
            box["pos"][0] + box["width"]/2, 
            box["pos"][1] + box["height"]/2,
            box["text"],
            ha='center', va='center',
            fontsize=11, fontweight='bold'
        )
    
    # Draw arrows
    arrows = [
        (2.5, 4.4, 3.5, 4.4),
        (5.0, 4.4, 6.0, 4.4),
        (7.5, 4.4, 8.5, 4.4),
    ]
    
    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', 
            mutation_scale=30,
            linewidth=3,
            color='black'
        )
        ax.add_patch(arrow)
    
    # Add bottom explanation boxes
    explanations = [
        {"pos": (1, 2), "width": 1.5, "height": 1.2, "text": "• False demand signals\n• Lost revenue\n• Inventory losses"},
        {"pos": (3.5, 2), "width": 1.5, "height": 1.2, "text": "• Forecast errors\n• Over-ordering\n• Wrong allocation"},
        {"pos": (6, 2), "width": 1.5, "height": 1.2, "text": "• Demand amplification\n• Excess inventory\n• Cost increases"},
        {"pos": (8.5, 2), "width": 1.5, "height": 1.2, "text": "• Coordination loss\n• Trust erosion\n• Efficiency decline"},
    ]
    
    for exp in explanations:
        fancy_box = FancyBboxPatch(
            exp["pos"], exp["width"], exp["height"],
            boxstyle="round,pad=0.05",
            edgecolor='gray',
            facecolor='white',
            linewidth=1,
            linestyle='--'
        )
        ax.add_patch(fancy_box)
        ax.text(
            exp["pos"][0] + exp["width"]/2,
            exp["pos"][1] + exp["height"]/2,
            exp["text"],
            ha='center', va='center',
            fontsize=9
        )
    
    # Add title
    ax.text(5, 5.5, 'Figure 1.1: Fraud as Supply Chain Operational Risk', 
            ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Figure 1.2: Research Framework
def create_figure_1_2():
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input layer (Left side, vertically centered)
    input_boxes = [
        {"pos": (0.5, 6), "width": 2.2, "height": 1.2, "text": "Transaction\nFeatures\n(57 features)", "color": "#a8e6cf"},
        {"pos": (0.5, 4), "width": 2.2, "height": 1.2, "text": "Network\nFeatures\n(4 features)", "color": "#ffd3b6"},
    ]
    
    for box in input_boxes:
        fancy_box = FancyBboxPatch(
            box["pos"], box["width"], box["height"],
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=box["color"],
            linewidth=2.5
        )
        ax.add_patch(fancy_box)
        ax.text(
            box["pos"][0] + box["width"]/2,
            box["pos"][1] + box["height"]/2,
            box["text"],
            ha='center', va='center',
            fontsize=11, fontweight='bold'
        )
    
    # Feature engineering (Middle-left, vertically centered)
    feature_box = {"pos": (4, 4.75), "width": 2.8, "height": 2, "text": "Feature\nEngineering\n& PCA\n(45 components)", "color": "#ffaaa5"}
    fancy_box = FancyBboxPatch(
        feature_box["pos"], feature_box["width"], feature_box["height"],
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=feature_box["color"],
        linewidth=2.5
    )
    ax.add_patch(fancy_box)
    ax.text(
        feature_box["pos"][0] + feature_box["width"]/2,
        feature_box["pos"][1] + feature_box["height"]/2,
        feature_box["text"],
        ha='center', va='center',
        fontsize=11, fontweight='bold'
    )
    
    # Deep Learning (Middle-right, vertically centered)
    dl_box = {"pos": (8, 4.75), "width": 2.8, "height": 2, "text": "Deep Neural\nNetwork\nEnsemble\n(3 models)", "color": "#b8b8d9"}
    fancy_box = FancyBboxPatch(
        dl_box["pos"], dl_box["width"], dl_box["height"],
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=dl_box["color"],
        linewidth=2.5
    )
    ax.add_patch(fancy_box)
    ax.text(
        dl_box["pos"][0] + dl_box["width"]/2,
        dl_box["pos"][1] + dl_box["height"]/2,
        dl_box["text"],
        ha='center', va='center',
        fontsize=11, fontweight='bold'
    )
    
    # Fraud Detection (Right side, vertically centered)
    detection_box = {"pos": (12, 4.75), "width": 2.8, "height": 2, "text": "Fraud\nDetection\n(74.83%\nRecall)", "color": "#ffef96"}
    fancy_box = FancyBboxPatch(
        detection_box["pos"], detection_box["width"], detection_box["height"],
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=detection_box["color"],
        linewidth=2.5
    )
    ax.add_patch(fancy_box)
    ax.text(
        detection_box["pos"][0] + detection_box["width"]/2,
        detection_box["pos"][1] + detection_box["height"]/2,
        detection_box["text"],
        ha='center', va='center',
        fontsize=11, fontweight='bold'
    )
    
    # Risk Reduction (Middle bottom, larger)
    risk_box = {"pos": (5.5, 2.5), "width": 5, "height": 1.3, "text": "Risk Reduction\nStrategy\n(Chapter 5)", "color": "#e6f7ff"}
    fancy_box = FancyBboxPatch(
        risk_box["pos"], risk_box["width"], risk_box["height"],
        boxstyle="round,pad=0.1",
        edgecolor='blue',
        facecolor=risk_box["color"],
        linewidth=2.5,
        linestyle='--'
    )
    ax.add_patch(fancy_box)
    ax.text(
        risk_box["pos"][0] + risk_box["width"]/2,
        risk_box["pos"][1] + risk_box["height"]/2,
        risk_box["text"],
        ha='center', va='center',
        fontsize=11, fontweight='bold',
        color='blue'
    )
    
    # Supply Chain Outcomes (Bottom row, evenly spaced)
    outcome_boxes = [
        {"pos": (0.8, 0.3), "width": 3.5, "height": 1.2, "text": "Bullwhip Effect\nMitigation\n(Chapter 7)", "color": "#fff4e6"},
        {"pos": (6.25, 0.3), "width": 3.5, "height": 1.2, "text": "Information\nQuality\nImprovement", "color": "#fff4e6"},
        {"pos": (11.7, 0.3), "width": 3.5, "height": 1.2, "text": "Business Value\n$88,900\nNet Benefit", "color": "#fff4e6"},
    ]
    
    for box in outcome_boxes:
        fancy_box = FancyBboxPatch(
            box["pos"], box["width"], box["height"],
            boxstyle="round,pad=0.1",
            edgecolor='green',
            facecolor=box["color"],
            linewidth=2.5,
            linestyle='--'
        )
        ax.add_patch(fancy_box)
        ax.text(
            box["pos"][0] + box["width"]/2,
            box["pos"][1] + box["height"]/2,
            box["text"],
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color='green'
        )
    
    # Draw arrows (horizontal flow)
    arrows = [
        # From Transaction Features to Feature Engineering
        (2.7, 6.6, 4, 5.8),
        # From Network Features to Feature Engineering
        (2.7, 4.6, 4, 5.7),
        # From Feature Engineering to DNN
        (6.8, 5.75, 8, 5.75),
        # From DNN to Fraud Detection
        (10.8, 5.75, 12, 5.75),
        # From Fraud Detection down to Risk Reduction
        (13.4, 4.75, 13.4, 3.8),
        (13.4, 3.8, 10.5, 3.1),
        # From Risk Reduction to outcomes
        (8, 2.5, 5.5, 1.5),
        (8, 3.1, 8, 1.5),
        (8, 3.1, 10.5, 1.5),
    ]
    
    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->',
            mutation_scale=25,
            linewidth=2.5,
            color='black'
        )
        ax.add_patch(arrow)
    
    # Add title
    ax.text(8, 9, 'Figure 1.2: Research Framework', 
            ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Usage:
if __name__ == '__main__':
    fig1 = create_figure_1_1()
    fig1.savefig('figure_1_1.png', dpi=300, bbox_inches='tight')
    print("Saved: figure_1_1.png")
    
    fig2 = create_figure_1_2()
    fig2.savefig('figure_1_2.png', dpi=300, bbox_inches='tight')
    print("Saved: figure_1_2.png")
    
    plt.show()  # Display figures