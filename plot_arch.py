import argparse
from graphviz import Digraph

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_actions", type=int, default=2)
    p.add_argument("--out", type=str, default="plots/model_arch.png")
    p.add_argument("--policy_index", type=int, default=0, help="which option-policy head to show")
    args = p.parse_args()

    C_BG = "white"
    C_CARD = "#F8EDDA93"
    C_BORDER = "#CBD5E1E1"
    C_EDGE = "#94A3B8FF"
    C_TITLE = "#0F172A"

    C_FEATURES = "#F3D1EBFF"
    C_QHEAD    = "#8FC0E15B"
    C_TERM     = "#E47DB98A"
    C_POLICY   = "#FBFCCBB3"

    g = Digraph("OptionCriticFeatures", format="png")
    g.attr(rankdir="LR", bgcolor=C_BG, splines="spline", nodesep="0.15", ranksep="0.5", pad="0", dpi="600",size="6,6!")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="8",
           color=C_BORDER, fontcolor=C_TITLE, margin="0.08,0.04", height="0.3")
    g.attr("edge", color=C_EDGE, arrowsize="0.7", penwidth="0.8")
    # --- Input ---
    g.node("input", f"Input obs\\nshape: (B,in)", fillcolor=C_CARD)

    with g.subgraph(name="cluster_features") as c:
        c.attr(label="Feature Trunk", color=C_BORDER, fontcolor="#344762", style="rounded,dashed", fontsize="12", labeljust="c")
        c.attr("node", fillcolor=C_FEATURES)
        c.node("feat_lin1", f"Linear\\n(B,in) → (B,8)")
        c.node("feat_relu", "ReLU\\n(B,8) → (B,8)")
        c.node("feat_lin2", "Linear\\n(B,8) → (B,in)")
        c.attr(margin="4")
        c.edges([("feat_lin1", "feat_relu"), ("feat_relu", "feat_lin2")])

    g.edge("input", "feat_lin1")

    with g.subgraph(name="cluster_q") as c:
        c.attr(label="Option-Value Function", 
           color=C_BORDER, 
           fontcolor="#334155", 
           style="rounded,dashed", 
           fontsize="12")
        c.attr("node", fillcolor=C_QHEAD)
        c.node("q_head", f"Head\\n(B,in) → (B,|Ω|)")
        c.node("q_out",  f"Output Q\\n(B,|Ω|)", fillcolor=C_CARD)
        c.attr(margin="4")
    with g.subgraph(name="cluster_t") as c:
        c.attr(label="Termination Function", color=C_BORDER, fontcolor="#334155", style="rounded,dashed", fontsize="12")
        c.attr("node", fillcolor=C_TERM)
        c.node("t_head", f"Head\\n(B,in) → (B,|Ω|)")
        c.node("t_sig",  f"Sigmoid\\n(B,|Ω|) → (B,|Ω|)", fillcolor=C_CARD)
        c.node("t_out",  f"Output β\\n(B,|Ω|)", fillcolor=C_CARD)
        c.edges([("t_head", "t_sig"), ("t_sig", "t_out")])
        c.attr(margin="4")
    with g.subgraph(name="cluster_p") as c:
        c.attr(label=f"Intra-Option Policy", color=C_BORDER, fontcolor="#334155", style="rounded,dashed", fontsize="12")
        c.attr("node", fillcolor=C_POLICY)
        c.node("p_head", f"Head\\n(B,in) → (B,out)")
        c.node("p_out",  f"Output logits\\n(B,out)", fillcolor=C_CARD)
        c.attr(margin="4")
        
    # Connect features output to heads
    g.edge("feat_lin2", "q_head")
    g.edge("feat_lin2", "t_head")
    g.edge("feat_lin2", "p_head")

    g.edge("q_head", "q_out")
    g.edge("p_head", "p_out")
    
    out_path = args.out
    if out_path.lower().endswith(".png"):
        filename = out_path[:-4]
    else:
        filename = out_path
    g.render(filename=filename, cleanup=True)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
