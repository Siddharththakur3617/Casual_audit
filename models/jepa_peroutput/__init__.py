"""Graph-prior arm P: per-output parent masks (report route 2, CDL-style) on the cart-pole JEPA.

  graph.py  slot layout and the change-parent mask from checks/expert_graph.EXPERT
  model.py  PerOutputPredictor (one masked network per slot) and JEPAPerOutput
  train.py  training loop for this arm

Independent of the masked-attention and Jacobian arms in models/jepa/. Plan and caveats:
docs/graph_priors/peroutput_plan.md.
"""
