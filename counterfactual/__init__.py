"""Counterfactual evaluation of the recovered causal graphs (docs/counterfactual_study.md).

  queries.py     query windows, interventions realised by re-solving the window's start state, simulation, rendering
  model_jepa.py  JEPA encoder / predictor on rendered windows (one step and autoregressive rollouts)
  scm.py         graph-restricted structural models (masked MLPs) fitted on true or model-read changes
  metrics.py     effects, agreement, episode bootstrap, the model's interventional graph
Entry point: python -m studies.counterfactual
"""
