https://www.researchgate.net/publication/322778777_Thinking_Clearly_About_Correlations_and_Causation_Graphical_Causal_Models_for_Observational_Data

---
Control:
```mermaid
flowchart TD
    Confounder --> X
    Confounder --> Y
```

---
Don't control:
```mermaid
flowchart LR
    X --> mediator --> Y
```

---
[[collider bias]]
Don't control:
```mermaid
flowchart TD
    X --> Collider
    Y --> Collider
```

---

- Randomized experiments are gold standard but often not possible/feasible/ethical