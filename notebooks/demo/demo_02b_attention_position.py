# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "plotly",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Attention & encodage de position")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "figure.dpi": 64,
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "font.size": 10,
        }
    )
    return np, plt


@app.cell
def _():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


@app.cell
def _(mo):
    use_gpu = mo.ui.switch(value=True, label="Calcul sur GPU (CUDA via torch)")
    use_gpu
    return (use_gpu,)


@app.function
def make_backend(use_cuda):
    """Import paresseux : torch n'est chargé que si on demande le calcul GPU."""
    import torch

    if use_cuda and torch.cuda.is_available():
        return torch, torch.device("cuda"), torch.cuda.get_device_name(0)
    return torch, torch.device("cpu"), "CPU (torch)"


@app.cell
def _(use_gpu):
    torch, device, device_label = make_backend(use_gpu.value)
    return device, device_label, torch


@app.function
def to_numpy(t):
    return t.detach().to("cpu").numpy()


@app.cell
def _(device, np, torch):
    def positional_encoding(n_positions, d_model, base=10_000.0):
        """Encodage sinusoïdal du Transformer original.

        Renvoie un tenseur (n_positions, d_model) sur le `device` courant :
        les premières dimensions oscillent très vite, les dernières très lentement.
        """
        freq = base ** (-2 * (np.arange(d_model) // 2) / d_model)
        omega = torch.as_tensor(freq, dtype=torch.float32, device=device).unsqueeze(0)
        pos = torch.arange(n_positions, dtype=torch.float32, device=device).unsqueeze(1)
        angles = pos * omega
        pairs = torch.arange(d_model, device=device) % 2
        return torch.where(pairs == 0, torch.sin(angles), torch.cos(angles))

    return (positional_encoding,)


@app.cell
def _(torch):
    def attention_matrix(emb, pe, alpha=0.0, temperature=0.4, mask_self=False):
        """Poids d'attention, chaque ligne somme à 1.

        score(i, j) = contenu(i)·contenu(j) + alpha * position(i)·position(j)
        """
        scores = emb @ emb.T
        if alpha:
            scores = scores + alpha * (pe @ pe.T) / (pe.shape[1] / 2.0)
        scores = scores / temperature
        if mask_self:
            scores = scores.clone()
            scores.fill_diagonal_(float("-inf"))
        scores = scores - scores.max(dim=1, keepdim=True).values
        weights = torch.exp(scores)
        return weights / weights.sum(dim=1, keepdim=True)

    return (attention_matrix,)


@app.cell
def _(device_label, mo, use_gpu):
    etat = (
        f"**Calcul :** `{device_label}`  ·  **Animations :** WebGL (Plotly), pilotées par "
        "curseur — aucune image de frame n'est embarquée dans le notebook."
    )
    if use_gpu.value:
        etat += (
            "\n\n> ⚠️ Importer `torch` coûte environ **0,6 Go de RAM** (contexte CUDA compris). "
            "Pour des tenseurs de quelques kilo-octets, le GPU ne fait donc **pas** gagner de "
            "mémoire : le gain vient de la suppression des frames d'animation. "
            "Coupe l'interrupteur au-dessus si la RAM est tendue — le calcul passera sur le CPU."
        )
    mo.md(etat)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Attention & position : proche ou éloigné ?

    Un Transformer ne « voit » pas l'ordre des mots : il reçoit une **suite de vecteurs**,
    sans notion de gauche ni de droite. Pour lui rendre l'ordre, on **ajoute à chaque
    token une signature de position** :

    $$X_p = \text{Embedding}(token_p) + PE(p)$$

    > **Le modèle mental**
    >
    > *Quoi ?* (le token) **+** *Où ?* (sa position) → projections Q, K, V →
    > scores d'attention → pondération des informations entre tokens.
    >
    > La position n'impose **aucune pénalité de distance** : elle *donne l'information*
    > que l'attention apprend ensuite à exploiter.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Sans signature de position, un mot est le même partout

    Le mot « chat » reçoit toujours le même vecteur. Qu'il soit au début ou au milieu
    de la phrase, le modèle voit **deux fois exactement la même chose** : il est
    incapable de savoir lequel vient en premier.
    """)
    return


@app.cell
def _(np, plt, positional_encoding):
    rng = np.random.default_rng(7)
    dim_demo = 16
    chat_vec = rng.normal(0, 1, dim_demo)
    pe_demo = to_numpy(positional_encoding(24, dim_demo))
    p_proche, p_loin = 2, 13
    idx = np.arange(dim_demo)

    fig1, axes1 = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    axes1[0].bar(idx - 0.2, chat_vec, 0.4, color="#1f77b4", label="« chat » en position 2")
    axes1[0].bar(idx + 0.2, chat_vec, 0.4, color="#ff7f0e", label="« chat » en position 13")
    axes1[0].set_title("Sans position\nles deux vecteurs sont identiques")

    axes1[1].bar(idx - 0.2, chat_vec + pe_demo[p_proche], 0.4, color="#1f77b4", label="« chat » en position 2")
    axes1[1].bar(idx + 0.2, chat_vec + pe_demo[p_loin], 0.4, color="#ff7f0e", label="« chat » en position 13")
    axes1[1].set_title("Avec la signature de position\nles deux vecteurs diffèrent")

    for ax_demo in axes1:
        ax_demo.axhline(0, color="0.75", lw=0.8)
        ax_demo.set_xlabel("dimension du vecteur")
        ax_demo.legend(frameon=False, fontsize=8, loc="upper right")
    axes1[0].set_ylabel("valeur")
    fig1.tight_layout()
    fig1
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. La signature de position : plusieurs fréquences à la fois

    Rappel : **une position = un token**. On reprend la phrase d'exemple
    « Le chat qui dort sur le canapé mange une souris » : position 0 = « Le »,
    position 1 = « chat », …, position 9 = « souris ».

    En **bas** de la carte, les dimensions rapides changent à chaque token. En **haut**,
    les dimensions lentes paraissent figées sur une phrase courte : leur période se compte
    en centaines, voire en milliers de positions. Elles ne prennent le relais que sur des
    textes longs.

    Bouge le curseur **nombre de positions** : la carte du milieu s'allonge, et le panneau
    de droite montre de plus en plus de dimensions qui se « réveillent ».
    """)
    return


@app.cell
def _(mo):
    n_positions = mo.ui.slider(
        steps=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        value=512,
        label="nombre de positions",
        show_value=True,
    )
    d_model = mo.ui.slider(8, 128, value=64, step=8, label="dimensions", show_value=True)
    mo.hstack([n_positions, d_model], justify="start", gap=2)
    return d_model, n_positions


@app.cell
def _(d_model, n_positions, np, plt, positional_encoding):
    phrase_demo = ["Le", "chat", "qui", "dort", "sur", "le", "canapé", "mange", "une", "souris"]
    dims = d_model.value
    n_pos = n_positions.value

    pe_phrase = to_numpy(positional_encoding(len(phrase_demo), dims))
    pe_long_t = positional_encoding(n_pos, dims)
    pe_long = to_numpy(pe_long_t)
    variation = to_numpy(pe_long_t.std(dim=0))

    fig2, (ax_phrase, ax_long, ax_var) = plt.subplots(
        1, 3, figsize=(12.5, 4), sharey=True, gridspec_kw={"width_ratios": [1, 1.7, 0.9]}
    )

    ax_phrase.imshow(pe_phrase.T, aspect="auto", cmap="RdBu_r", origin="lower", vmin=-1, vmax=1)
    ax_phrase.set_xticks(range(len(phrase_demo)))
    ax_phrase.set_xticklabels(phrase_demo, rotation=60, ha="right", fontsize=8)
    ax_phrase.set_ylabel("dimension de la signature")
    ax_phrase.set_title("La phrase d'exemple\n(10 tokens)")

    im2 = ax_long.imshow(pe_long.T, aspect="auto", cmap="RdBu_r", origin="lower", vmin=-1, vmax=1)
    ax_long.set_xlabel("position du token")
    ax_long.set_title(f"Le même encodage\nsur {n_pos} positions")
    fig2.colorbar(im2, ax=ax_long, label="valeur", shrink=0.85)

    ax_var.barh(np.arange(dims), variation, height=1.0, color="#1f77b4")
    ax_var.set_ylim(-0.5, dims - 0.5)
    ax_var.set_xlabel("écart-type dans le texte")
    ax_var.set_title(f"Dimensions qui varient :\n{int((variation > 0.05).sum())} / {dims}")

    fig2.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Animation (WebGL) : la signature d'un token qui se déplace

    Le token avance d'une position à la fois. À **gauche** (dimensions rapides), tout
    change à chaque pas. À **droite** (dimensions lentes), presque rien ne bouge sur une
    courte distance.

    *Les cartes denses restent en raster (une seule image) : c'est plus léger qu'un
    graphique WebGL cellule par cellule. Les courbes, elles, passent par le GPU.*
    """)
    return


@app.cell
def _(mo):
    pos_curseur = mo.ui.slider(0, 63, value=11, step=1, label="position du token courant", show_value=True)
    pos_curseur
    return (pos_curseur,)


@app.cell
def _(go, make_subplots, mo, pos_curseur, positional_encoding):
    N1, D1, p_ref1 = 64, 64, 10
    pe1 = to_numpy(positional_encoding(N1, D1))
    q1 = pos_curseur.value

    fig_a1 = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.42, 0.58],
        subplot_titles=("La carte des signatures", "Signature courante vs référence"),
        horizontal_spacing=0.12,
    )
    fig_a1.add_trace(
        go.Heatmap(z=pe1.T, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, showscale=False),
        row=1,
        col=1,
    )
    fig_a1.add_vline(x=p_ref1, line_color="crimson", line_width=2, row=1, col=1)
    fig_a1.add_vline(x=q1, line_color="black", line_dash="dash", line_width=2, row=1, col=1)
    fig_a1.add_trace(
        go.Scattergl(
            x=list(range(D1)), y=pe1[p_ref1], mode="lines",
            name=f"position {p_ref1} (référence)", line=dict(color="crimson"),
        ),
        row=1, col=2,
    )
    fig_a1.add_trace(
        go.Scattergl(
            x=list(range(D1)), y=pe1[q1], mode="lines",
            name=f"position {q1}", line=dict(color="black", dash="dash"),
        ),
        row=1, col=2,
    )
    fig_a1.update_xaxes(title_text="position", row=1, col=1)
    fig_a1.update_yaxes(title_text="dimension", row=1, col=1)
    fig_a1.update_xaxes(title_text="dimension", row=1, col=2)
    fig_a1.update_yaxes(range=[-1.15, 1.15], row=1, col=2)
    fig_a1.update_layout(
        height=420, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2)
    )
    mo.ui.plotly(fig_a1)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Proche ou éloigné : la similarité dépend du décalage

    Pour chaque paire sinus/cosinus de fréquence $\omega$ :

    $$PE(p)\cdot PE(q) = \sum_i \cos\big(\omega_i\,(p-q)\big)$$

    La similarité ne dépend **que de la distance** $p-q$, pas de l'endroit absolu.

    ⚠️ Cela ne veut **pas** dire qu'elle diminue quand la distance augmente : elle oscille,
    puis décroît lentement.
    """)
    return


@app.cell
def _(mo):
    distance_curseur = mo.ui.slider(0, 69, value=12, step=1, label="distance entre les deux tokens (Δ)", show_value=True)
    distance_curseur
    return (distance_curseur,)


@app.cell
def _(distance_curseur, go, mo, np, positional_encoding):
    N2, D2, p_ref2 = 90, 64, 15
    pe2_t = positional_encoding(N2, D2)
    sim_all_t = pe2_t @ pe2_t[p_ref2]
    sim_all_t = sim_all_t / sim_all_t[p_ref2]
    fast2_t = pe2_t[:, :8]
    sim_fast_t = fast2_t @ fast2_t[p_ref2]
    sim_fast_t = sim_fast_t / sim_fast_t[p_ref2]

    deltas2 = np.arange(N2) - p_ref2
    sim_all = to_numpy(sim_all_t)
    sim_fast = to_numpy(sim_fast_t)
    d0 = distance_curseur.value
    q0 = d0 + p_ref2

    fig_a2 = go.Figure()
    fig_a2.add_trace(
        go.Scattergl(x=deltas2, y=sim_all, mode="lines", name="signature complète (64 dimensions)")
    )
    fig_a2.add_trace(
        go.Scattergl(x=deltas2, y=sim_fast, mode="lines", name="8 dimensions rapides")
    )
    fig_a2.add_trace(
        go.Scattergl(
            x=[d0], y=[sim_all[q0]], mode="markers",
            marker=dict(size=13, color="black"), name=f"Δ = {d0}",
        )
    )
    fig_a2.update_layout(
        height=380,
        xaxis_title="distance signée entre les deux tokens  (q − p)",
        yaxis_title="similarité normalisée",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    mo.ui.plotly(fig_a2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Deux choses à lire :

    1. **La courbe des dimensions rapides se répète.** Deux distances très différentes y
       donnent la **même** valeur : les tokens deviennent interchangeables (aliasing).
    2. **La courbe complète ne se répète pas** sur la fenêtre : les dimensions lentes
       lèvent l'ambiguïté. C'est le rôle de la « bande figée » de droite dans la partie 2.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. L'attention : qui regarde qui ?

    On garde la **même phrase d'exemple** : « Le chat qui dort sur le canapé mange une
    souris ». Les tokens se **regardent** entre eux : choisis un token et on affiche le
    poids qu'il donne à chacun des autres.

    - $\alpha = 0$ : l'attention ne dépend **que du sens** → un token éloigné mais
      pertinent (le « chat » pour « mange ») reçoit un fort poids, la distance n'existe pas.
    - $\alpha > 0$ : la position entre dans les scores → proximité et éloignement
      deviennent exploitables.
    """)
    return


@app.cell
def _(device, np, torch):
    words = ["Le", "chat", "qui", "dort", "sur", "le", "canapé", "mange", "une", "souris"]
    # 4 axes : [animal, lieu/objet, manger, dormir]
    emb_np = np.array(
        [
            [0.05, 0.05, 0.00, 0.00],  # Le
            [1.00, 0.00, 0.10, 0.00],  # chat
            [0.05, 0.05, 0.00, 0.00],  # qui
            [0.10, 0.60, 0.00, 0.90],  # dort
            [0.00, 0.30, 0.00, 0.00],  # sur
            [0.05, 0.05, 0.00, 0.00],  # le
            [0.00, 1.00, 0.00, 0.30],  # canapé
            [0.80, 0.10, 0.90, 0.00],  # mange
            [0.05, 0.05, 0.00, 0.00],  # une
            [0.90, 0.00, 0.15, 0.00],  # souris
        ],
        dtype=np.float32,
    )
    emb_t = torch.as_tensor(emb_np, device=device)
    return emb_t, words


@app.cell
def _(mo, words):
    alpha = mo.ui.slider(0.0, 4.0, value=0.0, step=0.25, label="poids de la position α  (0 = sans position)")
    query = mo.ui.dropdown(options=words, value="mange", label="token qui regarde")
    mask_self = mo.ui.switch(value=True, label="ignorer la diagonale (le token sur lui-même)")
    mo.vstack([mo.hstack([alpha, query], justify="start", gap=2), mask_self])
    return alpha, mask_self, query


@app.cell
def _(
    alpha,
    attention_matrix,
    emb_t,
    mask_self,
    plt,
    positional_encoding,
    query,
    words,
):
    pe_att = positional_encoding(len(words), 64)
    att = to_numpy(attention_matrix(emb_t, pe_att, alpha=alpha.value, mask_self=mask_self.value))
    qi = words.index(query.value)

    fig3, (ax_h, ax_bars) = plt.subplots(1, 2, figsize=(11, 4.2))
    im3 = ax_h.imshow(att, cmap="viridis", vmin=0, vmax=att.max())
    ax_h.set_xticks(range(len(words)))
    ax_h.set_xticklabels(words, rotation=60, ha="right", fontsize=8)
    ax_h.set_yticks(range(len(words)))
    ax_h.set_yticklabels(words, fontsize=8)
    ax_h.set_title("Ligne = celui qui regarde, colonne = celui qui est regardé")
    fig3.colorbar(im3, ax=ax_h, shrink=0.8, label="poids")

    couleurs = ["crimson" if i == qi else "#1f77b4" for i in range(len(words))]
    ax_bars.bar(range(len(words)), att[qi], color=couleurs)
    ax_bars.set_xticks(range(len(words)))
    ax_bars.set_xticklabels(words, rotation=60, ha="right", fontsize=8)
    ax_bars.set_title(f"Où regarde « {query.value} » ?")
    ax_bars.set_ylabel("poids d'attention")
    fig3.tight_layout()
    fig3
    return


@app.cell
def _(mo, words):
    token_regarde = mo.ui.slider(
        0, len(words) - 1, value=7, step=1, label="token qui regarde (animation)", show_value=True
    )
    token_regarde
    return (token_regarde,)


@app.cell
def _(
    alpha,
    attention_matrix,
    emb_t,
    go,
    mask_self,
    mo,
    positional_encoding,
    token_regarde,
    words,
):
    pe_att3 = positional_encoding(len(words), 64)
    att3 = to_numpy(attention_matrix(emb_t, pe_att3, alpha=alpha.value, mask_self=mask_self.value))
    i3 = token_regarde.value

    fig_a3 = go.Figure(
        go.Bar(
            x=words,
            y=att3[i3],
            marker_color=["crimson" if k == i3 else "#1f77b4" for k in range(len(words))],
        )
    )
    fig_a3.update_layout(
        height=360,
        title=f"« {words[i3]} » regarde les autres tokens",
        yaxis_title="poids d'attention",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    mo.ui.plotly(fig_a3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Pourquoi la position est indispensable : le sac de mots

    Deux phrases avec **exactement les mêmes mots**, dans un ordre différent. Sans position,
    le verbe « mange » pondère les mêmes vecteurs et produit la **même** représentation : le
    modèle ne peut pas dire qui mange qui.
    """)
    return


@app.cell
def _(device, torch):
    sent_a = ["le", "chat", "mange", "la", "souris"]
    sent_b = ["la", "souris", "mange", "le", "chat"]
    lexique = {
        "le": [0.05, 0.05],
        "la": [0.05, 0.05],
        "chat": [1.0, 0.0],
        "souris": [0.9, 0.1],
        "mange": [0.85, 0.15],
    }

    def vectorise(sentence):
        return torch.as_tensor([lexique[w] for w in sentence], dtype=torch.float32, device=device)

    return sent_a, sent_b, vectorise


@app.cell
def _(
    alpha,
    attention_matrix,
    np,
    plt,
    positional_encoding,
    sent_a,
    sent_b,
    vectorise,
):
    def sortie_du_verbe(sentence, alpha_val):
        vecteurs = vectorise(sentence)
        pe_s = positional_encoding(len(sentence), 32)
        att_s = attention_matrix(vecteurs, pe_s, alpha=alpha_val, mask_self=True)
        verbe = sentence.index("mange")
        return att_s[verbe] @ vecteurs

    a0 = to_numpy(sortie_du_verbe(sent_a, 0.0))
    b0 = to_numpy(sortie_du_verbe(sent_b, 0.0))
    aa = to_numpy(sortie_du_verbe(sent_a, alpha.value))
    ba = to_numpy(sortie_du_verbe(sent_b, alpha.value))
    titre_avec = (
        "α = 0 : bouge le curseur de la partie 4"
        if alpha.value == 0
        else f"α = {alpha.value:g} (avec position) : différentes"
    )

    fig5, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    x5 = np.arange(2)
    for ax_cmp, oa, ob, titre in [
        (ax_l, a0, b0, "α = 0 (sans position) : identiques"),
        (ax_r, aa, ba, titre_avec),
    ]:
        ax_cmp.bar(x5 - 0.18, oa, 0.36, color="#1f77b4", label="« le chat mange la souris »")
        ax_cmp.bar(x5 + 0.18, ob, 0.36, color="#ff7f0e", label="« la souris mange le chat »")
        ax_cmp.set_xticks(x5)
        ax_cmp.set_xticklabels(["axe A", "axe B"])
        ax_cmp.set_title(titre, fontsize=10)
    ax_l.set_ylabel("représentation du verbe « mange »")
    ax_r.legend(frameon=False, fontsize=8)
    fig5.tight_layout()
    fig5
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. À retenir

    | Étape | Ce qui se passe |
    | --- | --- |
    | Token | *Quoi ?* — le sens du mot |
    | + signature de position | *Où ?* — sa place, sur plusieurs fréquences |
    | Q, K, V | représentations utilisées pour l'attention |
    | Scores d'attention | pertinence du contenu **+** information de position |
    | Pondération | un token **éloigné** peut recevoir une forte attention |

    **La phrase à mémoriser :** l'encodage de position ne mesure pas directement
    l'importance d'un token. Il encode sa position de façon structurée pour que la
    self-attention puisse apprendre à exploiter l'ordre et les distances, **conjointement**
    avec la sémantique.

    La position n'ajoute **aucune pénalité de distance**. Elle fournit l'information ;
    l'attention décide.

    *Cette description correspond au PE sinusoïdal du Transformer original. Les modèles
    modernes (par ex. RoPE) intègrent la position autrement, mais l'idée reste la même.*
    """)
    return


if __name__ == "__main__":
    app.run()
