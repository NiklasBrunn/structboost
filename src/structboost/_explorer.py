"""Interactive HTML explorer for BAE results.

Generates a self-contained HTML file with Plotly.js for interactive
exploration of UMAP embeddings, spatial tissue plots, gene expression
overlays, encoder coefficient bar charts, and dimension annotations.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path


def _subsample_adata(adata, *, max_cells: int, seed: int):
    """Subsample adata if it exceeds max_cells.

    Parameters
    ----------
    adata
        AnnData object.
    max_cells
        Maximum number of cells. If n_obs <= max_cells, returns adata unchanged.
    seed
        Random seed for reproducible subsampling.

    Returns
    -------
    AnnData, possibly subsampled.
    """
    import numpy as np

    if adata.n_obs <= max_cells:
        return adata

    warnings.warn(
        f"Subsampling from {adata.n_obs} to {max_cells} cells for HTML explorer. "
        f"Set max_cells to increase the limit.",
        UserWarning,
        stacklevel=3,
    )
    rng = np.random.default_rng(seed)
    indices = rng.choice(adata.n_obs, size=max_cells, replace=False)
    indices.sort()
    return adata[indices].copy()


def _build_explorer_payload(
    adata,
    *,
    model_key: str,
    embedding_key: str,
    spatial_key: str | None,
    latent_key: str,
    layers: list[str] | None,
    obs_keys: list[str] | None,
    top_k: int,
    annotations_key: str | None,
) -> dict:
    """Extract all data needed for the interactive explorer.

    Parameters
    ----------
    adata
        AnnData object with fitted BAE results.
    model_key
        Key in adata.varm for encoder weights.
    embedding_key
        Key in adata.obsm for 2D embedding coordinates.
    spatial_key
        Key in adata.obsm for spatial coordinates, or None.
    latent_key
        Key in adata.obsm for latent representation.
    layers
        Layer names to include. None = adata.X + all adata.layers.
    obs_keys
        Categorical obs column names. None = auto-detect.
    top_k
        Max genes per sign group per dimension.
    annotations_key
        Key in adata.uns for dimension annotations, or None.

    Returns
    -------
    dict
        JSON-serializable payload for the HTML template.
    """
    import numpy as np

    # --- Embedding coordinates ---
    umap = np.asarray(adata.obsm[embedding_key], dtype=np.float64)
    umap_list = umap.tolist()

    # --- Spatial coordinates ---
    spatial_list = None
    if spatial_key is not None and spatial_key in adata.obsm:
        spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        spatial_list = spatial.tolist()

    # --- Latent representation ---
    latent = np.asarray(adata.obsm[latent_key], dtype=np.float64)
    latent_list = latent.tolist()

    # --- Encoder weights and gene rankings per dimension ---
    W = np.asarray(adata.varm[model_key], dtype=np.float64)
    gene_names_all = list(adata.var_names)
    _n_genes, latent_dim = W.shape

    all_top_genes: set[str] = set()
    dimensions_data: list[dict] = []

    for dim_idx in range(latent_dim):
        w = W[:, dim_idx]

        # Positive genes
        pos_mask = w > 0
        pos_indices = np.where(pos_mask)[0]
        pos_order = np.argsort(-w[pos_indices])
        pos_indices = pos_indices[pos_order][:top_k]
        pos_genes = [
            {"name": gene_names_all[i], "weight": round(float(w[i]), 6)} for i in pos_indices
        ]

        # Negative genes
        neg_mask = w < 0
        neg_indices = np.where(neg_mask)[0]
        neg_order = np.argsort(w[neg_indices])
        neg_indices = neg_indices[neg_order][:top_k]
        neg_genes = [
            {"name": gene_names_all[i], "weight": round(float(w[i]), 6)} for i in neg_indices
        ]

        # Collect gene names for expression extraction
        for g in pos_genes:
            all_top_genes.add(g["name"])
        for g in neg_genes:
            all_top_genes.add(g["name"])

        # Annotation for this dimension
        annotation = None
        if annotations_key and annotations_key in adata.uns:
            ann_dict = adata.uns[annotations_key]
            dim_key = str(dim_idx)
            if dim_key in ann_dict:
                a = ann_dict[dim_key]
                annotation = {
                    "positive": a.get("positive_annotation", ""),
                    "negative": a.get("negative_annotation", ""),
                    "overall": a.get("overall_annotation", ""),
                }

        dimensions_data.append(
            {
                "index": dim_idx,
                "positive_genes": pos_genes,
                "negative_genes": neg_genes,
                "annotation": annotation,
            }
        )

    # --- Expression data for top genes only ---
    sorted_top_genes = sorted(all_top_genes)
    gene_to_idx = {name: i for i, name in enumerate(gene_names_all)}

    # Determine layers
    layer_names: list[str] = []
    if layers is not None:
        layer_names = list(layers)
    else:
        # anndata >= 0.13 exposes ``.X`` as ``layers[None]``, so the mapping's
        # keys include ``None``. Taking them verbatim would emit ``.X`` twice,
        # once as "X" and again under a ``None`` key that `json.dumps` writes as
        # "null" -- a duplicated matrix rather than a crash.
        layer_names = ["X"] + [name for name in adata.layers if name is not None]

    expression: dict[str, dict[str, list[float]]] = {}
    for layer_name in layer_names:
        layer_expr: dict[str, list[float]] = {}
        matrix = adata.X if layer_name == "X" else adata.layers[layer_name]
        for gene in sorted_top_genes:
            idx = gene_to_idx[gene]
            col = matrix[:, idx]
            if hasattr(col, "toarray"):
                col = col.toarray()
            col = np.asarray(col, dtype=np.float64).ravel()
            layer_expr[gene] = [round(float(v), 6) for v in col]
        expression[layer_name] = layer_expr

    # --- Obs columns ---
    obs_data: dict[str, list[str]] = {}
    if obs_keys is not None:
        keys_to_use = obs_keys
    else:
        keys_to_use = [col for col in adata.obs.columns if hasattr(adata.obs[col], "cat")]
    for key in keys_to_use:
        obs_data[key] = [str(v) for v in adata.obs[key]]

    return {
        "umap": umap_list,
        "spatial": spatial_list,
        "latent": latent_list,
        "dimensions": dimensions_data,
        "expression": expression,
        "obs": obs_data,
        "gene_names": sorted_top_genes,
    }


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _render_html(payload: dict, *, title: str) -> str:
    """Render the interactive HTML explorer from payload data.

    Parameters
    ----------
    payload
        Data payload from _build_explorer_payload.
    title
        HTML page title.

    Returns
    -------
    str
        Complete HTML document as a string.
    """
    data_json = json.dumps(payload, separators=(",", ":"))
    has_spatial = payload["spatial"] is not None
    has_annotations = any(d["annotation"] is not None for d in payload["dimensions"])

    # Build the spatial plot div and JS conditionally
    spatial_div = ""
    spatial_col_style = ""
    if has_spatial:
        spatial_div = '<div id="tissue-plot" style="width:100%;height:100%;"></div>'
        spatial_col_style = "flex:1;min-width:300px;"

    # Layout widths depend on spatial mode
    left_col_style = "flex:1;min-width:350px;" if has_spatial else "flex:1.2;min-width:400px;"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_escape_html(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{ --pos-color: #f2994a; --neg-color: #2b6cb0; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #fafafa; color: #333; }}
  h1 {{ padding: 12px 20px; background: #fff; border-bottom: 1px solid #ddd;
       font-size: 1.3em; font-weight: 600; }}
  .container {{ display: flex; flex-wrap: wrap; padding: 10px; gap: 10px;
               height: calc(100vh - 52px); }}
  .left-col {{ {left_col_style} display:flex; flex-direction:column; gap:10px; }}
  .left-col > div {{ background:#fff; border:1px solid #ddd; border-radius:6px;
                    overflow:hidden; }}
  .umap-box {{ flex:1; min-height:280px; }}
  .coeff-box {{ flex:1; min-height:220px; }}
  {
        "".join(
            [
                ".mid-col { "
                + spatial_col_style
                + " display:flex; flex-direction:column; gap:10px; }",
                ".mid-col > div { background:#fff; border:1px solid #ddd; border-radius:6px; overflow:hidden; flex:1; min-height:300px; }",
            ]
        )
        if has_spatial
        else ""
    }
  .right-col {{ flex:0.7; min-width:280px; max-width:360px; display:flex;
               flex-direction:column; gap:10px; }}
  .controls {{ background:#fff; border:1px solid #ddd; border-radius:6px;
              padding:14px; }}
  .controls label {{ display:block; font-size:0.8em; font-weight:600;
                    color:#666; margin-bottom:3px; margin-top:10px; }}
  .controls label:first-child {{ margin-top:0; }}
  .controls select {{ width:100%; padding:6px 8px; border:1px solid #ccc;
                     border-radius:4px; font-size:0.9em; }}
  .controls input[type="range"] {{ width:100%; accent-color:#1f77b4; }}
  .controls .range-value {{ display:block; margin-top:2px; font-size:0.8em; color:#555; }}
  .annotation-box {{ background:#fff; border:1px solid #ddd; border-radius:6px;
                    padding:14px; font-size:0.82em; line-height:1.5;
                    {"display:none;" if not has_annotations else ""} }}
  .annotation-box h3 {{ font-size:0.85em; margin-bottom:6px; }}
  .annotation-box .ann-label {{ font-weight:600; color:#555; }}
  .gene-list {{ background:#fff; border:1px solid #ddd; border-radius:6px;
               padding:10px; flex:1; overflow-y:auto; min-height:100px; }}
  .gene-list h3 {{ font-size:0.85em; margin-bottom:8px; color:#444; }}
  .gene-item {{ display:flex; justify-content:space-between; padding:3px 6px;
               cursor:pointer; border-radius:3px; font-size:0.82em;
               font-family:monospace; }}
  .gene-item:hover {{ background:#e8f0fe; }}
  .gene-item.selected {{ background:#c8ddf8; font-weight:bold; }}
  .gene-pos {{ color: var(--pos-color); }}
  .gene-neg {{ color: var(--neg-color); }}
</style>
</head>
<body>
<h1>{_escape_html(title)}</h1>
<div class="container">
  <div class="left-col">
    <div class="umap-box"><div id="umap-plot" style="width:100%;height:100%;"></div></div>
    <div class="coeff-box"><div id="coeff-plot" style="width:100%;height:100%;"></div></div>
  </div>
  {"<div class='mid-col'><div>" + spatial_div + "</div></div>" if has_spatial else ""}
  <div class="right-col">
    <div class="controls">
      <label for="color-select">Color by</label>
      <select id="color-select"></select>
      <label for="obs-select">Obs column</label>
      <select id="obs-select"></select>
      <label for="dim-select">Dimension</label>
      <select id="dim-select"></select>
      <label for="layer-select">Expression layer</label>
      <select id="layer-select"></select>
      <label for="colorscale-select">Continuous colors</label>
      <select id="colorscale-select">
        <option value="blueorange" selected>Blue-Orange</option>
        <option value="redblue">Red-Blue</option>
        <option value="viridis">Viridis</option>
      </select>
      <label for="point-size">Dot size</label>
      <input id="point-size" type="range" min="1" max="12" step="0.5" value="3">
      <span class="range-value" id="point-size-value">3.0</span>
    </div>
    <div class="annotation-box" id="annotation-box">
      <h3>Dimension Annotation</h3>
      <div id="annotation-content"></div>
    </div>
    <div class="gene-list" id="gene-list-box">
      <h3>Top Genes</h3>
      <div id="gene-list"></div>
    </div>
  </div>
</div>
<script>
const DATA = {data_json};
const HAS_SPATIAL = {"true" if has_spatial else "false"};
const OBS_KEYS = Object.keys(DATA.obs);
const LAYER_KEYS = Object.keys(DATA.expression);
const HAS_OBS_OPTIONS = OBS_KEYS.length > 0;
const HAS_LAYER_OPTIONS = LAYER_KEYS.length > 0;

// --- Initialize controls ---
const colorSel = document.getElementById("color-select");
const obsSel = document.getElementById("obs-select");
const dimSel = document.getElementById("dim-select");
const layerSel = document.getElementById("layer-select");
const colorscaleSel = document.getElementById("colorscale-select");
const pointSizeInput = document.getElementById("point-size");
const pointSizeValue = document.getElementById("point-size-value");

// Color-by options: latent dims, then obs selector mode, then gene expression
DATA.dimensions.forEach(d => {{
  const o = document.createElement("option");
  o.value = "latent_" + d.index;
  o.textContent = "Latent dim " + d.index;
  colorSel.appendChild(o);
}});
if (HAS_OBS_OPTIONS) {{
  const o = document.createElement("option");
  o.value = "obs";
  o.textContent = "Obs column";
  colorSel.appendChild(o);
}}
if (HAS_LAYER_OPTIONS) {{
  const o = document.createElement("option");
  o.value = "gene";
  o.textContent = "Gene expression";
  colorSel.appendChild(o);
}}

// Obs dropdown
if (HAS_OBS_OPTIONS) {{
  OBS_KEYS.forEach(k => {{
    const o = document.createElement("option");
    o.value = k;
    o.textContent = k;
    obsSel.appendChild(o);
  }});
}} else {{
  const o = document.createElement("option");
  o.value = "";
  o.textContent = "No obs columns available";
  obsSel.appendChild(o);
}}

// Dimension dropdown
DATA.dimensions.forEach(d => {{
  const o = document.createElement("option");
  o.value = d.index;
  o.textContent = "Dimension " + d.index;
  dimSel.appendChild(o);
}});

// Layer dropdown
if (HAS_LAYER_OPTIONS) {{
  LAYER_KEYS.forEach(k => {{
    const o = document.createElement("option");
    o.value = k;
    o.textContent = k;
    layerSel.appendChild(o);
  }});
}} else {{
  const o = document.createElement("option");
  o.value = "";
  o.textContent = "No layers available";
  layerSel.appendChild(o);
}}

// --- State ---
let currentGene = null;
let currentDim = 0;
let currentPointSize = parseFloat(pointSizeInput.value);

// --- Plotting ---
const COLOR_SCALES = {{
  blueorange: [
    [0.0, "#2b6cb0"],
    [0.5, "#f7f7f7"],
    [1.0, "#f2994a"],
  ],
  redblue: [
    [0.0, "#2166ac"],
    [0.5, "#f7f7f7"],
    [1.0, "#b2182b"],
  ],
  viridis: "Viridis",
}};
const SIGN_COLORS = {{
  blueorange: {{ neg: "#2b6cb0", pos: "#f2994a" }},
  redblue: {{ neg: "#2166ac", pos: "#b2182b" }},
  viridis: {{ neg: "#440154", pos: "#fde725" }},
}};

function getActiveColorscale() {{
  return COLOR_SCALES[colorscaleSel.value] || COLOR_SCALES.blueorange;
}}

function updateGeneSignColors() {{
  const palette = SIGN_COLORS[colorscaleSel.value] || SIGN_COLORS.blueorange;
  document.documentElement.style.setProperty("--pos-color", palette.pos);
  document.documentElement.style.setProperty("--neg-color", palette.neg);
}}

function getColorRange(values) {{
  const nums = values.filter(v => Number.isFinite(v));
  if (!nums.length) return [0, 1];

  const vmin = Math.min(...nums);
  const vmax = Math.max(...nums);
  if (colorscaleSel.value === "viridis") {{
    if (vmin === vmax) return [vmin - 1, vmax + 1];
    return [vmin, vmax];
  }}

  const absMax = Math.max(Math.abs(vmin), Math.abs(vmax), 1e-9);
  return [-absMax, absMax];
}}

function makeScatter(divId, coords, color, showscale, isDiscrete, cmin, cmax) {{
  const trace = {{
    x: coords.map(c => c[0]),
    y: coords.map(c => c[1]),
    mode: "markers",
    type: "scattergl",
    marker: {{
      size: currentPointSize,
      showscale: showscale && !isDiscrete,
    }},
    hovertemplate: "%{{x:.2f}}, %{{y:.2f}}<br>%{{text}}<extra></extra>",
  }};
  if (isDiscrete) {{
    trace.marker.color = color.map(v => _categoryColor(v));
    trace.text = color;
  }} else {{
    trace.marker.color = color;
    trace.marker.colorscale = getActiveColorscale();
    trace.marker.cmin = cmin;
    trace.marker.cmax = cmax;
    trace.text = color.map(v => typeof v === "number" ? v.toFixed(3) : v);
  }}
  const layout = {{
    margin: {{ t: 8, b: 30, l: 35, r: 10 }},
    xaxis: {{ zeroline: false, showgrid: false }},
    yaxis: {{
      zeroline: false,
      showgrid: false,
      scaleanchor: divId === "tissue-plot" ? "x" : undefined,
    }},
    dragmode: "pan",
  }};
  Plotly.react(divId, [trace], layout, {{ responsive: true, scrollZoom: true }});
}}

function makeCoeffBar(dimIdx) {{
  const dim = DATA.dimensions[dimIdx];
  const genes = [...dim.positive_genes, ...dim.negative_genes];
  const names = genes.map(g => g.name);
  const weights = genes.map(g => g.weight);
  const [cmin, cmax] = getColorRange(weights);

  const trace = {{
    y: names,
    x: weights,
    type: "bar",
    orientation: "h",
    marker: {{
      color: weights,
      colorscale: getActiveColorscale(),
      cmin: cmin,
      cmax: cmax,
    }},
    hovertemplate: "%{{y}}: %{{x:.4f}}<extra></extra>",
  }};
  const layout = {{
    margin: {{ t: 8, b: 30, l: 80, r: 10 }},
    xaxis: {{ title: "Encoder weight", zeroline: true }},
    yaxis: {{ autorange: "reversed" }},
    dragmode: false,
  }};
  Plotly.react("coeff-plot", [trace], layout, {{ responsive: true }});

  // Click handler on bars
  document.getElementById("coeff-plot").removeAllListeners?.("plotly_click");
  document.getElementById("coeff-plot").on("plotly_click", function(data) {{
    const geneName = data.points[0].y;
    selectGene(geneName);
  }});
}}

// Category colors (up to 20 distinct)
const CAT_COLORS = [
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
  "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
  "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
];
const _catMap = {{}};
let _catIdx = 0;
function _categoryColor(val) {{
  if (!(val in _catMap)) {{
    _catMap[val] = CAT_COLORS[_catIdx % CAT_COLORS.length];
    _catIdx++;
  }}
  return _catMap[val];
}}

function resetCatColors() {{
  for (const k in _catMap) delete _catMap[k];
  _catIdx = 0;
}}

function updateControlStates() {{
  const isObsMode = colorSel.value === "obs";
  const isGeneMode = colorSel.value === "gene";
  obsSel.disabled = !HAS_OBS_OPTIONS || !isObsMode;
  layerSel.disabled = !HAS_LAYER_OPTIONS || !isGeneMode;
}}

// --- Update functions ---
function updatePlots() {{
  const colorVal = colorSel.value;
  let color, isDiscrete = false;
  let cmin;
  let cmax;

  if (colorVal.startsWith("latent_")) {{
    const idx = parseInt(colorVal.split("_")[1]);
    color = DATA.latent.map(row => row[idx]);
    [cmin, cmax] = getColorRange(color);
  }} else if (colorVal === "obs" && HAS_OBS_OPTIONS) {{
    const key = obsSel.value || OBS_KEYS[0];
    color = DATA.obs[key];
    isDiscrete = true;
    resetCatColors();
  }} else if (colorVal.startsWith("obs_")) {{
    const key = colorVal.substring(4);
    if (key in DATA.obs) {{
      color = DATA.obs[key];
      isDiscrete = true;
      resetCatColors();
    }} else {{
      color = DATA.latent.map(row => row[0]);
      [cmin, cmax] = getColorRange(color);
    }}
  }} else if (colorVal === "gene" && currentGene) {{
    const layer = layerSel.value || LAYER_KEYS[0];
    const layerExpr = DATA.expression[layer];
    if (layerExpr && currentGene in layerExpr) {{
      color = layerExpr[currentGene];
    }} else {{
      color = DATA.latent.map(row => row[0]);
      [cmin, cmax] = getColorRange(color);
    }}
  }} else {{
    // Default: first latent dim
    color = DATA.latent.map(row => row[0]);
    [cmin, cmax] = getColorRange(color);
  }}

  if (!isDiscrete && (cmin === undefined || cmax === undefined)) {{
    [cmin, cmax] = getColorRange(color);
  }}

  makeScatter("umap-plot", DATA.umap, color, true, isDiscrete, cmin, cmax);
  if (HAS_SPATIAL) {{
    makeScatter("tissue-plot", DATA.spatial, color, false, isDiscrete, cmin, cmax);
  }}
}}

function updateDimension() {{
  currentDim = parseInt(dimSel.value);
  makeCoeffBar(currentDim);
  updateGeneList();
  updateAnnotation();

  // If in gene expression mode, select first gene automatically
  if (colorSel.value === "gene" && HAS_LAYER_OPTIONS) {{
    const dim = DATA.dimensions[currentDim];
    const firstGene = dim.positive_genes.length > 0
      ? dim.positive_genes[0].name
      : (dim.negative_genes.length > 0 ? dim.negative_genes[0].name : null);
    if (firstGene) selectGene(firstGene);
  }}
}}

function updateGeneList() {{
  const dim = DATA.dimensions[currentDim];
  const container = document.getElementById("gene-list");
  container.innerHTML = "";

  dim.positive_genes.forEach(g => {{
    const div = document.createElement("div");
    div.className = "gene-item" + (g.name === currentGene ? " selected" : "");
    div.innerHTML = '<span class="gene-pos">+ ' + g.name + '</span><span>' + g.weight.toFixed(4) + '</span>';
    div.onclick = () => selectGene(g.name);
    container.appendChild(div);
  }});
  dim.negative_genes.forEach(g => {{
    const div = document.createElement("div");
    div.className = "gene-item" + (g.name === currentGene ? " selected" : "");
    div.innerHTML = '<span class="gene-neg">\u2212 ' + g.name + '</span><span>' + g.weight.toFixed(4) + '</span>';
    div.onclick = () => selectGene(g.name);
    container.appendChild(div);
  }});
}}

function updateAnnotation() {{
  const dim = DATA.dimensions[currentDim];
  const box = document.getElementById("annotation-box");
  const content = document.getElementById("annotation-content");
  if (dim.annotation) {{
    box.style.display = "block";
    content.innerHTML =
      '<p><span class="ann-label">Positive:</span> ' + dim.annotation.positive + '</p>' +
      '<p><span class="ann-label">Negative:</span> ' + dim.annotation.negative + '</p>' +
      '<p><span class="ann-label">Overall:</span> ' + dim.annotation.overall + '</p>';
  }} else {{
    box.style.display = "none";
    content.innerHTML = "";
  }}
}}

function selectGene(geneName) {{
  if (!HAS_LAYER_OPTIONS) return;
  currentGene = geneName;
  colorSel.value = "gene";
  updateControlStates();
  updatePlots();
  updateGeneList();
}}

// --- Event listeners ---
colorSel.addEventListener("change", () => {{
  updateControlStates();
  updatePlots();
}});
obsSel.addEventListener("change", () => {{
  if (colorSel.value === "obs") updatePlots();
}});
dimSel.addEventListener("change", updateDimension);
layerSel.addEventListener("change", () => {{
  if (colorSel.value === "gene" && currentGene && HAS_LAYER_OPTIONS) updatePlots();
}});
colorscaleSel.addEventListener("change", () => {{
  updateGeneSignColors();
  updatePlots();
  makeCoeffBar(currentDim);
}});
pointSizeInput.addEventListener("input", () => {{
  currentPointSize = parseFloat(pointSizeInput.value);
  pointSizeValue.textContent = currentPointSize.toFixed(1);
  updatePlots();
}});

// --- Initialize ---
updateGeneSignColors();
updateControlStates();
pointSizeValue.textContent = currentPointSize.toFixed(1);
updateDimension();
updatePlots();
</script>
</body>
</html>"""
    return html


def export_interactive_html(
    adata,
    output_path: str | Path,
    *,
    model_key: str = "BAE_encoder_weights",
    embedding_key: str = "X_umap",
    spatial_key: str | None = "spatial",
    latent_key: str | None = None,
    layers: list[str] | None = None,
    obs_keys: list[str] | None = None,
    top_k: int = 20,
    max_cells: int = 50_000,
    annotations_key: str | None = None,
    title: str = "BAE Explorer",
    seed: int = 42,
) -> Path:
    """Export interactive HTML explorer for BAE results.

    Generates a self-contained HTML file with Plotly.js for interactive
    exploration of UMAP embeddings, gene expression overlays, encoder
    coefficient bar charts, and (optionally) spatial tissue plots and
    dimension annotations.

    Parameters
    ----------
    adata
        AnnData object with fitted BAE results. Must contain
        encoder weights in ``adata.varm[model_key]`` and a 2D embedding
        in ``adata.obsm[embedding_key]``.
    output_path
        Path where the HTML file will be written.
    model_key
        Key in ``adata.varm`` for encoder weight matrix.
    embedding_key
        Key in ``adata.obsm`` for 2D embedding (e.g. UMAP). Must be
        pre-computed.
    spatial_key
        Key in ``adata.obsm`` for spatial coordinates. Set to ``None``
        to disable the spatial tissue plot. If the key is not found in
        obsm, the spatial panel is silently omitted.
    latent_key
        Key in ``adata.obsm`` for latent representation. If ``None``,
        defaults to ``"X_bae"``.
    layers
        Which expression layers to include. ``None`` includes
        ``adata.X`` (as ``"X"``) plus all keys in ``adata.layers``.
    obs_keys
        Categorical obs columns to include as color-by options. ``None``
        auto-detects all categorical columns.
    top_k
        Number of top genes per sign group (positive/negative) per
        latent dimension.
    max_cells
        If ``adata.n_obs`` exceeds this, randomly subsample with a
        warning.
    annotations_key
        Key in ``adata.uns`` for dimension annotations. If ``None``,
        auto-detected as ``"bae_dimension_annotations"``.
    title
        HTML page title.
    seed
        Random seed for reproducible subsampling.

    Returns
    -------
    Path
        The output file path.

    Raises
    ------
    KeyError
        If ``embedding_key`` or ``model_key`` are not found.
    """
    output_path = Path(output_path)

    # --- Validate required keys ---
    if embedding_key not in adata.obsm:
        raise KeyError(
            f"{embedding_key!r} not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
        )
    if model_key not in adata.varm:
        raise KeyError(
            f"{model_key!r} not found in adata.varm. Available keys: {list(adata.varm.keys())}"
        )

    # --- Auto-detect latent_key ---
    if latent_key is None:
        latent_key = "X_bae"
    if latent_key not in adata.obsm:
        raise KeyError(
            f"Auto-detected latent_key {latent_key!r} not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    # --- Auto-detect spatial_key ---
    if spatial_key is not None and spatial_key not in adata.obsm:
        spatial_key = None

    # --- Auto-detect annotations_key ---
    if annotations_key is None:
        candidate = "bae_dimension_annotations"
        if candidate in adata.uns:
            annotations_key = candidate

    # --- Subsample ---
    adata = _subsample_adata(adata, max_cells=max_cells, seed=seed)

    # --- Build payload and render ---
    payload = _build_explorer_payload(
        adata,
        model_key=model_key,
        embedding_key=embedding_key,
        spatial_key=spatial_key,
        latent_key=latent_key,
        layers=layers,
        obs_keys=obs_keys,
        top_k=top_k,
        annotations_key=annotations_key,
    )

    html = _render_html(payload, title=title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    return output_path
