// Color generator for dynamic categories
const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
// specific overrides for visually important categories
const specificColors = {
    "Other": "#64748b",
    "SIM Card": "#10b981",
    "Cable": "#eab308",
    "Storage": "#f43f5e",
    "Adapter": "#a855f7",
    "PC Component": "#3b82f6",
    "Telecom": "#0284c7"
};

function getColor(category) {
    if (specificColors[category]) return specificColors[category];
    return colorScale(category);
}

let globalData = null;
let simulation = null;
let svg = null;
let g = null;
let zoom = null;
let nodeSelection = null;

// Initialization
let currentThreshold = 0.30;

/** Poll until initial embeddings exist (server serves the GUI before embeddings finish). */
async function ensureEmbeddingsReady() {
    const overlay = document.getElementById("embedding-overlay");
    const textEl = document.getElementById("embedding-overlay-text");
    const hintEl = overlay?.querySelector(".embedding-overlay-hint");
    const showOverlay = () => overlay?.classList.remove("is-hidden");
    const hideOverlay = () => overlay?.classList.add("is-hidden");

    for (;;) {
        let statusRes;
        try {
            statusRes = await fetch("/api/status");
        } catch (_) {
            await new Promise((r) => setTimeout(r, 1500));
            continue;
        }
        if (!statusRes.ok) {
            await new Promise((r) => setTimeout(r, 1500));
            continue;
        }
        const s = await statusRes.json();
        if (s.embeddings_status === "ready") {
            hideOverlay();
            return;
        }
        if (s.embeddings_status === "error") {
            hideOverlay();
            throw new Error(s.embeddings_error || "Embedding generation failed.");
        }
        if (s.embeddings_status !== "loading") {
            hideOverlay();
            return;
        }
        showOverlay();
        if (textEl) {
            const n = s.row_count != null ? Number(s.row_count).toLocaleString() : "?";
            textEl.textContent = `Generating embeddings for ${n} records…`;
        }
        if (hintEl) {
            hintEl.textContent =
                "The page is loaded; this step may take a few minutes on CPU.";
        }
        await new Promise((r) => setTimeout(r, 1200));
    }
}

async function init() {
    await fetchAndRenderData();
    setupUI();
}

async function fetchAndRenderData() {
    document.getElementById("embedding-overlay")?.classList.remove("embedding-overlay--error");
    // Show loading indicator
    document.getElementById("graph-container").classList.add("loading");
    document.getElementById("viz").style.opacity = "0.5";

    try {
        await ensureEmbeddingsReady();

        const response = await fetch(`/api/clusters?threshold=${currentThreshold}`);
        if (!response.ok) {
            const errBody = await response.json().catch(() => ({}));
            throw new Error(errBody.detail || `HTTP ${response.status}`);
        }
        globalData = await response.json();

        // Clear existing graph elements if any
        if (svg) {
            d3.select("#viz").selectAll("*").remove();
            clearSearch();
            document.getElementById("detail-panel").classList.remove("visible");
        }

        updateStatsAndLegend();
        drawGraph();
    } catch (err) {
        console.error("Failed to fetch clustering data from API.", err);
        const overlay = document.getElementById("embedding-overlay");
        const textEl = document.getElementById("embedding-overlay-text");
        const hintEl = overlay?.querySelector(".embedding-overlay-hint");
        if (overlay && textEl) {
            overlay.classList.remove("is-hidden");
            overlay.classList.add("embedding-overlay--error");
            textEl.textContent = err.message || String(err);
            if (hintEl) {
                hintEl.textContent =
                    "Check that the server is running (e.g. python server.py) and refresh the page.";
            }
        } else {
            document.getElementById("graph-container").innerHTML = `
            <div style="padding: 40px; text-align: center; color: #ef4444;">
                <h2>Error loading data</h2>
                <p>${(err.message || String(err)).replace(/</g, "&lt;")}</p>
                <p style="margin-top:1rem;">Make sure the backend server is running.</p>
            </div>
        `;
        }
    } finally {
        document.getElementById("graph-container").classList.remove("loading");
        document.getElementById("viz").style.opacity = "1";
    }
}

function updateStatsAndLegend() {
    // Stats
    const statsHtml = `
        <div class="stat-box">
            <span class="stat-val">${globalData.total_rows.toLocaleString()}</span>
            <span class="stat-label">Total Records</span>
        </div>
        <div class="stat-box">
            <span class="stat-val">${globalData.total_clusters.toLocaleString()}</span>
            <span class="stat-label">Clusters</span>
        </div>
    `;
    document.getElementById("stats-container").innerHTML = statsHtml;

    // Legend
    const legendHtml = Object.entries(globalData.category_counts)
        .sort((a, b) => b[1] - a[1]) // Sort by count desc
        .map(([cat, count]) => `
            <li class="legend-item">
                <div class="legend-color" style="background-color: ${getColor(cat)}; color: ${getColor(cat)}"></div>
                <span>${cat} <span style="color:var(--text-secondary); font-size:0.8em">(${count})</span></span>
            </li>
        `).join("");
    document.getElementById("legend-list").innerHTML = legendHtml;
}

function setupUI() {
    // Controls
    document.getElementById("search").addEventListener("input", handleSearch);

    // Setup size filter based on actual data
    const sizes = globalData.clusters.map(c => c.size);
    const maxSize = Math.max(...sizes);
    const sizeInput = document.getElementById("size-filter");
    sizeInput.max = Math.min(maxSize, 500); // cap slider max
    sizeInput.addEventListener("input", (e) => {
        const val = e.target.value;
        document.getElementById("size-val").textContent = val;
        filterNodesBySize(parseInt(val));
    });

    // Threshold Filter
    const thresholdInput = document.getElementById("threshold-filter");
    const thresholdValDisplay = document.getElementById("threshold-val");
    const reclusterBtn = document.getElementById("recluster-btn");
    const uploadBtn = document.getElementById("upload-btn");
    const fileInput = document.getElementById("file-upload");

    thresholdInput.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value).toFixed(2);
        thresholdValDisplay.textContent = val;
        currentThreshold = val;
    });

    reclusterBtn.addEventListener("click", () => {
        fetchAndRenderData();
    });

    // Custom Dataset Upload
    uploadBtn.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Reset the input so the same file can be uploaded again if needed
        e.target.value = "";

        const formData = new FormData();
        formData.append("file", file);
        formData.append("threshold", currentThreshold);

        // Show a custom loading indicator since embedding takes longer
        document.getElementById("graph-container").classList.add("loading");

        // Temporarily change the loading text
        const style = document.createElement('style');
        style.innerHTML = `#graph-container.loading::after { content: "Processing File & Generating Embeddings..."; }`;
        document.head.appendChild(style);

        document.getElementById("viz").style.opacity = "0.5";

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Upload failed");
            }

            globalData = await response.json();

            // Clear existing graph
            if (svg) {
                d3.select("#viz").selectAll("*").remove();
                clearSearch();
                document.getElementById("detail-panel").classList.remove("visible");
            }

            updateStatsAndLegend();
            drawGraph();
        } catch (err) {
            console.error("Failed to upload dataset.", err);
            alert("Error: " + err.message);
        } finally {
            document.getElementById("graph-container").classList.remove("loading");
            document.getElementById("viz").style.opacity = "1";
            document.head.removeChild(style); // Remove custom loading text
        }
    });

    document.getElementById("reset-zoom").addEventListener("click", () => {
        svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        clearSearch();

        // Reset min cluster size
        const sizeInput = document.getElementById("size-filter");
        sizeInput.value = 1;
        document.getElementById("size-val").textContent = "1";
        filterNodesBySize(1);
    });

    // Export to Excel
    document.getElementById("export-btn").addEventListener("click", async () => {
        try {
            // Open save dialog first (requires user gesture) before any async work
            let fileHandle = null;
            if ("showSaveFilePicker" in window) {
                try {
                    fileHandle = await window.showSaveFilePicker({
                        suggestedName: "clusters_export.xlsx",
                        types: [{ description: "Excel workbook", accept: { "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"] } }],
                    });
                } catch (e) {
                    if (e.name === "AbortError") return;
                    // Fall back to download if save picker fails (e.g. Firefox, Safari)
                    fileHandle = null;
                }
            }

            const response = await fetch(`/api/export?threshold=${currentThreshold}`);
            if (!response.ok) {
                const errText = await response.text();
                let msg = "Export failed";
                try {
                    const errJson = JSON.parse(errText);
                    msg = errJson.detail || errJson.message || msg;
                } catch (_) {
                    if (errText) msg = errText.slice(0, 200);
                }
                throw new Error(msg);
            }
            const blob = await response.blob();

            if (fileHandle) {
                const writable = await fileHandle.createWritable();
                await writable.write(blob);
                await writable.close();
            } else {
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = "clusters_export.xlsx";
                a.click();
                URL.revokeObjectURL(url);
            }
        } catch (err) {
            if (err.name === "AbortError") return; // User cancelled
            console.error("Export failed:", err);
            alert("Export failed: " + (err.message || String(err)));
        }
    });

    document.getElementById("close-detail").addEventListener("click", closeDetailPanel);
}

function drawGraph() {
    const width = document.getElementById("graph-container").clientWidth;
    const height = document.getElementById("graph-container").clientHeight;

    svg = d3.select("#viz")
        .attr("viewBox", [0, 0, width, height]);

    g = svg.append("g");

    // Zoom behavior
    zoom = d3.zoom()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => {
            g.attr("transform", event.transform);
        });

    svg.call(zoom);

    // Node scale - logarithmic so massive clusters don't obscure everything
    const sizeScale = d3.scaleLog()
        .domain([1, d3.max(globalData.clusters, d => d.size)])
        .range([5, 50])
        .clamp(true);

    const nodes = globalData.clusters.map(d => Object.create(d));

    // Force simulation - Radial clustering with collision
    simulation = d3.forceSimulation(nodes)
        // Pull towards center
        .force("charge", d3.forceManyBody().strength(d => -sizeScale(d.size) * 1.5))
        // Group by category radially
        .force("x", d3.forceX(width / 2).strength(0.05))
        .force("y", d3.forceY(height / 2).strength(0.05))
        // Prevent overlap based on node size + padding
        .force("collide", d3.forceCollide().radius(d => sizeScale(d.size) + 2).iterations(4))
        .on("tick", ticked);

    // Tooltip
    const tooltip = d3.select("#node-tooltip");

    nodeSelection = g.append("g")
        .selectAll("circle")
        .data(nodes)
        .join("circle")
        .attr("class", "node")
        .attr("r", d => sizeScale(d.size))
        .attr("fill", d => getColor(d.category))
        .attr("fill-opacity", 0.8)
        .on("mouseover", function (event, d) {
            d3.select(this).attr("stroke", "#fff").attr("stroke-width", 3);

            tooltip.transition().duration(200).style("opacity", 1);

            const sample = d.sample_records[0].join(" | ");

            // Calculate coordinates relative to the graph container to account for the sidebar
            const [x, y] = d3.pointer(event, document.getElementById("graph-container"));

            tooltip.html(`
                <div class="tooltip-title">Cluster ${d.id}</div>
                <div class="tooltip-stat">Category: <b>${d.category}${d.subcategory ? ` / ${d.subcategory}` : ""}</b></div>
                <div class="tooltip-stat">Records: <b>${d.size.toLocaleString()}</b></div>
                ${d.matched_device ? `<div class="tooltip-stat" style="color:#10b981;">🔩 ${d.matched_device}</div>` : ''}
                <div class="tooltip-sample">${sample}</div>
            `)
                .style("left", (x + 15) + "px")
                .style("top", (y + 15) + "px");
        })
        .on("mouseout", function () {
            if (!this.classList.contains("highlighted")) {
                d3.select(this).attr("stroke", "var(--bg-color)").attr("stroke-width", 1.5);
            }
            tooltip.transition().duration(500).style("opacity", 0);
        })
        .on("click", function (event, d) {
            showDetailPanel(d);

            // Highlight clicked node
            nodeSelection.classed("highlighted", false);
            d3.select(this).classed("highlighted", true);

            // Zoom to it
            const scale = 2;
            const tx = width / 2 - d.x * scale;
            const ty = height / 2 - d.y * scale;

            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity.translate(tx, ty).scale(scale)
            );
        })
        .call(drag(simulation));

    function ticked() {
        nodeSelection
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
    }
}

function handleSearch(e) {
    const term = e.target.value.toLowerCase();

    if (!term) {
        clearSearch();
        return;
    }

    nodeSelection.classed("dimmed", d => {
        // Check if node matches the search term
        const matchesCategory = d.category.toLowerCase().includes(term);
        const recordsStr = d.sample_records.flat().join(" ").toLowerCase();
        const matchesRecords = recordsStr.includes(term);

        // A node should be dimmed if it DOES NOT match the search term
        return !(matchesCategory || matchesRecords);
    });
}

function clearSearch() {
    document.getElementById("search").value = "";
    nodeSelection.classed("dimmed", false);
    nodeSelection.classed("highlighted", false);
}

function filterNodesBySize(minSize) {
    nodeSelection.style("display", d => d.size >= minSize ? "block" : "none");

    // Reheat simulation slightly to adjust for hidden nodes
    simulation.alpha(0.3).restart();
}

function showDetailPanel(cluster) {
    const detailPanel = document.getElementById("detail-panel");
    const content = document.getElementById("detail-content");

    const color = getColor(cluster.category);

    // Generate records HTML
    const recordsHtml = cluster.sample_records.map(rec => {
        return `<div class="record-item">${rec.join(" <span style='color:#64748b'>|</span> ")}</div>`;
    }).join("");

    let warningHtml = "";
    if (cluster.size > cluster.sample_records.length) {
        warningHtml = `<div style="margin-top: 16px; font-size: 0.8rem; color: var(--text-secondary); text-align: center;">Showing ${cluster.sample_records.length} of ${cluster.size.toLocaleString()} records</div>`;
    }

    const matchedDeviceHtml = cluster.matched_device ? `
        <div style="
            background: linear-gradient(135deg, #064e3b22, #10b98122);
            border: 1px solid #10b98140;
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 18px;
        ">
            <div style="font-size: 0.72rem; color: #10b981; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
                🔩 Matched iFixit Device
            </div>
            <div style="font-size: 1rem; font-weight: 600; color: white; margin-bottom: 4px;">
                ${cluster.device_url
                    ? `<a href="${cluster.device_url}" target="_blank" rel="noopener noreferrer"
                        style="color: #34d399; text-decoration: none;"
                        onmouseover="this.style.textDecoration='underline'"
                        onmouseout="this.style.textDecoration='none'"
                       >${cluster.matched_device} ↗</a>`
                    : cluster.matched_device
                }
            </div>
            ${cluster.matched_device_category ? `<div style="font-size: 0.8rem; color: #6ee7b7;">Category: ${cluster.matched_device_category}</div>` : ''}
        </div>
    ` : '';

    content.innerHTML = `
        <div class="detail-header">
            <div class="cluster-badge" style="background-color: ${color}20; color: ${color}; border: 1px solid ${color}40">
                ${cluster.category}${cluster.subcategory ? ` / ${cluster.subcategory}` : ""}
            </div>
            <h2 class="detail-title">Cluster ${cluster.id}</h2>
            <div class="detail-stats">
                <span><b style="color:white">${cluster.size.toLocaleString()}</b> Records</span>
            </div>
        </div>
        
        ${matchedDeviceHtml}
        
        <h3>Sample Records</h3>
        <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 16px;">
            Columns: ${globalData.columns.join(", ")}
        </p>
        
        <div class="records-list">
            ${recordsHtml}
        </div>
        ${warningHtml}
    `;

    detailPanel.classList.add("visible");
}


function closeDetailPanel() {
    document.getElementById("detail-panel").classList.remove("visible");
    nodeSelection.classed("highlighted", false);
}

// Drag behavior for nodes
function drag(simulation) {
    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }

    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }

    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }

    return d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
}

// Start
init();
