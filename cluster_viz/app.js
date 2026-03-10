// Color mapping matching the CSS variables
const colorMap = {
    "Router": "#ef4444",
    "Cable": "#eab308",
    "Webcam": "#a855f7",
    "Mobile Phone": "#3b82f6",
    "SIM Card": "#10b981",
    "Keyboard": "#f43f5e",
    "Mouse": "#14b8a6",
    "Headphones": "#8b5cf6",
    "Monitor": "#0ea5e9",
    "Laptops": "#f97316",
    "Other": "#64748b"
};

let globalData = null;
let simulation = null;
let svg = null;
let g = null;
let zoom = null;
let nodeSelection = null;

// Initialization
let currentThreshold = 0.50;

async function init() {
    await fetchAndRenderData();
    setupUI();
}

async function fetchAndRenderData() {
    // Show loading indicator
    document.getElementById("graph-container").classList.add("loading");
    document.getElementById("viz").style.opacity = "0.5";

    try {
        const response = await fetch(`/api/clusters?threshold=${currentThreshold}`);
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
        document.getElementById("graph-container").innerHTML = `
            <div style="padding: 40px; text-align: center; color: #ef4444;">
                <h2>Error loading data</h2>
                <p>Run <code>python3 -m http.server</code> in the cluster_viz directory and visit localhost:8000</p>
            </div>
        `;
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
                <div class="legend-color" style="background-color: ${colorMap[cat]}; color: ${colorMap[cat]}"></div>
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

    thresholdInput.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value).toFixed(2);
        thresholdValDisplay.textContent = val;
        currentThreshold = val;
    });

    reclusterBtn.addEventListener("click", () => {
        fetchAndRenderData();
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
        .attr("fill", d => colorMap[d.category] || colorMap["Other"])
        .attr("fill-opacity", 0.8)
        .on("mouseover", function (event, d) {
            d3.select(this).attr("stroke", "#fff").attr("stroke-width", 3);

            tooltip.transition().duration(200).style("opacity", 1);

            const sample = d.sample_records[0].join(" | ");

            // Calculate coordinates relative to the graph container to account for the sidebar
            const [x, y] = d3.pointer(event, document.getElementById("graph-container"));

            tooltip.html(`
                <div class="tooltip-title">Cluster ${d.id}</div>
                <div class="tooltip-stat">Category: <b>${d.category}</b></div>
                <div class="tooltip-stat">Records: <b>${d.size.toLocaleString()}</b></div>
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

    const color = colorMap[cluster.category] || colorMap["Other"];

    // Generate records HTML
    const recordsHtml = cluster.sample_records.map(rec => {
        return `<div class="record-item">${rec.join(" <span style='color:#64748b'>|</span> ")}</div>`;
    }).join("");

    let warningHtml = "";
    if (cluster.size > cluster.sample_records.length) {
        warningHtml = `<div style="margin-top: 16px; font-size: 0.8rem; color: var(--text-secondary); text-align: center;">Showing ${cluster.sample_records.length} of ${cluster.size.toLocaleString()} records</div>`;
    }

    content.innerHTML = `
        <div class="detail-header">
            <div class="cluster-badge" style="background-color: ${color}20; color: ${color}; border: 1px solid ${color}40">
                ${cluster.category}
            </div>
            <h2 class="detail-title">Cluster ${cluster.id}</h2>
            <div class="detail-stats">
                <span><b style="color:white">${cluster.size.toLocaleString()}</b> Records</span>
            </div>
        </div>
        
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
