//! # Terminal Spectral Harness
//!
//! Wraps [`spectral_graph_agent::SpectralGraph`] into the terminal's
//! dashboard format — the eigenvalue computation lives in the metal library,
//! this crate **only** does UI formatting and graph building.
//!
//! ## Key insight
//!
//! `spectral-graph-agent-rs` already provides:
//!
//! - Full QR-based eigendecomposition (Wilkinson shifts)
//! - `fiedler_value()` / `fiedler_vector()`
//! - `cheeger_constant()` via sweep cut on the Fiedler vector
//! - `mixing_time()` via the normalized Laplacian
//! - `expander_quality()` with Ramanujan bounds
//!
//! This crate maps those onto the terminal's `AgentGraph` / `SpectralDashboard`
//! types so the terminal can display spectral metrics without re-implementing
//! any eigenvalue math.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use spectral_graph_agent::SpectralGraph;

// ── Terminal-compatible graph types ───────────────────────────────────

/// A node (agent) in the collaboration graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentNode {
    /// Agent identifier (e.g. "copilot", "claude").
    pub id: String,
    /// Display label.
    pub label: String,
    /// Whether the session is currently alive.
    pub alive: bool,
}

/// A weighted edge between two agent nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollabEdge {
    /// Source node index.
    pub source: usize,
    /// Target node index.
    pub target: usize,
    /// Edge weight (shared context strength, 0.0–1.0).
    pub weight: f64,
}

// ── AgentGraph: builds SpectralGraph under the hood ──────────────────

/// A collaboration graph of agents that delegates all spectral computation
/// to [`SpectralGraph`] from the metal library.
///
/// The graph is first built in a phase: nodes are registered by name, edges
/// are added by agent ID, and then [`finalize`](AgentGraph::finalize)
/// constructs the backing [`SpectralGraph`] from node indices.
#[derive(Debug, Clone)]
pub struct AgentGraph {
    /// Nodes (agents).
    pub nodes: Vec<AgentNode>,
    /// Weighted edges (terminal format, stored during construction).
    pub edges: Vec<CollabEdge>,
    /// The backing metal-library graph. Created on finalize.
    backing: Option<SpectralGraph>,
    /// Whether the backing graph needs rebuilding.
    dirty: bool,
}

impl AgentGraph {
    /// Create a new empty agent collaboration graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            backing: None,
            dirty: true,
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, id: &str, label: &str, alive: bool) {
        self.nodes.push(AgentNode {
            id: id.to_string(),
            label: label.to_string(),
            alive,
        });
        self.invalidate_cache();
    }

    /// Add a weighted edge between two agents by their id strings.
    /// Returns an error if either id is unknown.
    pub fn add_edge(&mut self, source_id: &str, target_id: &str, weight: f64) -> Result<(), String> {
        let source = self
            .nodes
            .iter()
            .position(|n| n.id == source_id)
            .ok_or_else(|| format!("unknown source agent: {source_id}"))?;
        let target = self
            .nodes
            .iter()
            .position(|n| n.id == target_id)
            .ok_or_else(|| format!("unknown target agent: {target_id}"))?;

        // Avoid duplicate edges; update weight if exists.
        for e in &mut self.edges {
            if (e.source == source && e.target == target)
                || (e.source == target && e.target == source)
            {
                e.weight = weight.max(e.weight);
                self.dirty = true;
                return Ok(());
            }
        }

        self.edges.push(CollabEdge {
            source,
            target,
            weight: weight.clamp(0.0, 1.0),
        });
        self.dirty = true;
        Ok(())
    }

    /// Remove an agent node by id.
    pub fn remove_node(&mut self, id: &str) {
        if let Some(pos) = self.nodes.iter().position(|n| n.id == id) {
            self.nodes.remove(pos);
            // Remove edges that reference this node.
            self.edges.retain(|e| e.source != pos && e.target != pos);
            // Re-index edges for nodes after the removed one.
            for e in &mut self.edges {
                if e.source > pos {
                    e.source -= 1;
                }
                if e.target > pos {
                    e.target -= 1;
                }
            }
            self.dirty = true;
        }
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Finalize the backing [`SpectralGraph`] from the current nodes and edges.
    ///
    /// Must be called before any spectral query. Calling it multiple times
    /// is safe — no-ops when already up-to-date.
    pub fn finalize(&mut self) {
        if !self.dirty && self.backing.is_some() {
            return;
        }

        let n = self.nodes.len();
        if n == 0 {
            self.backing = None;
            self.dirty = false;
            return;
        }

        let mut sg = SpectralGraph::new(n, false);
        for e in &self.edges {
            // SpectralGraph uses undirected edges; add_edge handles symmetry.
            let _ = sg.add_edge(e.source, e.target, e.weight);
        }
        sg.finalize();

        self.backing = Some(sg);
        self.dirty = false;
    }

    /// Get a reference to the backing [`SpectralGraph`], finalizing if needed.
    fn backing(&mut self) -> Option<&SpectralGraph> {
        self.finalize();
        self.backing.as_ref()
    }

    // ── Matrix access (delegated) ──────────────────────────

    /// Get the adjacency matrix. Delegates to SpectralGraph.
    pub fn adjacency_matrix(&mut self) -> Option<DMatrix<f64>> {
        self.backing().map(|sg| sg.adjacency_matrix())
    }

    /// Get the Laplacian matrix. Delegates to SpectralGraph.
    pub fn laplacian_matrix(&mut self) -> Option<DMatrix<f64>> {
        self.backing().map(|sg| sg.laplacian())
    }

    // ── Spectral metrics (delegated to SpectralGraph) ──────

    /// Compute the Fiedler value (algebraic connectivity): λ₂ of the Laplacian.
    ///
    /// Delegates to [`SpectralGraph::fiedler_value`], which uses QR-based
    /// eigendecomposition with Wilkinson shifts. Returns `None` for graphs
    /// with fewer than 2 nodes or on computation failure.
    pub fn fiedler_value(&mut self) -> Option<f64> {
        let n = self.nodes.len();
        if n < 2 {
            return None;
        }
        self.backing()?.fiedler_value().ok()
    }

    /// Compute the Cheeger constant using a sweep cut on the Fiedler vector.
    ///
    /// Delegates to [`SpectralGraph::cheeger_constant`]. Returns `None`
    /// for graphs with fewer than 2 nodes or on computation failure.
    pub fn cheeger_constant(&mut self) -> Option<f64> {
        let n = self.nodes.len();
        if n < 2 {
            return None;
        }
        self.backing()?.cheeger_constant().ok()
    }

    /// Estimate the mixing time: the number of steps for a random walk
    /// on the graph to approach the stationary distribution.
    ///
    /// Delegates to [`SpectralGraph::mixing_time`]. Returns `None` for
    /// graphs with fewer than 2 nodes or disconnected graphs.
    pub fn mixing_time(&mut self) -> Option<f64> {
        let n = self.nodes.len();
        if n < 2 {
            return None;
        }
        self.backing()?.mixing_time()
    }

    /// Compute the expander quality: how close the graph is to Ramanujan.
    ///
    /// Higher is better. Delegates to [`SpectralGraph::expander_quality`].
    pub fn expander_quality(&mut self) -> Option<f64> {
        self.backing()?.expander_quality().ok()
    }

    /// Compute a compact status bar display string.
    ///
    /// Format: `λ₂=0.34 h=0.21 t=3` or empty string for small graphs.
    pub fn status_bar_indicator(&mut self) -> String {
        let n = self.nodes.len();
        if n < 2 {
            return String::new();
        }

        let fiedler = self.fiedler_value().unwrap_or(0.0);
        let cheeger = self.cheeger_constant().unwrap_or(0.0);
        let mixing = self.mixing_time().unwrap_or(0.0);

        format!("λ₂={fiedler:.2} h={cheeger:.2} t={mixing:.0}")
    }

    /// Reset cache so the backing graph is rebuilt on the next query.
    pub fn invalidate_cache(&mut self) {
        self.dirty = true;
    }
}

impl Default for AgentGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── SpectralDashboard: periodic recomputation wrapper ─────────────────

/// A dashboard that holds an [`AgentGraph`] and provides periodic
/// recomputation of spectral metrics.
#[derive(Debug, Clone)]
pub struct SpectralDashboard {
    /// The agent collaboration graph (backs onto SpectralGraph).
    pub graph: AgentGraph,
    /// Last computed Fiedler value.
    pub last_fiedler: Option<f64>,
    /// Last computed Cheeger constant.
    pub last_cheeger: Option<f64>,
    /// Last computed mixing time.
    pub last_mixing_time: Option<f64>,
    /// Tick counter for periodic recomputation.
    ticks_since_update: u64,
    /// Recompute interval in ticks (default 10).
    pub recompute_interval: u64,
}

impl SpectralDashboard {
    /// Create a new spectral dashboard with the default recompute interval.
    pub fn new() -> Self {
        Self {
            graph: AgentGraph::new(),
            last_fiedler: None,
            last_cheeger: None,
            last_mixing_time: None,
            ticks_since_update: 0,
            recompute_interval: 10,
        }
    }

    /// Set the recompute interval (in ticks).
    pub fn set_recompute_interval(&mut self, interval: u64) {
        self.recompute_interval = interval;
    }

    /// Called on each app tick. Periodically recomputes spectral metrics.
    pub fn tick(&mut self) {
        self.ticks_since_update += 1;
        if self.ticks_since_update >= self.recompute_interval {
            self.recompute();
            self.ticks_since_update = 0;
        }
    }

    /// Force recomputation of spectral metrics.
    pub fn recompute(&mut self) {
        if self.graph.num_nodes() >= 2 {
            // Finalize the backing graph first (no-op if already done).
            self.graph.finalize();
            self.last_fiedler = self.graph.fiedler_value();
            self.last_cheeger = self.graph.cheeger_constant();
            self.last_mixing_time = self.graph.mixing_time();
        }
    }

    /// Get the status bar indicator string.
    pub fn status_bar_indicator(&mut self) -> String {
        self.graph.status_bar_indicator()
    }
}

impl Default for SpectralDashboard {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Graph Construction ───────────────────────────────────────────

    #[test]
    fn empty_graph() {
        let mut g = AgentGraph::new();
        assert_eq!(g.num_nodes(), 0);
        assert_eq!(g.num_edges(), 0);
        assert!(g.fiedler_value().is_none());
    }

    #[test]
    fn add_nodes() {
        let mut g = AgentGraph::new();
        g.add_node("copilot", "Copilot", true);
        g.add_node("claude", "Claude", true);
        assert_eq!(g.num_nodes(), 2);
    }

    #[test]
    fn add_edge_connects_nodes() {
        let mut g = AgentGraph::new();
        g.add_node("a", "Agent A", true);
        g.add_node("b", "Agent B", true);
        assert!(g.add_edge("a", "b", 0.8).is_ok());
        assert_eq!(g.num_edges(), 1);
    }

    #[test]
    fn add_edge_unknown_node_fails() {
        let mut g = AgentGraph::new();
        assert!(g.add_edge("nonexistent", "b", 0.5).is_err());
    }

    #[test]
    fn remove_node_removes_edges() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        g.add_node("b", "B", true);
        g.add_node("c", "C", true);
        let _ = g.add_edge("a", "b", 1.0);
        let _ = g.add_edge("b", "c", 1.0);
        g.remove_node("b");
        assert_eq!(g.num_nodes(), 2);
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn duplicate_edge_updates_weight() {
        let mut g = AgentGraph::new();
        g.add_node("x", "X", true);
        g.add_node("y", "Y", true);
        let _ = g.add_edge("x", "y", 0.5);
        let _ = g.add_edge("y", "x", 0.9);
        assert_eq!(g.num_edges(), 1);
        assert!((g.edges[0].weight - 0.9).abs() < 1e-10);
    }

    #[test]
    fn adjacency_matrix_symmetric() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        g.add_node("b", "B", true);
        let _ = g.add_edge("a", "b", 0.7);

        let adj = g.adjacency_matrix().unwrap();
        assert!((adj[(0, 1)] - 0.7).abs() < 1e-10);
        assert!((adj[(1, 0)] - 0.7).abs() < 1e-10);
    }

    // ─── Spectral (Delegated) ────────────────────────────────────────

    #[test]
    fn fiedler_value_two_nodes_full_edge() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        g.add_node("b", "B", true);
        let _ = g.add_edge("a", "b", 1.0);

        let fv = g.fiedler_value().unwrap();
        // For a 2-node graph with weight 1.0:
        // Laplacian = [[1, -1], [-1, 1]], eigenvalues = [0, 2].
        // Fiedler value = 2.
        assert!(
            (fv - 2.0).abs() < 0.1,
            "Fiedler value should be ~2.0, got {fv}"
        );
    }

    #[test]
    fn fiedler_value_disconnected_graph() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        g.add_node("b", "B", true);
        g.add_node("c", "C", true);
        // Only connect a-b, leave c isolated.
        let _ = g.add_edge("a", "b", 1.0);

        let fv = g.fiedler_value().unwrap();
        assert!(
            fv < 1.0,
            "disconnected graph should have near-zero Fiedler value, got {fv}"
        );
    }

    #[test]
    fn cheeger_constant_two_node() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        g.add_node("b", "B", true);
        let _ = g.add_edge("a", "b", 1.0);

        let h = g.cheeger_constant().unwrap();
        assert!(
            h > 0.0 && h <= 1.0,
            "Cheeger for two connected nodes should be in (0,1], got {h}"
        );
    }

    #[test]
    fn mixing_time_finite() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        g.add_node("b", "B", true);
        g.add_node("c", "C", true);
        g.add_node("d", "D", true);
        let _ = g.add_edge("a", "b", 0.5);
        let _ = g.add_edge("b", "c", 0.5);
        let _ = g.add_edge("c", "d", 0.5);
        let _ = g.add_edge("d", "a", 0.5);
        let _ = g.add_edge("a", "c", 0.3);
        let _ = g.add_edge("b", "d", 0.3);

        let mt = g.mixing_time();
        assert!(mt.is_some(), "mixing time should be Some for connected graph");
        let mtv = mt.unwrap();
        assert!(mtv > 0.0 && mtv < 1_000.0, "mixing time should be reasonable, got {mtv}");
    }

    #[test]
    fn fiedler_none_for_single_node() {
        let mut g = AgentGraph::new();
        g.add_node("only", "Only", true);
        assert!(g.fiedler_value().is_none());
    }

    #[test]
    fn cheeger_none_for_single_node() {
        let mut g = AgentGraph::new();
        g.add_node("only", "Only", true);
        assert!(g.cheeger_constant().is_none());
    }

    #[test]
    fn status_bar_indicator_empty_for_small_graph() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        assert!(g.status_bar_indicator().is_empty());
    }

    #[test]
    fn status_bar_indicator_format() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        g.add_node("b", "B", true);
        let _ = g.add_edge("a", "b", 1.0);

        let indicator = g.status_bar_indicator();
        assert!(!indicator.is_empty(), "indicator should not be empty for a 2-node graph");
        assert!(indicator.contains("λ₂="), "indicator should contain Fiedler label");
        assert!(indicator.contains("h="), "indicator should contain Cheeger label");
        assert!(indicator.contains("t="), "indicator should contain mixing-time label");
    }

    // ─── Dashboard ────────────────────────────────────────────────────

    #[test]
    fn dashboard_empty_on_creation() {
        let db = SpectralDashboard::new();
        assert!(db.last_fiedler.is_none());
        assert!(db.last_cheeger.is_none());
        assert!(db.last_mixing_time.is_none());
        assert_eq!(db.graph.num_nodes(), 0);
        assert_eq!(db.recompute_interval, 10);
    }

    #[test]
    fn dashboard_default() {
        let db = SpectralDashboard::default();
        assert!(db.last_fiedler.is_none());
    }

    #[test]
    fn dashboard_recompute_two_nodes() {
        let mut db = SpectralDashboard::new();
        db.graph.add_node("cli", "CLI Agent", true);
        db.graph.add_node("code", "Code Agent", true);
        let _ = db.graph.add_edge("cli", "code", 0.8);
        db.recompute();
        assert!(db.last_fiedler.is_some(), "Fiedler should be computed");
        assert!(db.last_cheeger.is_some(), "Cheeger should be computed");
        assert!(db.last_mixing_time.is_some(), "mixing time should be computed");
    }

    #[test]
    fn dashboard_tick_calls_recompute() {
        let mut db = SpectralDashboard::new();
        db.recompute_interval = 2;
        db.graph.add_node("a", "A", true);
        db.graph.add_node("b", "B", true);
        let _ = db.graph.add_edge("a", "b", 1.0);
        db.tick();
        assert!(db.last_fiedler.is_none(), "after 1 tick, should not yet recompute");
        db.tick();
        assert!(db.last_fiedler.is_some(), "after 2 ticks, should recompute");
    }

    #[test]
    fn dashboard_set_recompute_interval() {
        let mut db = SpectralDashboard::new();
        db.set_recompute_interval(5);
        assert_eq!(db.recompute_interval, 5);
    }

    // ─── Cache Invalidation ──────────────────────────────────────────

    #[test]
    fn invalidation_clears_dirty_flag() {
        let mut g = AgentGraph::new();
        g.add_node("x", "X", true);
        g.add_node("y", "Y", true);
        let _ = g.add_edge("x", "y", 1.0);
        let _ = g.fiedler_value();
        g.invalidate_cache();
        assert!(g.dirty, "should be marked dirty after invalidation");
    }

    #[test]
    fn adding_node_invalidates_cache() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        g.add_node("b", "B", true);
        let _ = g.add_edge("a", "b", 1.0);
        let _ = g.fiedler_value();
        assert!(!g.dirty);
        g.add_node("c", "C", true);
        assert!(g.dirty, "adding a node should mark graph dirty");
    }

    // ─── Expandability / Extra Metrics ────────────────────────────────

    #[test]
    fn expander_quality_available() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        g.add_node("b", "B", true);
        g.add_node("c", "C", true);
        g.add_node("d", "D", true);
        let _ = g.add_edge("a", "b", 1.0);
        let _ = g.add_edge("b", "c", 1.0);
        let _ = g.add_edge("c", "d", 1.0);
        let _ = g.add_edge("d", "a", 1.0);

        // Complete graph K4 is 3-regular with expander quality > 0.
        let q = g.expander_quality();
        assert!(q.is_some(), "expander quality should be Some for a connected regular graph");
        let qv = q.unwrap();
        assert!(qv >= 0.0, "expander quality should be non-negative, got {qv}");
    }

    #[test]
    fn expander_quality_none_for_empty() {
        let mut g = AgentGraph::new();
        assert!(g.expander_quality().is_none());
    }

    #[test]
    fn expander_quality_none_for_single() {
        let mut g = AgentGraph::new();
        g.add_node("only", "Only", true);
        assert!(g.expander_quality().is_none());
    }

    // ─── Finalize / matrix delegation ─────────────────────────────────

    #[test]
    fn finalize_rebuilds_on_change() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        g.add_node("b", "B", true);
        let _ = g.add_edge("a", "b", 1.0);

        // Before finalize, backing is None.
        assert!(g.backing.is_none());

        // Accessing a spectral method triggers finalize.
        let _ = g.fiedler_value();
        assert!(g.fiedler_value().is_some());
    }

    #[test]
    fn laplacian_matrix_produced() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        g.add_node("b", "B", true);
        let _ = g.add_edge("a", "b", 1.0);

        let lap = g.laplacian_matrix();
        assert!(lap.is_some(), "laplacian_matrix should return Some for a 2-node graph");
        let lap = lap.unwrap();
        assert_eq!(lap.nrows(), 2);
        assert_eq!(lap.ncols(), 2);
        assert!((lap[(0, 0)] - 1.0).abs() < 1e-10, "diagonal should be degree=1");
        assert!((lap[(0, 1)] - (-1.0)).abs() < 1e-10, "off-diagonal should be -weight");
    }

    #[test]
    fn adjacency_matrix_none_for_empty() {
        let mut g = AgentGraph::new();
        assert!(g.adjacency_matrix().is_none());
    }

    // ─── Serde round-trip ─────────────────────────────────────────────

    #[test]
    fn agent_node_serde_roundtrip() {
        let node = AgentNode {
            id: "test".into(),
            label: "Test Agent".into(),
            alive: true,
        };
        let json = serde_json::to_string(&node).unwrap();
        let parsed: AgentNode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "test");
        assert_eq!(parsed.label, "Test Agent");
        assert!(parsed.alive);
    }

    #[test]
    fn collab_edge_serde_roundtrip() {
        let edge = CollabEdge {
            source: 0,
            target: 1,
            weight: 0.75,
        };
        let json = serde_json::to_string(&edge).unwrap();
        let parsed: CollabEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source, 0);
        assert_eq!(parsed.target, 1);
        assert!((parsed.weight - 0.75).abs() < 1e-10);
    }

    // ─── State bar indicator formatting ───────────────────────────────

    #[test]
    fn status_bar_indicator_contains_all_metrics() {
        let mut g = AgentGraph::new();
        g.add_node("a", "A", true);
        g.add_node("b", "B", true);
        let _ = g.add_edge("a", "b", 1.0);

        let s = g.status_bar_indicator();
        // Expected format: "λ₂=X.XX h=X.XX t=X"
        assert!(s.starts_with("λ₂="), "should start with λ₂=, got: {s:?}");
        let parts: Vec<&str> = s.split_whitespace().collect();
        assert!(parts.len() >= 3, "should have at least 3 space-separated parts, got: {parts:?}");
    }

    #[test]
    fn status_bar_indicator_dashboard() {
        let mut db = SpectralDashboard::new();
        db.graph.add_node("alice", "Alice", true);
        db.graph.add_node("bob", "Bob", true);
        let _ = db.graph.add_edge("alice", "bob", 0.9);
        db.recompute();

        let s = db.status_bar_indicator();
        assert!(!s.is_empty(), "dashboard status bar should be non-empty after recompute");
        assert!(s.contains("λ₂="));
        assert!(s.contains("h="));
        assert!(s.contains("t="));
    }
}
