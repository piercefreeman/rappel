//! Shared DAG traversal logic for completion and exception handling.
//!
//! This module provides loop-aware BFS traversal that correctly handles:
//! - Loop-back edges (allowing re-visitation of loop nodes)
//! - Guard expression evaluation
//! - Infinite loop protection
//!
//! Both the completion handler and exception handler need this logic to
//! properly continue execution after an action completes or an exception
//! is caught inside a for loop.

use std::collections::{HashMap, HashSet, VecDeque};
use tracing::{debug, warn};

use crate::ast_evaluator::ExpressionEvaluator;
use crate::dag::DAGEdge;
use crate::dag_state::DAGHelper;
use crate::value::WorkflowValue;

/// Type alias for inline scope (variable name -> value).
pub type InlineScope = HashMap<String, WorkflowValue>;

/// Maximum number of loop iterations before terminating traversal.
/// This prevents infinite loops from hanging the system.
pub const MAX_LOOP_ITERATIONS: usize = 10000;

/// Result of evaluating a guard expression.
#[derive(Debug, Clone)]
pub enum GuardResult {
    /// Guard passed (true) - edge should be followed.
    Pass,
    /// Guard failed (false) - edge should not be followed.
    Fail,
    /// Guard evaluation had an error.
    Error(String),
}

/// Information about a successor edge during traversal.
#[derive(Debug, Clone)]
pub struct TraversalEdge {
    /// Target node ID.
    pub target: String,
    /// Whether this edge is a loop-back edge.
    pub is_loop_back: bool,
    /// Whether this is a default else edge.
    pub is_else: bool,
    /// Guard expression (if any).
    pub guard_expr: Option<crate::parser::ast::Expr>,
    /// Original condition string (if any).
    pub condition: Option<String>,
}

impl<'a> From<&'a DAGEdge> for TraversalEdge {
    fn from(edge: &'a DAGEdge) -> Self {
        Self {
            target: edge.target.clone(),
            is_loop_back: edge.is_loop_back,
            is_else: edge.is_else,
            guard_expr: edge.guard_expr.clone(),
            condition: edge.condition.clone(),
        }
    }
}

/// Evaluate a guard expression against a scope.
///
/// Returns `GuardResult::Pass` if no guard is present.
pub fn evaluate_guard(
    guard_expr: Option<&crate::parser::ast::Expr>,
    scope: &InlineScope,
    target_id: &str,
) -> GuardResult {
    let Some(guard) = guard_expr else {
        return GuardResult::Pass;
    };

    match ExpressionEvaluator::evaluate(guard, scope) {
        Ok(val) => {
            let is_true = val.is_truthy();
            debug!(
                target_id = %target_id,
                guard_expr = ?guard,
                result = ?val,
                is_true = is_true,
                "evaluated guard expression"
            );
            if is_true {
                GuardResult::Pass
            } else {
                GuardResult::Fail
            }
        }
        Err(e) => {
            warn!(
                target_id = %target_id,
                error = %e,
                "failed to evaluate guard expression"
            );
            GuardResult::Error(e.to_string())
        }
    }
}

/// Filter and select successor edges based on guard evaluation.
///
/// Handles the following logic:
/// - Edges without guards are always included
/// - Guarded edges are only included if their guard passes
/// - Else edges are only included if no guarded edge passed (and no error occurred)
///
/// Returns the selected edges and appends any guard errors to `guard_errors`.
pub fn select_guarded_edges(
    edges: Vec<TraversalEdge>,
    scope: &InlineScope,
    guard_errors: &mut Vec<(String, String)>,
) -> Vec<TraversalEdge> {
    let mut selected = Vec::new();
    let mut else_edges = Vec::new();
    let mut has_guarded_edges = false;
    let mut guard_passed = false;
    let mut guard_error = false;

    for edge in edges {
        if edge.is_else {
            else_edges.push(edge);
            continue;
        }

        if let Some(ref guard) = edge.guard_expr {
            has_guarded_edges = true;
            match evaluate_guard(Some(guard), scope, &edge.target) {
                GuardResult::Pass => {
                    guard_passed = true;
                    selected.push(edge);
                }
                GuardResult::Fail => {}
                GuardResult::Error(err) => {
                    guard_error = true;
                    guard_errors.push((edge.target.clone(), err));
                }
            }
            continue;
        }

        // No guard - always include
        selected.push(edge);
    }

    // Include else edges only if no guarded edge passed and no error
    if has_guarded_edges && !guard_passed && !guard_error {
        selected.extend(else_edges);
    }

    selected
}

/// Get successor edges for traversal from a node.
///
/// This uses `get_state_machine_successors` which:
/// - Includes loop-back edges (needed for for-loop continuation)
/// - Excludes exception edges (handled separately)
///
/// The returned edges preserve the `is_loop_back` flag for proper tracking.
pub fn get_traversal_successors<'a>(
    helper: &'a DAGHelper<'a>,
    node_id: &str,
) -> Vec<TraversalEdge> {
    helper
        .get_state_machine_successors(node_id)
        .into_iter()
        .map(TraversalEdge::from)
        .collect()
}

/// State for loop-aware BFS traversal.
///
/// Tracks visited nodes and loop iteration counts to:
/// - Allow re-visiting nodes reached via loop-back edges
/// - Prevent infinite loops
pub struct LoopAwareTraversal {
    /// Nodes visited during traversal (not via loop-back).
    visited: HashSet<String>,
    /// Loop iteration counts per node (for loop-back visits).
    loop_iterations: HashMap<String, usize>,
    /// Maximum allowed loop iterations.
    max_iterations: usize,
}

impl Default for LoopAwareTraversal {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopAwareTraversal {
    /// Create a new traversal state with default max iterations.
    pub fn new() -> Self {
        Self {
            visited: HashSet::new(),
            loop_iterations: HashMap::new(),
            max_iterations: MAX_LOOP_ITERATIONS,
        }
    }

    /// Create a new traversal state with custom max iterations.
    pub fn with_max_iterations(max_iterations: usize) -> Self {
        Self {
            visited: HashSet::new(),
            loop_iterations: HashMap::new(),
            max_iterations,
        }
    }

    /// Check if a node should be visited.
    ///
    /// For loop-back visits:
    /// - Allows re-visiting but tracks iteration count
    /// - Returns false if max iterations exceeded
    ///
    /// For normal visits:
    /// - Returns false if already visited
    ///
    /// If this returns true, the node is marked as visited.
    pub fn should_visit(&mut self, node_id: &str, via_loop_back: bool) -> bool {
        if via_loop_back {
            let count = self.loop_iterations.entry(node_id.to_string()).or_insert(0);
            *count += 1;
            if *count > self.max_iterations {
                warn!(
                    node_id = %node_id,
                    iterations = *count,
                    max = self.max_iterations,
                    "loop exceeded max iterations, terminating"
                );
                return false;
            }
            // For loop-back visits, we DO mark as visited but allow future loop-back visits
            self.visited.insert(node_id.to_string());
            true
        } else {
            // Normal visit - only visit once
            if self.visited.contains(node_id) {
                return false;
            }
            self.visited.insert(node_id.to_string());
            true
        }
    }

    /// Check if a node has been visited (not including loop-back context).
    pub fn was_visited(&self, node_id: &str) -> bool {
        self.visited.contains(node_id)
    }

    /// Reset the traversal state.
    pub fn reset(&mut self) {
        self.visited.clear();
        self.loop_iterations.clear();
    }
}

/// Entry in the BFS work queue.
#[derive(Debug, Clone)]
pub struct WorkQueueEntry<T> {
    /// Node ID to process.
    pub node_id: String,
    /// Whether this node was reached via a loop-back edge.
    pub via_loop_back: bool,
    /// Additional data to carry through traversal.
    pub data: T,
}

impl<T> WorkQueueEntry<T> {
    pub fn new(node_id: String, via_loop_back: bool, data: T) -> Self {
        Self {
            node_id,
            via_loop_back,
            data,
        }
    }
}

/// A BFS traversal queue that is loop-aware.
///
/// This combines the work queue with traversal state to provide
/// a unified interface for loop-aware DAG traversal.
pub struct TraversalQueue<T> {
    /// The underlying work queue.
    queue: VecDeque<WorkQueueEntry<T>>,
    /// Traversal state tracking visited nodes and loop iterations.
    state: LoopAwareTraversal,
}

impl<T> Default for TraversalQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TraversalQueue<T> {
    /// Create a new empty traversal queue.
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            state: LoopAwareTraversal::new(),
        }
    }

    /// Create a traversal queue with custom max loop iterations.
    pub fn with_max_iterations(max_iterations: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            state: LoopAwareTraversal::with_max_iterations(max_iterations),
        }
    }

    /// Push an entry to the back of the queue.
    pub fn push(&mut self, entry: WorkQueueEntry<T>) {
        self.queue.push_back(entry);
    }

    /// Push a node with data to the queue.
    pub fn push_node(&mut self, node_id: String, via_loop_back: bool, data: T) {
        self.push(WorkQueueEntry::new(node_id, via_loop_back, data));
    }

    /// Pop the next entry from the queue, checking visit status.
    ///
    /// Returns `None` if the queue is empty.
    /// Skips nodes that shouldn't be visited (already visited non-loop-back,
    /// or exceeded loop iteration limit).
    pub fn pop_next(&mut self) -> Option<WorkQueueEntry<T>> {
        while let Some(entry) = self.queue.pop_front() {
            if self.state.should_visit(&entry.node_id, entry.via_loop_back) {
                return Some(entry);
            }
        }
        None
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get a reference to the traversal state.
    pub fn state(&self) -> &LoopAwareTraversal {
        &self.state
    }

    /// Get a mutable reference to the traversal state.
    pub fn state_mut(&mut self) -> &mut LoopAwareTraversal {
        &mut self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_aware_traversal_normal_visit() {
        let mut traversal = LoopAwareTraversal::new();

        // First visit should succeed
        assert!(traversal.should_visit("node_1", false));

        // Second visit (non-loop-back) should fail
        assert!(!traversal.should_visit("node_1", false));

        // Different node should succeed
        assert!(traversal.should_visit("node_2", false));
    }

    #[test]
    fn test_loop_aware_traversal_loop_back_visit() {
        let mut traversal = LoopAwareTraversal::with_max_iterations(3);

        // First visit via loop-back
        assert!(traversal.should_visit("loop_node", true));

        // Second visit via loop-back (iteration 2)
        assert!(traversal.should_visit("loop_node", true));

        // Third visit via loop-back (iteration 3)
        assert!(traversal.should_visit("loop_node", true));

        // Fourth visit should exceed max
        assert!(!traversal.should_visit("loop_node", true));
    }

    #[test]
    fn test_select_guarded_edges_no_guards() {
        let edges = vec![
            TraversalEdge {
                target: "node_1".to_string(),
                is_loop_back: false,
                is_else: false,
                guard_expr: None,
                condition: None,
            },
            TraversalEdge {
                target: "node_2".to_string(),
                is_loop_back: false,
                is_else: false,
                guard_expr: None,
                condition: None,
            },
        ];

        let scope = InlineScope::new();
        let mut errors = Vec::new();

        let selected = select_guarded_edges(edges, &scope, &mut errors);

        assert_eq!(selected.len(), 2);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_traversal_queue_basic() {
        let mut queue: TraversalQueue<i32> = TraversalQueue::new();

        queue.push_node("node_1".to_string(), false, 1);
        queue.push_node("node_2".to_string(), false, 2);
        queue.push_node("node_1".to_string(), false, 3); // Duplicate, should be skipped

        let entry1 = queue.pop_next().unwrap();
        assert_eq!(entry1.node_id, "node_1");
        assert_eq!(entry1.data, 1);

        let entry2 = queue.pop_next().unwrap();
        assert_eq!(entry2.node_id, "node_2");
        assert_eq!(entry2.data, 2);

        // node_1 duplicate should be skipped
        assert!(queue.pop_next().is_none());
    }

    #[test]
    fn test_traversal_queue_loop_back() {
        let mut queue: TraversalQueue<i32> = TraversalQueue::with_max_iterations(2);

        // Push same node multiple times via loop-back
        queue.push_node("loop_node".to_string(), true, 1);
        queue.push_node("loop_node".to_string(), true, 2);
        queue.push_node("loop_node".to_string(), true, 3); // Should exceed max

        let entry1 = queue.pop_next().unwrap();
        assert_eq!(entry1.data, 1);

        let entry2 = queue.pop_next().unwrap();
        assert_eq!(entry2.data, 2);

        // Third should be skipped due to max iterations
        assert!(queue.pop_next().is_none());
    }
}
