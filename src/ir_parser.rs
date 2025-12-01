//! IR types for Rappel workflows.
//!
//! This module re-exports the protobuf-generated IR types. These are the native
//! Rust representations that are sent over the wire from Python.

pub mod proto {
    tonic::include_proto!("rappel.ir");
}

// Re-export all IR types at the module level for convenience
pub use proto::*;

/// Parse protobuf bytes into a Workflow.
pub fn parse_workflow_bytes(bytes: &[u8]) -> Result<Workflow, prost::DecodeError> {
    use prost::Message;
    Workflow::decode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use std::collections::HashMap;

    fn make_location(line: u32, col: u32) -> SourceLocation {
        SourceLocation {
            lineno: line,
            col_offset: col,
            end_lineno: Some(line),
            end_col_offset: Some(col + 10),
        }
    }

    fn make_action_call(name: &str, target: Option<&str>) -> ActionCall {
        ActionCall {
            action: name.to_string(),
            module: Some("test_module".to_string()),
            kwargs: HashMap::from([("arg1".to_string(), "value1".to_string())]),
            target: target.map(|s| s.to_string()),
            config: None,
            location: Some(make_location(1, 0)),
        }
    }

    #[test]
    fn test_action_call_structure() {
        let action = make_action_call("test_action", Some("result"));

        assert_eq!(action.action, "test_action");
        assert_eq!(action.module, Some("test_module".to_string()));
        assert_eq!(action.target, Some("result".to_string()));
        assert_eq!(action.kwargs.get("arg1"), Some(&"value1".to_string()));
        assert!(action.location.is_some());
    }

    #[test]
    fn test_gather_with_actions() {
        let gather = Gather {
            calls: vec![
                GatherCall {
                    kind: Some(gather_call::Kind::Action(make_action_call("action1", None))),
                },
                GatherCall {
                    kind: Some(gather_call::Kind::Action(make_action_call("action2", None))),
                },
            ],
            target: Some("results".to_string()),
            location: Some(make_location(1, 0)),
        };

        assert_eq!(gather.calls.len(), 2);
        assert_eq!(gather.target, Some("results".to_string()));

        match &gather.calls[0].kind {
            Some(gather_call::Kind::Action(a)) => assert_eq!(a.action, "action1"),
            _ => panic!("Expected Action"),
        }
        match &gather.calls[1].kind {
            Some(gather_call::Kind::Action(a)) => assert_eq!(a.action, "action2"),
            _ => panic!("Expected Action"),
        }
    }

    #[test]
    fn test_gather_with_subgraph() {
        let gather = Gather {
            calls: vec![
                GatherCall {
                    kind: Some(gather_call::Kind::Action(make_action_call("action1", None))),
                },
                GatherCall {
                    kind: Some(gather_call::Kind::Subgraph(SubgraphCall {
                        method_name: "process_data".to_string(),
                        kwargs: HashMap::from([("value".to_string(), "x".to_string())]),
                        target: None,
                        location: Some(make_location(2, 0)),
                    })),
                },
            ],
            target: Some("results".to_string()),
            location: Some(make_location(1, 0)),
        };

        assert_eq!(gather.calls.len(), 2);

        match &gather.calls[0].kind {
            Some(gather_call::Kind::Action(a)) => assert_eq!(a.action, "action1"),
            _ => panic!("Expected Action"),
        }
        match &gather.calls[1].kind {
            Some(gather_call::Kind::Subgraph(s)) => {
                assert_eq!(s.method_name, "process_data");
                assert_eq!(s.kwargs.get("value"), Some(&"x".to_string()));
            }
            _ => panic!("Expected Subgraph"),
        }
    }

    #[test]
    fn test_workflow_structure() {
        let workflow = Workflow {
            name: "TestWorkflow".to_string(),
            params: vec![WorkflowParam {
                name: "x".to_string(),
                type_annotation: Some("int".to_string()),
            }],
            body: vec![
                Statement {
                    kind: Some(statement::Kind::ActionCall(make_action_call(
                        "fetch_data",
                        Some("data"),
                    ))),
                },
                Statement {
                    kind: Some(statement::Kind::ReturnStmt(Return {
                        value: Some(r#return::Value::Expr("data".to_string())),
                        location: Some(make_location(3, 0)),
                    })),
                },
            ],
            return_type: Some("str".to_string()),
        };

        assert_eq!(workflow.name, "TestWorkflow");
        assert_eq!(workflow.params.len(), 1);
        assert_eq!(workflow.params[0].name, "x");
        assert_eq!(workflow.body.len(), 2);
        assert_eq!(workflow.return_type, Some("str".to_string()));

        match &workflow.body[0].kind {
            Some(statement::Kind::ActionCall(a)) => assert_eq!(a.action, "fetch_data"),
            _ => panic!("Expected ActionCall"),
        }
        match &workflow.body[1].kind {
            Some(statement::Kind::ReturnStmt(r)) => {
                assert_eq!(r.value, Some(r#return::Value::Expr("data".to_string())));
            }
            _ => panic!("Expected Return"),
        }
    }

    #[test]
    fn test_conditional() {
        let cond = Conditional {
            branches: vec![
                Branch {
                    guard: "x > 0".to_string(),
                    preamble: vec![],
                    actions: vec![make_action_call("positive_action", Some("result"))],
                    postamble: vec![],
                    location: Some(make_location(1, 0)),
                },
                Branch {
                    guard: "True".to_string(),
                    preamble: vec![],
                    actions: vec![make_action_call("negative_action", Some("result"))],
                    postamble: vec![],
                    location: Some(make_location(3, 0)),
                },
            ],
            target: Some("result".to_string()),
            location: Some(make_location(1, 0)),
        };

        assert_eq!(cond.branches.len(), 2);
        assert_eq!(cond.branches[0].guard, "x > 0");
        assert_eq!(cond.branches[1].guard, "True");
        assert_eq!(cond.target, Some("result".to_string()));
    }

    #[test]
    fn test_loop() {
        // Loop body now contains statements in sequence
        let loop_ = Loop {
            iterator_expr: "items".to_string(),
            loop_var: "item".to_string(),
            accumulator: "results".to_string(),
            body: vec![
                // Preamble as a PythonBlock statement
                Statement {
                    kind: Some(statement::Kind::PythonBlock(PythonBlock {
                        code: "x = item * 2".to_string(),
                        imports: vec![],
                        definitions: vec![],
                        inputs: vec!["item".to_string()],
                        outputs: vec!["x".to_string()],
                        location: None,
                    })),
                },
                // Action as a statement
                Statement {
                    kind: Some(statement::Kind::ActionCall(make_action_call("process", Some("processed")))),
                },
                // Append as a PythonBlock statement
                Statement {
                    kind: Some(statement::Kind::PythonBlock(PythonBlock {
                        code: "results.append(processed)".to_string(),
                        imports: vec![],
                        definitions: vec![],
                        inputs: vec!["processed".to_string(), "results".to_string()],
                        outputs: vec![],
                        location: None,
                    })),
                },
            ],
            location: Some(make_location(1, 0)),
        };

        assert_eq!(loop_.iterator_expr, "items");
        assert_eq!(loop_.loop_var, "item");
        assert_eq!(loop_.accumulator, "results");
        assert_eq!(loop_.body.len(), 3);  // preamble + action + append
    }

    #[test]
    fn test_roundtrip_workflow_bytes() {
        let workflow = Workflow {
            name: "RoundtripTest".to_string(),
            params: vec![],
            body: vec![Statement {
                kind: Some(statement::Kind::ActionCall(make_action_call(
                    "test_action",
                    Some("result"),
                ))),
            }],
            return_type: None,
        };

        let bytes = workflow.encode_to_vec();
        let parsed = parse_workflow_bytes(&bytes).unwrap();

        assert_eq!(parsed.name, "RoundtripTest");
        assert_eq!(parsed.body.len(), 1);
    }
}
