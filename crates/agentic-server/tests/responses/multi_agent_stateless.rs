//! Run recorded client turns through the real HTTP gateway with recorded external dependencies.
#[path = "multi_agent_dependencies.rs"]
mod dependencies;

use std::fs::read_to_string;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::common::{spawn_gateway, test_config, test_state};
use crate::multi_agent_contract::{ComparisonPolicy, RecordedSession, assert_multi_agent_contract};
use crate::support;
use agentic_core::events::normalize_sse_line;
use agentic_core::executor::{ConversationHandler, ExecutionContext, ResponseHandler};
use agentic_core::storage::{ConversationStore, ResponseStore, create_pool_with_schema};
use agentic_core::tool::WebSearchHandler;
use agentic_core::types::request_response::ResponsePayload;
use dependencies::Dependencies;
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use tokio::time::timeout;

const DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../agentic-server-core/tests/cassettes/store_false"
);

fn terminal(response: &Value) -> Value {
    if let Some(body) = response.get("body") {
        return body.clone();
    }
    response["sse"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .flat_map(str::lines)
        .filter_map(normalize_sse_line)
        .find(|frame| {
            frame
                .wire
                .rest
                .get("response")
                .is_some_and(|response| response["status"] == "completed")
        })
        .expect("completed response")
        .wire
        .rest["response"]
        .clone()
}

async fn replay_mode(mode: &str) {
    let model = Dependencies::start(&format!("{DIR}/multi-agent-store-false-model-dependencies-{mode}.yaml")).await;
    let search = Dependencies::start(&format!(
        "{DIR}/multi-agent-store-false-search-dependencies-{mode}.yaml"
    ))
    .await;
    let mcp = Dependencies::start(&format!("{DIR}/multi-agent-store-false-mcp-dependencies-{mode}.yaml")).await;
    let pool = create_pool_with_schema(Some("sqlite::memory:")).await.unwrap();
    let client = Arc::new(Client::new());
    let config = test_config(&model.url);
    let mut state = test_state(&config);
    state.exec_ctx = Arc::new(
        ExecutionContext::new(
            ConversationHandler::new(ConversationStore::new(pool.clone())),
            ResponseHandler::new(ResponseStore::new(pool.clone())),
            client.clone(),
            model.url.clone(),
        )
        .with_gateway_executor(Arc::new(WebSearchHandler::with_api_key(
            client.clone(),
            "recorded-search-key".into(),
            &search.url,
        ))),
    );
    let (url, gateway) = spawn_gateway(state).await;
    let capture: Value = serde_yaml::from_str(
        &read_to_string(format!(
            "{DIR}/multi-agent-store-false-gateway-mixed-tools-Qwen-Qwen3.8-27B-FP8-{mode}.yaml"
        ))
        .unwrap(),
    )
    .unwrap();
    let mut actual_turns = Vec::new();
    let mut history = Vec::new();
    let mut recorded_prefix = 0;
    for turn in capture["turns"].as_array().unwrap() {
        let mut request = turn["request"]["body"].clone();
        let recorded_input = request["input"].as_array().unwrap();
        history.extend_from_slice(&recorded_input[recorded_prefix..]);
        recorded_prefix = recorded_input.len() + terminal(&turn["response"])["output"].as_array().unwrap().len();
        request["input"] = json!(history);
        for tool in request["tools"].as_array_mut().unwrap() {
            if tool["type"] == "mcp" {
                tool["server_url"] = json!(format!("{}/openai/tiktoken", mcp.url));
            }
        }
        assert_eq!(request["store"], false);
        assert!(request.get("previous_response_id").is_none());
        let response = client
            .post(format!("{url}/v1/responses"))
            .bearer_auth("test-key")
            .header("OpenAI-Beta", "responses_multi_agent=v1")
            .json(&request)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let text = response.text().await.unwrap();
        model.assert_no_errors().await;
        search.assert_no_errors().await;
        mcp.assert_no_errors().await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let response = if mode == "streaming" {
            json!({"sse": text.split_inclusive('\n').collect::<Vec<_>>()})
        } else {
            json!({"body": serde_json::from_str::<Value>(&text).unwrap()})
        };
        let payload = terminal(&response);
        let typed: ResponsePayload = serde_json::from_value(payload.clone()).unwrap();
        assert_eq!(typed.status, "completed");
        assert_eq!(payload["store"], false);
        let retrieval = client
            .get(format!("{url}/v1/responses/{}", typed.id))
            .bearer_auth("test-key")
            .send()
            .await
            .unwrap();
        assert_eq!(retrieval.status(), StatusCode::NOT_FOUND);
        history.extend_from_slice(payload["output"].as_array().unwrap());
        actual_turns.push(json!({"request":{"path":"/v1/responses","body":request},"response":response}));
    }
    assert_reference_contract(mode, &actual_turns);
    for table in ["responses", "items", "conversations"] {
        let count: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
            .fetch_one(pool.as_ref())
            .await
            .unwrap();
        assert_eq!(count, 0, "store:false wrote {table}");
    }
    model.assert_consumed().await;
    search.assert_consumed().await;
    mcp.assert_consumed().await;
    gateway.abort();
}

fn assert_reference_contract(mode: &str, actual_turns: &[Value]) {
    let cassette: support::Cassette = serde_json::from_value(json!({"turns":actual_turns})).unwrap();
    let actual = RecordedSession::from_cassette(&cassette).unwrap();
    let reference = RecordedSession::load(Path::new(&format!(
        "{DIR}/multi-agent-store-false-openai-reference-mixed-tools-gpt-5.6-sol-{mode}.yaml"
    )))
    .unwrap();
    assert_multi_agent_contract(
        &reference,
        &actual,
        &ComparisonPolicy {
            require_reference_tool_kinds: true,
            minimum_delegated_agents: Some(3),
        },
    )
    .unwrap();
}

#[tokio::test]
async fn mixed_tools_store_false_json_runs_through_gateway() {
    timeout(Duration::from_secs(60), replay_mode("nonstreaming"))
        .await
        .expect("gateway replay timed out");
}

#[tokio::test]
async fn mixed_tools_store_false_sse_runs_through_gateway() {
    timeout(Duration::from_secs(60), replay_mode("streaming"))
        .await
        .expect("gateway replay timed out");
}
