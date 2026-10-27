//! Execute original recorded requests; only inference and external tool backends are mocked.
use super::{CodeInterpreterExecutor, CodeInterpreterRuntimeConfig, FakeProvider};
use crate::executor::engine::upstream_events;
use crate::executor::{ConversationHandler, ExecuteRequest, ExecutionContext, ResponseHandler};
use crate::storage::{ConversationStore, DbPool, ResponseStore, create_pool_with_schema};
use crate::tool::{GatewayExecutors, WebSearchHandler};
use crate::types::io::{InOutItem, OutputItem};
use crate::types::request_response::{RequestPayload, ResponsePayload};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Json, Router, routing::post};
use either::Either;
use futures::StreamExt;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

const DIRECTORY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/store_false");

fn terminal(turn: &Value) -> Value {
    if let Some(body) = turn["response"].get("body") {
        return body.clone();
    }
    turn["response"]["sse"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|raw| raw.as_str().unwrap().lines())
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .find(|event| event["type"] == "response.completed")
        .unwrap()["response"]
        .clone()
}

// These are upstream model responses, never client request input. The executor
// must create the collaboration tree, execute hosted calls and translate outputs.
fn model_item(item: &Value) -> Option<Value> {
    let (name, arguments) = match item["type"].as_str().unwrap() {
        "multi_agent_call" if item["action"] == "wait_agent" => return None,
        "multi_agent_call" => (
            item["action"].as_str().unwrap().to_owned(),
            serde_json::from_str(item["arguments"].as_str().unwrap()).unwrap(),
        ),
        "web_search_call" => (
            "web_search".to_owned(),
            json!({"queries":
            item["action"].get("queries").cloned().unwrap_or_else(|| json!([item["action"]["query"]]))}),
        ),
        "mcp_call" => (
            format!(
                "mcp__{}__{}",
                item["server_label"].as_str().unwrap(),
                item["name"].as_str().unwrap()
            ),
            serde_json::from_str(item["arguments"].as_str().unwrap()).unwrap(),
        ),
        "code_interpreter_call" => ("code_interpreter".to_owned(), json!({"code":item["code"]})),
        "tool_search_call" => ("tool_search".to_owned(), item["arguments"].clone()),
        "function_call" => {
            let mut call = item.clone();
            call.as_object_mut().unwrap().remove("agent");
            if let Some(namespace) = call["namespace"].as_str() {
                call["name"] = json!(format!("agentic_ns__{namespace}__{}", call["name"].as_str().unwrap()));
                call.as_object_mut().unwrap().remove("namespace");
            }
            return Some(call);
        }
        "shell_call" | "custom_tool_call" => {
            let input: InOutItem = serde_json::from_value(item.clone()).unwrap();
            let mut call = serde_json::to_value(InOutItem::into_input_items(vec![input]).remove(0)).unwrap();
            call.as_object_mut().unwrap().remove("agent");
            return Some(call);
        }
        "message" if item["phase"] == "final_answer" => {
            let mut message = item.clone();
            message.as_object_mut().unwrap().remove("agent");
            return Some(message);
        }
        _ => return None,
    };
    Some(json!({"type":"function_call", "id":item["id"],
        "call_id":item.get("call_id").unwrap_or(&item["id"]),
        "name":name, "arguments":arguments.to_string(), "status":"completed"}))
}

struct Backends {
    url: String,
    model: Arc<Mutex<HashMap<String, VecDeque<Vec<Value>>>>>,
    task: JoinHandle<()>,
}

fn assert_loaded_tools(request: &Value) {
    let input = request["input"].as_array().unwrap();
    for call in input
        .iter()
        .filter(|item| item["type"] == "function_call" && item["name"] == "tool_search")
    {
        let Some(output) = input
            .iter()
            .find(|item| item["type"] == "function_call_output" && item["call_id"] == call["call_id"])
        else {
            continue;
        };
        let loaded: Value = serde_json::from_str(output["output"].as_str().unwrap()).unwrap();
        for tool in loaded["tools"].as_array().unwrap() {
            let name = match tool["type"].as_str().unwrap() {
                "function" => tool["name"].as_str().unwrap(),
                "namespace" => {
                    assert_eq!(tool["name"], "travel");
                    "agentic_ns__travel__get_timezone"
                }
                other => panic!("unexpected loaded tool {other}"),
            };
            assert!(
                request["tools"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|tool| tool["name"] == name),
                "loaded tool {name} must remain callable in every subsequent inference request"
            );
        }
    }
}

impl Drop for Backends {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Backends {
    async fn start(turns: &[Value], store: bool) -> Self {
        let mut scripts = HashMap::<String, VecDeque<Vec<Value>>>::new();
        let mut mcp_tools = Vec::<Value>::new();
        let mut mcp_results = Vec::new();
        for turn in turns {
            let mut rounds = HashMap::<String, Vec<Vec<Value>>>::new();
            for item in terminal(turn)["output"].as_array().unwrap() {
                if let Some(output) = model_item(item) {
                    let rounds = rounds
                        .entry(item["agent"]["agent_name"].as_str().unwrap().to_owned())
                        .or_default();
                    if output["name"] == "tool_search"
                        && let Some(round) = rounds
                            .last_mut()
                            .filter(|round| round.last().is_some_and(|item| item["name"] == "tool_search"))
                    {
                        round.push(output);
                    } else {
                        rounds.push(vec![output]);
                    }
                }
                if item["type"] == "mcp_list_tools" {
                    for mut tool in item["tools"].as_array().unwrap().clone() {
                        tool["inputSchema"] = tool["input_schema"].take();
                        tool.as_object_mut().unwrap().remove("input_schema");
                        if !mcp_tools.iter().any(|existing| existing["name"] == tool["name"]) {
                            mcp_tools.push(tool);
                        }
                    }
                }
                if item["type"] == "mcp_call" {
                    mcp_results.push(item.clone());
                }
            }
            for (owner, rounds) in rounds {
                scripts.entry(owner).or_default().extend(rounds);
            }
        }
        let model = Arc::new(Mutex::new(scripts));
        let queues = model.clone();
        let router = Router::new()
            .route(
                "/v1/responses",
                post(move |Json(request): Json<Value>| {
                    let queues = queues.clone();
                    async move {
                        if store {
                            assert!(request.get("store").is_none());
                        } else {
                            assert_eq!(request["store"], false);
                        }
                        assert!(request.get("previous_response_id").is_none());
                        assert!(request.get("conversation").is_none());
                        let guidance = request["input"].as_array().unwrap().last().unwrap()["content"]
                            .as_str()
                            .unwrap();
                        let owner = guidance.strip_prefix("You are `").unwrap().split('`').next().unwrap();
                        assert_loaded_tools(&request);
                        let mut queues = queues.lock().await;
                        let queue = queues.get_mut(owner).unwrap();
                        let unfinished = guidance
                            .lines()
                            .any(|line| line.starts_with("/root/") && line.contains(": Active("));
                        let mut output = Vec::new();
                        if owner == "/root"
                            && unfinished
                            && queue.front().is_some_and(|round| round[0]["type"] == "message")
                        {
                            output.push(json!({"type":"function_call", "id":"fc_replay_wait",
                                "call_id":format!("wait_replay_{}", request["input"].as_array().unwrap().len()),
                                "name":"wait_agent", "arguments":"{\"timeout_ms\":10000}", "status":"completed"}));
                        } else {
                            output = queue
                                .pop_front()
                                .unwrap_or_else(|| panic!("unexpected model round for {owner}"));
                        }
                        let payload = json!({"id":"resp_model", "object":"response", "model":request["model"],
                            "created_at":0, "status":"completed", "output":output});
                        if request["stream"] == true {
                            ([("content-type", "text/event-stream")], upstream_events(&payload)).into_response()
                        } else {
                            Json(payload).into_response()
                        }
                    }
                }),
            )
            .route(
                "/v1/search",
                axum::routing::get(|| async {
                    Json(json!({"results":{
                "web":[{"url":"https://example.com", "title":"recorded tool replay", "description":"search result"}],
                "news":[]}}))
                }),
            )
            .merge(Self::mcp_router(mcp_tools, mcp_results));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        Self { url, model, task }
    }

    fn execution_context(&self, pool: Arc<DbPool>) -> Arc<ExecutionContext> {
        let client = Arc::new(reqwest::Client::new());
        let interpreter =
            CodeInterpreterExecutor::with_provider(CodeInterpreterRuntimeConfig::default(), Arc::new(FakeProvider))
                .unwrap();
        let mut exec = ExecutionContext::new(
            ConversationHandler::new(ConversationStore::new(pool.clone())),
            ResponseHandler::new(ResponseStore::new(pool)),
            client.clone(),
            self.url.clone(),
        );
        exec.gateway_executors = GatewayExecutors::with_code_interpreter(interpreter);
        Arc::new(exec.with_gateway_executor(Arc::new(WebSearchHandler::with_api_key(
            client,
            "recorded-search-key".into(),
            &self.url,
        ))))
    }
    fn mcp_router(mcp_tools: Vec<Value>, mcp_results: Vec<Value>) -> Router {
        Router::new().route(
            "/mcp",
            post(move |Json(request): Json<Value>| {
                let tools = mcp_tools.clone();
                let results = mcp_results.clone();
                async move {
                    if request["id"].is_null() {
                        return StatusCode::ACCEPTED.into_response();
                    }
                    let result = match request["method"].as_str().unwrap() {
                        "initialize" => json!({"protocolVersion":"2025-06-18", "capabilities":{"tools":{}},
                            "serverInfo":{"name":"recorded-mcp", "version":"1"}}),
                        "tools/list" => json!({"tools":tools}),
                        "tools/call" => {
                            let item = results
                                .iter()
                                .find(|item| {
                                    item["name"] == request["params"]["name"]
                                        && serde_json::from_str::<Value>(item["arguments"].as_str().unwrap()).unwrap()
                                            == request["params"]["arguments"]
                                })
                                .expect("recorded MCP invocation");
                            let content = serde_json::from_str::<Value>(item["output"].as_str().unwrap())
                                .ok()
                                .filter(Value::is_array)
                                .unwrap_or_else(|| json!([{"type":"text", "text":item["output"]}]));
                            json!({"content":content, "isError":false})
                        }
                        other => panic!("unexpected MCP method {other}"),
                    };
                    Json(json!({"jsonrpc":"2.0", "id":request["id"], "result":result})).into_response()
                }
            }),
        )
    }
}

async fn execute(request: RequestPayload, exec: Arc<ExecutionContext>, discovery_count: usize) -> ResponsePayload {
    let store = request.store;
    let response = match ExecuteRequest::new(request, exec.clone()).run().await.unwrap() {
        Either::Left(response) => response,
        Either::Right(mut stream) => {
            let mut terminal = None;
            while let Some(chunk) = stream.next().await {
                for line in chunk.lines().filter_map(|line| line.strip_prefix("data: ")) {
                    if let Ok(event) = serde_json::from_str::<Value>(line) {
                        assert_ne!(event["type"], "error", "{event}");
                        if event["type"] == "response.completed" {
                            terminal = Some(event["response"].clone());
                        }
                    }
                }
            }
            serde_json::from_value(terminal.expect("terminal response")).unwrap()
        }
    };
    assert_eq!(response.status, "completed");
    assert_eq!(response.standard_fields.store, store);
    assert_eq!(exec.resp_handler.retrieve(&response.id).await.is_ok(), store);
    assert_eq!(
        response
            .output
            .iter()
            .filter(|item| matches!(item, OutputItem::McpListTools(_)))
            .count(),
        discovery_count,
        "equivalent MCP discovery must not be emitted again for restored children"
    );
    response
}

async fn replay(scenario: &str, stream: bool, store: bool) {
    let mode = if stream { "streaming" } else { "nonstreaming" };
    let path = format!("{DIRECTORY}/multi-agent-store-false-gateway-{scenario}-Qwen-Qwen3.8-27B-FP8-{mode}.yaml");
    let capture: Value = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let turns = capture["turns"].as_array().unwrap();
    let backends = Backends::start(turns, store).await;
    let pool = create_pool_with_schema(Some("sqlite::memory:")).await.unwrap();
    let exec = backends.execution_context(pool.clone());
    let mut history = Vec::<Value>::new();
    let mut recorded_history = Vec::<Value>::new();
    let mut kinds = HashSet::new();
    let mut search_calls = HashSet::new();
    let mut previous = None;
    for turn in turns {
        let mut body = turns[0]["request"]["body"].clone();
        body["store"] = json!(store);
        body["stream"] = json!(stream);
        body.as_object_mut().unwrap().remove("previous_response_id");
        if let Some(tools) = body["tools"].as_array_mut() {
            for tool in tools {
                if tool["type"] == "mcp" {
                    tool["server_url"] = json!(format!("{}/mcp", backends.url));
                }
            }
        }
        let input = turn["request"]["body"]["input"].as_array().unwrap();
        assert!(
            input.starts_with(&recorded_history),
            "recorded continuation must retain full history"
        );
        let new_input = &input[recorded_history.len()..];
        history.extend(new_input.iter().cloned());
        recorded_history = input.clone();
        recorded_history.extend(terminal(turn)["output"].as_array().unwrap().iter().cloned());
        body["input"] = if store { json!(new_input) } else { json!(history) };
        if store {
            body["previous_response_id"] = json!(previous);
        }
        let request: RequestPayload = serde_json::from_value(body).unwrap();
        let discovery_count =
            usize::from(scenario == "mixed-tools" && history.iter().all(|item| item["type"] != "mcp_list_tools"));
        let response = execute(request, exec.clone(), discovery_count).await;
        previous = Some(response.id.clone());
        for item in &response.output {
            let wire = serde_json::to_value(item).unwrap();
            kinds.insert(wire["type"].as_str().unwrap().to_owned());
            if wire["type"] == "tool_search_call" {
                assert!(
                    search_calls.insert(wire["call_id"].as_str().unwrap().to_owned()),
                    "a model search call must be emitted only once"
                );
            }
            if matches!(
                item,
                OutputItem::WebSearchCall(_) | OutputItem::McpCall(_) | OutputItem::CodeInterpreterCall(_)
            ) {
                assert_eq!(wire["status"], "completed");
                assert!(item.agent().is_some());
            }
        }
        history.extend(
            response
                .output
                .into_iter()
                .map(|item| serde_json::to_value(item).unwrap()),
        );
    }
    let required: &[&str] = match scenario {
        "mixed-tools" => &["web_search_call", "mcp_call", "shell_call"],
        "client-owned-tools" => &["function_call", "custom_tool_call", "tool_search_call"],
        "code-interpreter" => &["web_search_call", "code_interpreter_call"],
        _ => unreachable!(),
    };
    for kind in required {
        assert!(kinds.contains(*kind), "missing {kind}");
    }
    assert!(
        backends.model.lock().await.values().all(VecDeque::is_empty),
        "all recorded model actions must run"
    );
    let expected_search_calls: HashSet<_> = turns
        .iter()
        .flat_map(|turn| terminal(turn)["output"].as_array().unwrap().clone())
        .filter(|item| item["type"] == "tool_search_call")
        .map(|item| item["call_id"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        search_calls, expected_search_calls,
        "only actual model search calls may be emitted"
    );
    for table in ["responses", "items", "conversations"] {
        let count: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
            .fetch_one(pool.as_ref())
            .await
            .unwrap();
        if !store {
            assert_eq!(count, 0);
        }
    }
}

#[tokio::test]
async fn recorded_mixed_tools_store_false() {
    for stream in [false, true] {
        tokio::time::timeout(Duration::from_secs(60), replay("mixed-tools", stream, false))
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn recorded_mixed_tools_stored_continuation_suppresses_discovery() {
    for stream in [false, true] {
        tokio::time::timeout(Duration::from_secs(60), replay("mixed-tools", stream, true))
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn recorded_client_owned_tools_store_false() {
    for stream in [false, true] {
        tokio::time::timeout(Duration::from_secs(60), replay("client-owned-tools", stream, false))
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn recorded_code_interpreter_store_false() {
    for stream in [false, true] {
        tokio::time::timeout(Duration::from_secs(60), replay("code-interpreter", stream, false))
            .await
            .unwrap();
    }
}
