//! A handler's connection lifecycle: schemas alone, a ready client, or connection on first use.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::OnceCell;

use crate::tool::ToolError;
use crate::tool::mcp::{McpClient, McpClientPool, McpServerEntry};

pub(super) enum McpConnection {
    SpecOnly,
    Ready(Arc<McpClient>),
    Deferred {
        entry: Box<McpServerEntry>,
        client: OnceCell<Arc<McpClient>>,
    },
}

impl McpConnection {
    pub(super) async fn client(&self, server_label: &str) -> Result<&Arc<McpClient>, ToolError> {
        match self {
            Self::SpecOnly => Err(ToolError::Config(
                "MCP tool spec-only handler cannot execute tools".to_owned(),
            )),
            Self::Ready(client) => Ok(client),
            Self::Deferred { entry, client } => {
                client
                    .get_or_try_init(|| async {
                        let pool = McpClientPool::from_config(HashMap::from([(
                            server_label.to_owned(),
                            entry.as_ref().clone(),
                        )]))
                        .await;
                        pool.get(server_label).cloned().ok_or_else(|| {
                            ToolError::Execution(format!(
                                "MCP server '{server_label}' failed to connect: {}",
                                pool.connection_error(server_label)
                                    .unwrap_or("unknown connection error")
                            ))
                        })
                    })
                    .await
            }
        }
    }
}
