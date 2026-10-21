//! Client-carried multi-agent checkpoints. No request state is retained by the gateway.

use std::{borrow::Cow, sync::OnceLock};

use base64::Engine;
use ring::{
    digest,
    rand::{SecureRandom, SystemRandom},
};

use super::{CheckpointLimits, ValidatedTreeCheckpoint, collaboration::TranscriptSealer};
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::types::{
    agent_tree::StoredTreeSnapshot,
    io::{AgentAttribution, AgentMessage, AgentMessageContent, InOutItem, InputItem, OutputItem},
};
use crate::utils::{
    common::{serialize_to_value, serialized_size_up_to},
    uuid7_str,
};

const PREFIX: &str = "ma_state_v1:";
const DOMAIN: &[u8] = b"agentic-api/multi-agent/replay/v1";
const KEY_ENV: &str = "AGENTIC_MULTI_AGENT_REPLAY_KEY";

impl ValidatedTreeCheckpoint {
    pub(in crate::executor) fn replay_item<'a>(
        &self,
        input: &[InOutItem],
        output: impl Iterator<Item = &'a OutputItem>,
        max_bytes: usize,
    ) -> ExecutorResult<OutputItem> {
        let (count, hash) = history_digest(
            input
                .iter()
                .map(Cow::Borrowed)
                .chain(output.map(|item| Cow::Owned(InOutItem::Output(item.clone())))),
            max_bytes,
        )?;
        let state = (&self.snapshot, count, hash);
        if serialized_size_up_to(&state, max_bytes)
            .map_err(ExecutorError::JsonError)?
            .is_none()
        {
            return Err(invalid("multi-agent replay state exceeds the retained-byte limit"));
        }
        let bytes = serde_json::to_vec(&state).map_err(ExecutorError::JsonError)?;
        let encrypted_content = format!("{PREFIX}{}", replay_sealer()?.seal_bytes(&bytes, DOMAIN)?);
        Ok(OutputItem::AgentMessage(AgentMessage {
            id: uuid7_str("amsg_"),
            author: "/root".into(),
            recipient: "/root".into(),
            agent: Some(AgentAttribution {
                agent_name: "/root".into(),
            }),
            content: vec![AgentMessageContent::EncryptedContent { encrypted_content }],
        }))
    }

    pub(in crate::executor) fn from_replay(input: &[InOutItem], max_bytes: usize) -> ExecutorResult<Option<Self>> {
        for (index, item) in input.iter().enumerate().rev() {
            let Some(item) = item.as_input() else { continue };
            let InputItem::AgentMessage(message) = item.as_ref() else {
                continue;
            };
            let Some(encoded) = message.content.iter().find_map(|part| match part {
                AgentMessageContent::EncryptedContent { encrypted_content } => encrypted_content.strip_prefix(PREFIX),
                _ => None,
            }) else {
                continue;
            };
            if message.author != "/root" || message.recipient != "/root" || message.content.len() != 1 {
                return Err(invalid("invalid multi-agent replay state item"));
            }
            // Bound base64 decoding and JSON allocation before inspecting the authenticated snapshot.
            if encoded.len() > max_bytes.saturating_mul(4) / 3 + 128 {
                return Err(invalid("multi-agent replay state exceeds the retained-byte limit"));
            }
            let bytes = replay_sealer()?.open_bytes(encoded, DOMAIN)?;
            if bytes.len() > max_bytes {
                return Err(invalid("multi-agent replay state exceeds the retained-byte limit"));
            }
            let (snapshot, count, hash): (StoredTreeSnapshot, usize, [u8; 32]) =
                serde_json::from_slice(&bytes).map_err(|_| invalid("invalid multi-agent replay state"))?;
            let start = index
                .checked_sub(count)
                .ok_or_else(|| invalid("incomplete multi-agent replay history"))?;
            let (_, actual) = history_digest(input[start..index].iter().map(Cow::Borrowed), max_bytes)?;
            if actual != hash {
                return Err(invalid("multi-agent replay history does not match its state item"));
            }
            let mut checkpoint = Self::from_stored(snapshot, &CheckpointLimits::for_response(max_bytes))?;
            checkpoint.replayed_items = index + 1;
            return Ok(Some(checkpoint));
        }
        Ok(None)
    }
}

fn history_digest<'a>(
    items: impl Iterator<Item = Cow<'a, InOutItem>>,
    max_bytes: usize,
) -> ExecutorResult<(usize, [u8; 32])> {
    let mut digest = digest::Context::new(&digest::SHA256);
    let mut remaining = max_bytes;
    let mut count = 0;
    for item in items {
        let bytes = serialized_size_up_to(&item, remaining)
            .map_err(ExecutorError::JsonError)?
            .ok_or_else(|| invalid("multi-agent replay history exceeds the retained-byte limit"))?;
        remaining -= bytes;
        // Public input and output structs order fields differently. Hash their
        // canonical JSON representation without model-facing tool conversion.
        let mut value = serialize_to_value(&item).map_err(ExecutorError::JsonError)?;
        value.sort_all_objects();
        digest.update(&serde_json::to_vec(&value).map_err(ExecutorError::JsonError)?);
        digest.update(b"\n");
        count += 1;
    }
    Ok((count, digest.finish().as_ref().try_into().expect("SHA256 is 32 bytes")))
}

/// A deployment key survives requests, not request histories. Configure the same
/// key across replicas and restarts; the default is scoped to this process.
fn replay_sealer() -> ExecutorResult<TranscriptSealer> {
    static KEY: OnceLock<Result<[u8; 32], &'static str>> = OnceLock::new();
    let key = KEY
        .get_or_init(|| match std::env::var(KEY_ENV) {
            Ok(value) => base64::engine::general_purpose::STANDARD
                .decode(value)
                .ok()
                .and_then(|bytes| bytes.try_into().ok())
                .ok_or("AGENTIC_MULTI_AGENT_REPLAY_KEY must be a base64-encoded 32-byte key"),
            Err(std::env::VarError::NotPresent) => {
                let mut key = [0; 32];
                SystemRandom::new()
                    .fill(&mut key)
                    .map_err(|_| "could not generate multi-agent replay key")?;
                Ok(key)
            }
            Err(std::env::VarError::NotUnicode(_)) => Err("AGENTIC_MULTI_AGENT_REPLAY_KEY must be valid base64"),
        })
        .as_ref()
        .map_err(|message| ExecutorError::StreamError((*message).into()))?;
    TranscriptSealer::with_key(key)
}

fn invalid(message: &str) -> ExecutorError {
    ExecutorError::InvalidRequest(message.into())
}
