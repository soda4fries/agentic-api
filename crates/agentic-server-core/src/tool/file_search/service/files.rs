//! File upload, retrieval, and deletion operations.

use super::{
    Arc, CancellationToken, Collection, DeleteObject, FileObject, FilePublicationFailure, FileSearchError,
    FileSearchService, ListParams, ListResponse, LocalFiles, MAX_FILE_BYTES, OwnedSemaphorePermit, decode_content,
    deleted, invalid, page, size_i64, validate_list,
};

impl FileSearchService {
    /// Stores uploaded bytes locally and publishes their SQL metadata afterwards.
    ///
    /// # Errors
    /// Returns validation, resource-limit, or storage errors.
    pub async fn upload_file(
        &self,
        filename: &str,
        content_type: &str,
        purpose: &str,
        bytes: Vec<u8>,
    ) -> Result<FileObject, FileSearchError> {
        let permit = self.permit()?;
        if filename.is_empty() || filename.len() > 255 || filename.chars().any(char::is_control) {
            return invalid("filename must contain 1 to 255 bytes without control characters");
        }
        if !matches!(purpose, "assistants" | "user_data") {
            return invalid("file search accepts purpose assistants or user_data");
        }
        if bytes.is_empty() || bytes.len() > MAX_FILE_BYTES {
            return invalid("file must contain 1 byte to 20 MiB");
        }
        if content_type.is_empty() || content_type.len() > 256 || content_type.chars().any(char::is_control) {
            return invalid("content_type must contain 1 to 256 bytes without control characters");
        }
        let file = FileObject {
            id: format!("file-{}", uuid::Uuid::now_v7()),
            object: "file".into(),
            bytes: size_i64(bytes.len())?,
            created_at: chrono::Utc::now().timestamp(),
            filename: filename.into(),
            purpose: purpose.into(),
            status: "processed".into(),
        };
        let storage = self.storage.clone();
        let files = self.files.clone();
        let content_type = content_type.to_owned();
        let cancelled = CancellationToken::new();
        let _cancel_on_drop = cancelled.clone().drop_guard();
        // The caller joins this operation normally. On caller cancellation it
        // retains its permit and finishes rollback/cleanup without abandoning I/O.
        tokio::spawn(async move {
            let _permit = permit;
            files.publish(&file.id, &bytes, &cancelled).await?;
            match storage.upload(&file, &content_type, &cancelled).await {
                Ok(()) => Ok(file),
                Err(FilePublicationFailure::SafeToRemove(error)) => {
                    files.remove(&file.id).await?;
                    Err(error)
                }
                Err(FilePublicationFailure::Indeterminate(error)) => Err(error.into()),
            }
        })
        .await?
    }

    /// # Errors
    /// Returns pagination validation or storage errors.
    pub async fn list_files(&self, params: &ListParams) -> Result<ListResponse<FileObject>, FileSearchError> {
        validate_list(params)?;
        Ok(page(
            self.storage.list(Collection::Files, None, params).await?,
            params,
            |file: &FileObject| &file.id,
        ))
    }

    /// # Errors
    /// Returns not-found or storage errors.
    pub async fn get_file(&self, id: &str) -> Result<FileObject, FileSearchError> {
        LocalFiles::validate_id(id)?;
        self.storage.file_object(id).await
    }

    /// # Errors
    /// Returns not-found, resource-limit, or storage errors.
    pub async fn file_content(&self, id: &str) -> Result<Vec<u8>, FileSearchError> {
        let permit = self.permit()?;
        LocalFiles::validate_id(id)?;
        let file = self.storage.file(id).await?;
        let object: FileObject = serde_json::from_str(&file.data)?;
        self.read_content(id, object.bytes, file.content_base64, permit).await
    }

    /// Atomically deletes metadata, attachments, and chunks, then removes the local blob.
    ///
    /// # Errors
    /// Returns not-found or storage errors.
    pub async fn delete_file(&self, id: &str) -> Result<DeleteObject, FileSearchError> {
        LocalFiles::validate_id(id)?;
        let permit = self.permit()?;
        let storage = self.storage.clone();
        let files = self.files.clone();
        let id = id.to_owned();
        // Once deletion starts, finish SQL cascading deletion and blob cleanup
        // even if the caller disconnects. A failed SQL commit retains the blob.
        tokio::spawn(async move {
            let _permit = permit;
            let file = storage.file(&id).await?;
            storage.delete(Collection::Files, &id, None).await?;
            if file.content_base64.is_empty() {
                files.remove(&id).await?;
            }
            Ok(deleted(&id, "file.deleted"))
        })
        .await?
    }

    pub(super) async fn read_content(
        &self,
        id: &str,
        expected_bytes: i64,
        legacy: String,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Result<Vec<u8>, FileSearchError> {
        if legacy.is_empty() {
            let files = self.files.clone();
            let id = id.to_owned();
            return tokio::spawn(async move {
                let _permit = permit;
                files.read(&id, expected_bytes, MAX_FILE_BYTES).await
            })
            .await?;
        }
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            decode_content(&legacy)
        })
        .await?
    }
}
