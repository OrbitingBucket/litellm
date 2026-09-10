use super::super::OcrAdapter;
use crate::Error;
use crate::auth::azure::AzureAuthInputs;
#[cfg(test)]
use crate::auth::azure::AzureAuthService;
use crate::constants::AZURE_AI_OCR_PATH;
use crate::ocr::OcrClient;
use crate::ocr::codecs::mistral::{self, MistralOcrParams, MistralOcrResponse};
use crate::ocr::document::{inline_remote_document, validate_inline_document};
use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
use crate::ocr::prepare::{_prepare_ocr_request, credential_env, transform_request_body};
use crate::ocr::registry::OcrProvider;
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection};
use crate::url_utils::ApiUrl;

const AZURE_AI_API_KEY_ENV: &str = "AZURE_AI_API_KEY";
const AZURE_AI_API_BASE_ENV: &str = "AZURE_AI_API_BASE";

#[derive(Clone, Debug)]
pub(crate) struct AzureMistralAdapter;

impl OcrAdapter for AzureMistralAdapter {
    type ProviderResponse = MistralOcrResponse;
    const PROVIDER: OcrProvider = OcrProvider::AzureAi;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn transform_ocr_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, OcrError> {
        let params: MistralOcrParams = _prepare_ocr_request(request)?;
        let config =
            AzureAuthInputs::from_optional_params(&request.optional_params).map_err(Error::from)?;
        let headers = validate_environment(&request.connection, &config, &credential_env).await?;
        let url = get_complete_url(request.connection.api_base.as_deref(), &credential_env)?;
        let document = inline_remote_document(
            client.document_fetcher(),
            request.document.clone(),
            &request.connection,
        )
        .await?;
        let body = mistral::transform_ocr_request(&request.model, document, &params)?;
        transform_request_body(client, request, &url, &headers, body, |body| {
            validate_inline_document(&body.document)
        })
        .await
    }

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: Self::ProviderResponse,
    ) -> Result<LiteLLMOcrResponse, OcrResponseError> {
        mistral::transform_ocr_response(&request.model, response)
    }
}

fn get_complete_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, OcrError> {
    let base = nonblank(api_base.map(str::to_string))
        .or_else(|| nonblank(env_lookup(AZURE_AI_API_BASE_ENV)))
        .ok_or_else(|| Error::Auth(
            "Missing Azure AI API Base - Set AZURE_AI_API_BASE environment variable or pass api_base parameter".into(),
        ))?;
    let path: Vec<&str> = AZURE_AI_OCR_PATH.trim_matches('/').split('/').collect();
    ApiUrl::parse(&base)
        .and_then(|url| url.complete_path(&path))
        .map(|url| url.into_string())
        .map_err(|_| {
            OcrRequestError::RequestField {
                path: "api_base".into(),
            }
            .into()
        })
}

async fn validate_environment(
    connection: &OcrConnection,
    config: &AzureAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, OcrError> {
    if crate::http_utils::has_header(&connection.extra_headers, "authorization") {
        return Ok(connection.extra_headers.clone());
    }
    if let Some(key) =
        nonblank(connection.api_key.clone()).or_else(|| nonblank(env_lookup(AZURE_AI_API_KEY_ENV)))
    {
        return Ok(bearer_headers(connection, &key));
    }
    let credential = super::resolve_entra(config, env_lookup)
        .await?
        .ok_or(Error::MissingAzureAiCredentials)?;
    Ok(bearer_headers(connection, &credential))
}

#[cfg(test)]
async fn authenticate_with_service(
    connection: &OcrConnection,
    config: &AzureAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    service: &AzureAuthService,
) -> Result<Vec<(String, String)>, OcrError> {
    if crate::http_utils::has_header(&connection.extra_headers, "authorization") {
        return Ok(connection.extra_headers.clone());
    }
    let credential = if let Some(key) =
        nonblank(connection.api_key.clone()).or_else(|| nonblank(env_lookup(AZURE_AI_API_KEY_ENV)))
    {
        key
    } else if let Some(credential) = service
        .resolve(config, env_lookup)
        .await
        .map_err(Error::from)?
    {
        credential.secret().expose().to_string()
    } else {
        return Err(Error::MissingAzureAiCredentials.into());
    };
    Ok(bearer_headers(connection, &credential))
}

fn bearer_headers(connection: &OcrConnection, credential: &str) -> Vec<(String, String)> {
    std::iter::once(("Authorization".into(), format!("Bearer {credential}")))
        .chain(connection.extra_headers.clone())
        .collect()
}

fn nonblank(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use azure_core::http::headers::Headers;
    use azure_core::http::{AsyncRawResponse, HttpClient, Request, StatusCode, Transport};
    use azure_core::{Bytes, Result as AzureResult};

    use super::*;

    #[derive(Debug, Default)]
    struct RecordingTokenClient {
        requests: Mutex<Vec<String>>,
    }

    impl HttpClient for RecordingTokenClient {
        fn execute_request<'life0, 'life1, 'async_trait>(
            &'life0 self,
            request: &'life1 Request,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = AzureResult<AsyncRawResponse>> + Send + 'async_trait>,
        >
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move {
                self.requests.lock().unwrap().push(format!(
                    "{} {}",
                    request.url(),
                    String::from_utf8(Bytes::from(request.body()).to_vec()).unwrap()
                ));
                Ok(AsyncRawResponse::from_bytes(
                    StatusCode::Ok,
                    Headers::new(),
                    r#"{"token_type":"Bearer","expires_in":3600,"ext_expires_in":3600,"access_token":"native-token"}"#,
                ))
            })
        }
    }

    #[test]
    fn completes_azure_path_and_preserves_query() {
        assert_eq!(
            get_complete_url(Some("https://example.com/?tenant=a"), &|_| None).unwrap(),
            "https://example.com/providers/mistral/azure/ocr?tenant=a"
        );
        assert_eq!(
            get_complete_url(
                Some("https://example.com/providers/mistral/azure/ocr"),
                &|_| None
            )
            .unwrap(),
            "https://example.com/providers/mistral/azure/ocr"
        );
    }

    #[tokio::test]
    async fn supplied_authorization_precedes_keys() {
        let connection = OcrConnection {
            api_key: Some("request-key".into()),
            extra_headers: vec![("authorization".into(), "Bearer prepared".into())],
            ..Default::default()
        };
        assert_eq!(
            validate_environment(&connection, &Default::default(), &|_| {
                Some("environment-key".into())
            })
            .await
            .unwrap(),
            connection.extra_headers
        );
    }

    #[tokio::test]
    async fn request_key_precedes_environment_key() {
        let connection = OcrConnection {
            api_key: Some("request-key".into()),
            ..Default::default()
        };
        assert_eq!(
            validate_environment(&connection, &Default::default(), &|_| {
                Some("environment-key".into())
            })
            .await
            .unwrap()[0],
            ("Authorization".into(), "Bearer request-key".into())
        );
    }

    #[tokio::test]
    async fn supplied_entra_token_is_acquired_by_rust() {
        let params = serde_json::json!({"azure_ad_token":"entra-token"});
        let config = AzureAuthInputs::from_optional_params(params.as_object().unwrap()).unwrap();
        let headers = validate_environment(&OcrConnection::default(), &config, &|_| None)
            .await
            .unwrap();
        assert_eq!(
            headers[0],
            ("Authorization".into(), "Bearer entra-token".into())
        );
    }

    #[tokio::test]
    async fn typed_client_secret_reaches_injected_sdk_transport_once() {
        let token_client = Arc::new(RecordingTokenClient::default());
        let service = AzureAuthService::with_transport(4, Transport::new(token_client.clone()));
        let params = serde_json::json!({
            "tenant_id":"tenant",
            "client_id":"client",
            "client_secret":"secret",
            "azure_scope":"https://service.test/.default",
            "azure_authority_host":"https://login.test"
        });
        let config = AzureAuthInputs::from_optional_params(params.as_object().unwrap()).unwrap();

        let headers =
            authenticate_with_service(&OcrConnection::default(), &config, &|_| None, &service)
                .await
                .unwrap();

        assert_eq!(
            headers[0],
            ("Authorization".into(), "Bearer native-token".into())
        );
        let requests = token_client.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].contains("/tenant/oauth2/v2.0/token"));
        assert!(requests[0].contains("client_secret=secret"));
    }

    #[tokio::test]
    async fn oidc_reference_resolves_before_injected_sdk_acquisition() {
        let token_client = Arc::new(RecordingTokenClient::default());
        let service = AzureAuthService::with_transport(4, Transport::new(token_client.clone()));
        let params = serde_json::json!({
            "azure_ad_token":"oidc/env/ASSERTION",
            "tenant_id":"tenant",
            "client_id":"client",
            "azure_authority_host":"https://login.test"
        });
        let config = AzureAuthInputs::from_optional_params(params.as_object().unwrap()).unwrap();

        let headers = authenticate_with_service(
            &OcrConnection::default(),
            &config,
            &|name| (name == "ASSERTION").then(|| "signed-assertion".into()),
            &service,
        )
        .await
        .unwrap();

        assert_eq!(headers[0].1, "Bearer native-token");
        let requests = token_client.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].contains("client_assertion=signed-assertion"));
    }
}
