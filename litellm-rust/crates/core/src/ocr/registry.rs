use super::adapters::OcrAdapter;
use crate::Error;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

macro_rules! define_adapter_types {
    ($( $variant:ident, $adapter:ty, $instance:expr, $provider:ident; )+) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum OcrAdapterKind {
            $( $variant, )+
        }

        impl OcrAdapterKind {
            pub(crate) const fn provider(self) -> OcrProvider {
                match self {
                    $( Self::$variant => <$adapter>::PROVIDER, )+
                }
            }
        }
    };
}

super::adapters::for_each_ocr_adapter!(define_adapter_types);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OcrProvider {
    Mistral,
    AzureAi,
    Reducto,
    VertexAi,
}

impl OcrProvider {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Mistral => "mistral",
            Self::AzureAi => "azure_ai",
            Self::Reducto => "reducto",
            Self::VertexAi => "vertex_ai",
        }
    }
}

pub(crate) fn resolve_wire_adapter(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<(String, OcrAdapterKind), Error> {
    let provider =
        get_custom_llm_provider(model, custom_llm_provider).unwrap_or(CustomLlmProvider {
            model,
            custom_llm_provider: OcrProvider::Mistral.as_str(),
        });
    let typed_provider = match provider.custom_llm_provider {
        "mistral" => OcrProvider::Mistral,
        "azure_ai" => OcrProvider::AzureAi,
        "reducto" => OcrProvider::Reducto,
        "vertex_ai" => OcrProvider::VertexAi,
        value => return Err(Error::InvalidProvider(value.to_string())),
    };
    match typed_provider {
        OcrProvider::Mistral => Ok((provider.model.to_string(), OcrAdapterKind::Mistral)),
        OcrProvider::AzureAi if is_document_intelligence_model(provider.model) => Ok((
            provider.model.to_string(),
            OcrAdapterKind::AzureDocumentIntelligence,
        )),
        OcrProvider::AzureAi => Ok((provider.model.to_string(), OcrAdapterKind::AzureMistral)),
        OcrProvider::Reducto if provider.model.eq_ignore_ascii_case("parse-legacy") => {
            Ok((provider.model.to_string(), OcrAdapterKind::ReductoLegacy))
        }
        OcrProvider::Reducto => Ok((provider.model.to_string(), OcrAdapterKind::ReductoV3)),
        OcrProvider::VertexAi if provider.model.to_ascii_lowercase().contains("deepseek") => {
            Err(Error::Unsupported("Vertex DeepSeek OCR"))
        }
        OcrProvider::VertexAi => Ok((provider.model.to_string(), OcrAdapterKind::VertexMistral)),
    }
}

fn is_document_intelligence_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("doc-intelligence") || model.contains("documentintelligence")
}
