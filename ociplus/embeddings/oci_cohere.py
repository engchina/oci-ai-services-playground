import math
from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env


class OciCohereEmbeddings(BaseModel, Embeddings):
    """Cohere embedding models.

    To use, you should have the ``cohere`` python package installed, and the
    environment variable ``COHERE_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import CohereEmbeddings
            cohere = CohereEmbeddings(
                model="embed-english-light-v2.0", cohere_api_key="my-api-key"
            )
    """

    client: Any  #: :meta private:
    """Cohere client."""
    async_client: Any  #: :meta private:
    """Cohere async client."""
    model: str = "cohere.embed-english-light-v2.0"
    """Model name to use."""

    config: str = "~/.oci/config"
    """Path to the config file. Defaults to ~/.oci/config."""

    profile: str = "DEFAULT"
    """The profile to load from the config file. Defaults to "DEFAULT"."""

    endpoint: str = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"
    """OCI Generative AI endpoint. Defaults to https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com."""

    compartment_id: str
    """Compartment ID to use."""

    truncate: Optional[str] = None
    """Truncate embeddings that are too long from start or end ("NONE"|"START"|"END")"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        print("### in validate_environment ###")
        try:
            import oci

            config = get_from_dict_or_env(
                values, "config", "CONFIG"
            )
            profile = get_from_dict_or_env(
                values, "profile", "PROFILE"
            )
            endpoint = get_from_dict_or_env(
                values, "endpoint", "ENDPOINT"
            )
            config = oci.config.from_file(config, profile)
            values["client"] = oci.generative_ai.GenerativeAiClient(config=config, service_endpoint=endpoint,
                                                                    retry_strategy=oci.retry.NoneRetryStrategy(),
                                                                    timeout=(10, 240))
            # values["async_client"] = oci.generative_ai.GenerativeAiClient(config=config, service_endpoint=endpoint,
            #                                                               retry_strategy=oci.retry.NoneRetryStrategy(),
            #                                                               timeout=(10, 240))
        except ImportError:
            raise ValueError(
                "Could not import cohere python package. "
                "Please install it with `pip install cohere`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Cohere's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        print("### in embed_documents ###")
        import oci
        embed_text_detail = oci.generative_ai.models.EmbedTextDetails()
        embed_text_detail.serving_mode = oci.generative_ai.models.OnDemandServingMode(
            model_id=self.model)
        embed_text_detail.compartment_id = self.compartment_id
        embed_text_detail.truncate = self.truncate
        texts_len = math.ceil((len(texts) / 16))
        completion_embeddings = []
        for i in range(texts_len):
            if i != (texts_len - 1):
                embed_text_detail.inputs = texts[16 * i: 16 * (i + 1)]
            else:
                embed_text_detail.inputs = texts[16 * i: len(texts)]
            embeddings = self.client.embed_text(embed_text_detail)
            completion_embeddings.extend(embeddings.data.embeddings)
        # embeddings = self.client.embed(
        #     model=self.model, texts=texts, truncate=self.truncate
        # ).embeddings
        return [list(map(float, e)) for e in completion_embeddings]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async call out to Cohere's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        print("### in aembed_documents ###")
        embeddings = await self.async_client.embed(
            model=self.model, texts=texts, truncate=self.truncate
        )
        return [list(map(float, e)) for e in embeddings.embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Call out to Cohere's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        print("### in embed_query ###")
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Async call out to Cohere's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        print("### in aembed_query ###")
        embeddings = await self.aembed_documents([text])
        return embeddings[0]
