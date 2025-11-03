from langchain.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import ChatVertexAI
import json, logging
from langchain_core.messages import HumanMessage


class Gemini:
    def __init__(
        self,
        gemini_model=None,
        temperature=None,
        max_output_tokens=None,
        response_schema=None,
        max_retries=None,
        prompt_text=None,
        prompt_image=None,
    ):
        self.gemini_model = gemini_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.response_schema = response_schema
        self.max_retries = max_retries
        self.prompt_text = prompt_text
        self.prompt_image = prompt_image
        self.prompt = None
        self.summarize_chain_text = None
        self.summarize_chain_image = None
        if prompt_text:
            self.prompt = PromptTemplate.from_template(prompt_text)
            self._create_sumarized_chain_text()
        if prompt_image:
            self._create_sumarized_chain_image()

    def _create_sumarized_chain_text(self):

        self.summarize_chain_text = self.prompt | VertexAI(
            model_name=self.gemini_model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_schema=self.response_schema,
            response_mime_type="application/json",
            max_retries=self.max_retries,
        )

    def _create_sumarized_chain_image(self):
        self.summarize_chain_image = ChatVertexAI(
            model=self.gemini_model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_schema=self.response_schema,
            response_mime_type="application/json",
            max_retries=self.max_retries,
        )

    def _generate_text_summaries(self, content):
        import re, logging, json

        def clean_text(text):
            if text and len(text.split()) > 3:
                return re.sub(r"\u2022", "", text).strip()
            return None

        content = clean_text(content)
        if not content or content == "":
            return None
        try:
            logging.info(f"Generating summary for text {content[0:10]}...")
            response = self.summarize_chain_text.invoke({"element": content})
            json.loads(response)
            return response
        except json.JSONDecodeError as e:
            logging.info(f"JSON decode error: {e}. Retrying...")
            return None

    def _genenate_image_summaries(self, content):
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{content}"},
        }
        text_message = {
            "type": "text",
            "text": self.prompt_image,
        }
        message = HumanMessage(content=[image_message, text_message])
        response = self.summarize_chain_image.invoke([message])
        if not response.content or response.content.strip() == "":
            return None
        try:
            json.loads(response.content)
            return response.content
        except json.JSONDecodeError:
            logging.error(
                f"Failed to decode JSON from model response: {response.content}"
            )
            return None
