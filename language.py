import json

import oci
import os

import gradio as gr
from dotenv import load_dotenv, find_dotenv
from oci.ai_language.models import DominantLanguageDocumentResult, BatchDetectDominantLanguageResult, DetectedLanguage

# read local .env file
_ = load_dotenv(find_dotenv())
compartment_id = os.environ["COMPARTMENT"]  # TODO Specify your compartmentId here

ai_client = oci.ai_language.AIServiceLanguageClient(oci.config.from_file("./.oci/config"))


def language(prompt, history):
    key = "doc"
    doc = oci.ai_language.models.DominantLanguageDocument(key=key, text=prompt)
    documents = [doc]
    batch_detect_dominant_language_details = oci.ai_language.models.BatchDetectDominantLanguageDetails(
        documents=documents, compartment_id=compartment_id)
    output = ai_client.batch_detect_dominant_language(batch_detect_dominant_language_details)
    output_data = output.data
    if isinstance(output_data, BatchDetectDominantLanguageResult):
        output_data_documents = output_data.documents
        for output_data_document in output_data_documents:
            if isinstance(output_data_document, DominantLanguageDocumentResult):
                output_data_document_languages = output_data_document.languages
                dict_result = [instance.__dict__ for instance in output_data_document_languages]
                filtered_result = [{k.strip('_'): v for k, v in d.items() if k in ['_name', '_code', '_score']} for d in
                                   dict_result]
                return json.dumps(filtered_result, indent=4)
    else:
        return json.dumps({"errors": ""}, indent=4)


demo = gr.ChatInterface(fn=language,
                        examples=[
                            "Hello Support Team, I am reaching out to seek help with my credit card number 1234 5678 "
                            "9873 2345 expiring on 11/23. There was a suspicious transaction on 12-Aug-2022 which I "
                            "reported by calling from my mobile number +1 (423) 111-9999 also I emailed from my email "
                            "id sarah.jones1234@hotmail.com. Would you please let me know the refund status? Regards, "
                            "Sarah",
                            "Using high-performance GPU systems in the Oracle Cloud, OCI will be the cloud engine for "
                            "the artificial intelligence models that drive the MIT Driverless cars competing in the "
                            "Indy Autonomous Challenge."],
                        title="(Unofficial) Chat with Cohere Coral")

demo.queue()
if __name__ == "__main__":
    demo.launch()
