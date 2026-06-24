"""AWS Bedrock provider."""

import json
import os
import logging

logger = logging.getLogger(__name__)

_NO_STREAM = True


def create_client(proc):
    import boto3
    return boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def call(proc, system, user, response_format=None, max_tokens=None):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens or proc.max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "temperature": proc.temperature,
    })
    resp = proc._call_with_retry(
        lambda: proc.client.invoke_model(
            modelId=proc.model,
            contentType="application/json",
            accept="application/json",
            body=body,
        ),
        lambda r: (0, 0),
        provider_label="Bedrock",
    )
    data = json.loads(resp["body"].read())
    return data["content"][0]["text"].strip()
