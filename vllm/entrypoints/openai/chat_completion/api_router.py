# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.orca_metrics import metrics_header
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def batch_chat(request: Request) -> OpenAIServingChatBatch | None:
    return request.app.state.openai_serving_chat_batch


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


async def _rwkv_session_action(
    action: str, session_id: str | None, raw_request: Request
):
    handler = chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support RWKV sessions")
    try:
        result = await handler.rwkv_session_action(action, session_id)
    except KeyError as error:
        return JSONResponse(
            content={
                "error": {"message": str(error), "type": "not_found", "code": 404}
            },
            status_code=HTTPStatus.NOT_FOUND,
        )
    except NotImplementedError as error:
        return JSONResponse(
            content={
                "error": {
                    "message": str(error),
                    "type": "not_implemented",
                    "code": HTTPStatus.NOT_IMPLEMENTED.value,
                }
            },
            status_code=HTTPStatus.NOT_IMPLEMENTED.value,
        )
    except (RuntimeError, ValueError) as error:
        return JSONResponse(
            content={
                "error": {
                    "message": str(error),
                    "type": "invalid_request_error",
                    "code": HTTPStatus.CONFLICT.value,
                }
            },
            status_code=HTTPStatus.CONFLICT.value,
        )
    except Exception as error:
        # collective_rpc wraps worker-side KeyError in a generic Exception.
        # Preserve the public 404 contract for missing sessions and avoid
        # leaking an ASGI exception to clients for other backend failures.
        message = str(error)
        if "Unknown RWKV session" in message:
            return JSONResponse(
                content={
                    "error": {
                        "message": message,
                        "type": "not_found",
                        "code": HTTPStatus.NOT_FOUND.value,
                    }
                },
                status_code=HTTPStatus.NOT_FOUND.value,
            )
        logger.exception("RWKV session action failed: %s", message)
        return JSONResponse(
            content={
                "error": {
                    "message": "RWKV session operation failed",
                    "type": "internal_server_error",
                    "code": HTTPStatus.INTERNAL_SERVER_ERROR.value,
                }
            },
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )
    return JSONResponse(content=result)


@router.get("/v1/rwkv/sessions/{session_id}")
async def get_rwkv_session(session_id: str, raw_request: Request):
    return await _rwkv_session_action("get", session_id, raw_request)


@router.post("/v1/rwkv/sessions/{session_id}/reset")
async def reset_rwkv_session(session_id: str, raw_request: Request):
    return await _rwkv_session_action("reset", session_id, raw_request)


@router.delete("/v1/rwkv/sessions/{session_id}")
async def delete_rwkv_session(session_id: str, raw_request: Request):
    return await _rwkv_session_action("delete", session_id, raw_request)


@router.post("/v1/rwkv/sessions/evict")
async def evict_rwkv_sessions(raw_request: Request):
    return await _rwkv_session_action("evict", None, raw_request)


@router.post(
    "/v1/chat/completions/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_batch_chat_completion(
    request: BatchChatCompletionRequest, raw_request: Request
):
    handler = batch_chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    result = await handler.create_batch_chat_completion(request, raw_request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result.model_dump())


def attach_router(app: FastAPI):
    app.include_router(router)
