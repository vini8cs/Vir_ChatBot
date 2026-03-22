import logging

from fastapi import APIRouter, HTTPException

from agents.vir_chatbot.tasks import app as celery_app

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}/status")
async def get_task_status(task_id: str):
    """
    Get the status and progress of a Celery task.
    """
    try:

        def _handle_progress():
            info = task_result.info or {}
            return {
                "current": info.get("current", 0),
                "total": info.get("total", 0),
                "percent": info.get("percent", 0),
                "step": info.get("step", ""),
                "details": info.get("details", ""),
            }

        def _handle_success():
            result = task_result.result or {}
            return {"result": result, "percent": 100}

        def _handle_failure():
            return {
                "error": (str(task_result.result) if task_result.result else "Unknown error"),
                "percent": 0,
            }

        def _handle_pending():
            return {"percent": 0, "step": "Waiting", "details": "Task queued..."}

        task_result = celery_app.AsyncResult(task_id)

        response = {
            "task_id": task_id,
            "status": task_result.status,
            "ready": task_result.ready(),
            "successful": task_result.successful() if task_result.ready() else None,
        }

        _handlers = {
            "PROGRESS": _handle_progress,
            "SUCCESS": _handle_success,
            "FAILURE": _handle_failure,
            "PENDING": _handle_pending,
        }

        handler = _handlers.get(task_result.status)
        if handler:
            response.update(handler())

        return response

    except Exception as e:
        logging.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e


@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running or pending Celery task."""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        logging.info(f"Task {task_id} revoked.")
        return {"status": "cancelled", "task_id": task_id}
    except Exception as e:
        logging.error(f"Error cancelling task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Error cancelling task") from e
