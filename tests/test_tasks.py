"""Tests for agents/vir_chatbot/tasks.py (Celery task functions).

For bind=True tasks, ``task.run`` is already a bound method — the Celery
task instance is ``self`` automatically. We call ``task.run(*args)``
(without a fake self) and patch ``update_task_progress`` at module level
to avoid needing a live broker for state updates.
"""

from unittest.mock import MagicMock, patch

from agents.vir_chatbot.tasks import (
    create_vectorstore_uploaded_pdfs,
    delete_pdfs_from_vectorstore,
    update_task_progress,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_self():
    s = MagicMock()
    s.update_state = MagicMock()
    return s


def _creator(
    has_cache=False,
    has_vectorstore=False,
    diff_raises=None,
):
    c = MagicMock()
    c._check_chache.return_value = has_cache
    c._check_vectorstore_exists.return_value = has_vectorstore
    if diff_raises:
        c._diff_vs_cache.side_effect = diff_raises
    return c


# ---------------------------------------------------------------------------
# update_task_progress (plain function — call with fake self directly)
# ---------------------------------------------------------------------------


class TestUpdateTaskProgress:
    def test_calls_update_state_with_correct_meta(self):
        task = _fake_self()
        update_task_progress(task, 2, 5, "step", "details")
        task.update_state.assert_called_once_with(
            state="PROGRESS",
            meta={
                "current": 2,
                "total": 5,
                "percent": 40,
                "step": "step",
                "details": "details",
            },
        )

    def test_percent_zero_when_total_zero(self):
        task = _fake_self()
        update_task_progress(task, 0, 0, "s")
        meta = task.update_state.call_args[1]["meta"]
        assert meta["percent"] == 0

    def test_default_details_empty_string(self):
        task = _fake_self()
        update_task_progress(task, 1, 2, "step")
        meta = task.update_state.call_args[1]["meta"]
        assert meta["details"] == ""


# ---------------------------------------------------------------------------
# create_vectorstore_uploaded_pdfs
# ---------------------------------------------------------------------------

PATCH_CREATOR = "agents.vir_chatbot.tasks.VectorStoreCreator"
PATCH_PROGRESS = "agents.vir_chatbot.tasks.update_task_progress"


class TestCreateVectorstoreUploadedPdfs:
    def _run(self, creator, pdfs=None, **kwargs):
        with (
            patch(PATCH_CREATOR, return_value=creator),
            patch(PATCH_PROGRESS),
        ):
            return create_vectorstore_uploaded_pdfs.run(
                pdfs or ["/tmp/a.pdf"], **kwargs
            )

    def test_returns_success_on_happy_path(self):
        creator = _creator(has_cache=False, has_vectorstore=False)
        result = self._run(creator)
        assert result["status"] == "Success"
        assert result["percent"] == 100

    def test_builds_new_vectorstore_when_none_exists(self):
        creator = _creator(has_cache=False, has_vectorstore=False)
        self._run(creator)
        creator._save_faiss_vectorstore.assert_called_once()
        creator._adding_chunks_to_vectorstore.assert_not_called()

    def test_adds_to_existing_vectorstore(self):
        creator = _creator(has_cache=False, has_vectorstore=True)
        self._run(creator)
        creator._load_faiss_vectorstore.assert_called_once()
        creator._adding_chunks_to_vectorstore.assert_called_once()
        creator._save_faiss_vectorstore.assert_not_called()

    def test_skips_diff_when_no_cache(self):
        creator = _creator(has_cache=False)
        self._run(creator)
        creator._load_cache.assert_not_called()
        creator._diff_vs_cache.assert_not_called()

    def test_diffs_cache_when_cache_exists(self):
        creator = _creator(has_cache=True)
        self._run(creator)
        creator._load_cache.assert_called_once()
        creator._diff_vs_cache.assert_called_once()

    def test_returns_success_when_no_new_pdfs(self):
        from agents.vir_chatbot.vectorstore import NoNewPDFError

        creator = _creator(has_cache=True, diff_raises=NoNewPDFError())
        result = self._run(creator)
        assert result["status"] == "Success"
        assert "already exist" in result["message"]

    def test_returns_error_on_unexpected_exception(self):
        creator = _creator(has_cache=False)
        creator._start_chunking_process.side_effect = RuntimeError("boom")
        result = self._run(creator)
        assert result["status"] == "Error"
        assert "boom" in result["error"]

    def test_temp_files_removed_in_finally(self, tmp_path):
        f = tmp_path / "upload.pdf"
        f.write_bytes(b"data")
        creator = _creator(has_cache=False, has_vectorstore=False)
        with (
            patch(PATCH_CREATOR, return_value=creator),
            patch(PATCH_PROGRESS),
        ):
            create_vectorstore_uploaded_pdfs.run([str(f)])
        assert not f.exists()

    def test_non_existent_temp_files_skipped_in_finally(self):
        creator = _creator(has_cache=False, has_vectorstore=False)
        result = self._run(creator, pdfs=["/does/not/exist.pdf"])
        assert result["status"] == "Success"


# ---------------------------------------------------------------------------
# delete_pdfs_from_vectorstore
# ---------------------------------------------------------------------------


class TestDeletePdfsFromVectorstore:
    def _run(self, creator, filenames=None):
        with (
            patch(PATCH_CREATOR, return_value=creator),
            patch(PATCH_PROGRESS),
        ):
            return delete_pdfs_from_vectorstore.run(filenames or ["a.pdf"])

    def test_returns_success_on_happy_path(self):
        creator = _creator(has_cache=True, has_vectorstore=True)
        result = self._run(creator)
        assert result["status"] == "Success"
        assert result["percent"] == 100

    def test_returns_error_when_no_cache(self):
        creator = _creator(has_cache=False)
        result = self._run(creator)
        assert result["status"] == "Error"

    def test_returns_error_when_no_vectorstore(self):
        creator = _creator(has_cache=True, has_vectorstore=False)
        result = self._run(creator)
        assert result["status"] == "Error"

    def test_calls_full_pipeline(self):
        creator = _creator(has_cache=True, has_vectorstore=True)
        self._run(creator, filenames=["x.pdf", "y.pdf"])
        creator._load_cache.assert_called_once()
        creator.recover_deleted_pdfs_from_cache.assert_called_once()
        creator._load_faiss_vectorstore.assert_called_once()
        creator.delete_uuids_from_vectorstore.assert_called_once()
        creator._reformat_chache_after_deletion.assert_called_once()
