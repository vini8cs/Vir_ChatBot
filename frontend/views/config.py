import asyncio

import streamlit as st
from api_client import get_config_api, reset_config_api, update_config_api

GEMINI_MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]

_CONFIG_WIDGET_KEYS = (
    "config_model",
    "config_temperature",
    "config_max_tokens",
    "config_retriever_limit",
    "config_max_retries",
    "config_summarize",
    "prompt_editor",
)


def model_selection(
    config=None, key="config_model", config_field="gemini_model"
):
    current_model = config.get(config_field, "gemini-2.5-flash")
    model_index = (
        GEMINI_MODELS.index(current_model)
        if current_model in GEMINI_MODELS
        else 0
    )
    return st.selectbox(
        "🧬 Gemini Model",
        options=GEMINI_MODELS,
        index=model_index,
        key=key,
    )


def temperature_slider(config=None, key="config_temperature"):
    return st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(config.get("temperature", 0.1)),
        step=0.1,
        key=key,
        help="Higher = more creative, Lower = more focused",
    )


def max_tokens_input(
    config=None, key="config_max_tokens", config_field="max_output_tokens"
):
    return st.number_input(
        "📝 Max Output Tokens",
        min_value=256,
        max_value=8192,
        value=int(config.get(config_field, 2048)),
        step=256,
        key=key,
    )


def retriever_limit_input(config=None, key="config_retriever_limit"):
    return st.number_input(
        "🔍 Retriever Limit (k)",
        min_value=1,
        max_value=20,
        value=int(config.get("retriever_limit", 5)),
        step=1,
        key=key,
        help="Number of documents to retrieve",
    )


def max_retries_input(config=None, key="config_max_retries"):
    return st.number_input(
        "🔄 Max Retries",
        min_value=1,
        max_value=10,
        value=int(config.get("max_retries", 3)),
        step=1,
        key=key,
    )


def summarize_toggle(config=None, key="config_summarize"):
    return st.toggle(
        "📋 Summarize Context",
        value=bool(config.get("summarize", False)),
        key=key,
    )


async def run_save_config(config_changed, updates, key="save_config"):
    if st.button(
        "💾 Save",
        use_container_width=True,
        type="primary" if config_changed else "secondary",
        disabled=not config_changed,
        key=key,
    ):
        result = await update_config_api(updates)
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success("✅ Configuration saved!")
            st.session_state.runtime_config = result.get(
                "config", await get_config_api()
            )
            for widget_key in _CONFIG_WIDGET_KEYS:
                st.session_state.pop(widget_key, None)
            st.rerun()


def run_reset_config(key: str = "reset_config", keys_prefix: str = "config"):
    """Render a Reset button. The on-click schedules the async reset."""
    st.button(
        "🔄 Reset Defaults",
        use_container_width=True,
        on_click=_make_reset_config_callback(keys_prefix),
        key=key,
    )

    error_key = f"{keys_prefix}_reset_error"
    success_key = f"{keys_prefix}_reset_success"

    if st.session_state.get(error_key):
        st.error(f"Error: {st.session_state[error_key]}")
        del st.session_state[error_key]
    if st.session_state.get(success_key):
        st.success("✅ Configuration reset!")
        del st.session_state[success_key]


def _make_reset_config_callback(keys_prefix: str = "config"):
    """Return a synchronous callback that schedules the async reset."""

    def _schedule_reset():
        async def _run():
            result = await reset_config_api(reload_vectorstore=False)
            if "error" in result:
                st.session_state[f"{keys_prefix}_reset_error"] = result[
                    "error"
                ]
                return
            new_config = await get_config_api()
            st.session_state.runtime_config = new_config
            for widget_key in _CONFIG_WIDGET_KEYS:
                st.session_state.pop(widget_key, None)
            st.session_state[f"{keys_prefix}_reset_success"] = True

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_run())
        except RuntimeError:
            asyncio.run(_run())

    return _schedule_reset


async def run_llm_config():
    with st.expander("🔧 LLM Settings", expanded=False):
        config = st.session_state.runtime_config

        if "error" in config:
            st.error(f"Error loading config: {config['error']}")
        else:
            new_model = model_selection(config)
            new_temperature = temperature_slider(config)
            new_max_tokens = max_tokens_input(config)
            new_retriever_limit = retriever_limit_input(config)
            new_max_retries = max_retries_input(config)
            new_summarize = summarize_toggle(config, key="config_summarize")

            st.divider()

            config_changed = (
                new_model != config.get("gemini_model")
                or new_temperature != config.get("temperature")
                or new_max_tokens != config.get("max_output_tokens")
                or new_retriever_limit != config.get("retriever_limit")
                or new_max_retries != config.get("max_retries")
                or new_summarize != config.get("summarize")
            )

            updates = {
                "gemini_model": new_model,
                "temperature": new_temperature,
                "max_output_tokens": new_max_tokens,
                "retriever_limit": new_retriever_limit,
                "max_retries": new_max_retries,
                "summarize": new_summarize,
            }

            col1, col2 = st.columns(2)
            with col1:
                await run_save_config(config_changed, updates)
            with col2:
                run_reset_config()

            if config_changed:
                st.info("💡 You have unsaved changes")


async def run_prompt_config():
    with st.expander("📝 System Prompt", expanded=False):
        config = st.session_state.runtime_config

        if "error" in config:
            st.error(f"Error loading config: {config['error']}")
            return

        current_prompt = config.get("system_prompt", "")
        new_prompt = st.text_area(
            "System Prompt",
            value=current_prompt,
            height=300,
            key="prompt_editor",
            label_visibility="collapsed",
        )

        prompt_changed = new_prompt != current_prompt

        col1, col2 = st.columns(2)
        with col1:
            await run_save_config(
                prompt_changed,
                {"system_prompt": new_prompt},
                key="save_prompt_config",
            )
        with col2:
            run_reset_config(
                key="reset_prompt_config", keys_prefix="prompt_config"
            )

        if prompt_changed:
            st.info("💡 You have unsaved changes")
