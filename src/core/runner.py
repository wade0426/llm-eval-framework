from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.checkpoint import CheckpointManager
from src.core.conversation_logger import ConversationLogger
from src.core.csv_handler import CsvHandler
from src.core.judge import JudgeEvaluator
from src.core.llm_client import LLMClient, RateLimiter
from src.core.prompt_builder import build_user_prompt
from src.models.config_schema import AppConfig


LOGGER = logging.getLogger("llm_eval")


def run(config: AppConfig) -> None:
    csv_handler = CsvHandler(config)
    checkpoint = CheckpointManager(
        checkpoint_path=config.execution.checkpoint_path,
        input_path=config.dataset.input_path,
    )
    primary_client = LLMClient(
        config.primary_llm,
        max_retries=config.execution.max_retries,
        retry_delay_seconds=config.execution.retry_delay_seconds,
    )
    judge_evaluator = (
        JudgeEvaluator(
            config.judge,
            max_retries=config.execution.max_retries,
            retry_delay_seconds=config.execution.retry_delay_seconds,
        )
        if config.judge.enabled
        else None
    )
    limiter = RateLimiter(config.execution.rate_limit_rpm)
    conversation_logger = ConversationLogger(config.conversation_log)

    df = csv_handler.load()
    processed_indices = checkpoint.load()
    pending_rows = csv_handler.get_pending_rows(df, processed_indices)

    if not pending_rows:
        LOGGER.info("所有資料已處理完畢，無需重跑")
        csv_handler.save(df)
        conversation_logger.close()
        return

    api_error_count = 0
    success_count = 0
    processed_count = 0
    counter_lock = threading.Lock()

    def _process_single_row(index: int) -> None:
        nonlocal api_error_count, success_count, processed_count

        limiter.wait()
        row = df.loc[index]
        user_prompt = build_user_prompt(row, config)

        started = time.perf_counter()
        llm_answer = primary_client.call(
            system_prompt=config.primary_llm.system_prompt,
            user_prompt=user_prompt,
        )
        duration_seconds = time.perf_counter() - started

        csv_handler.update_row(df, index, config.column_mapping.output_column, llm_answer)

        if judge_evaluator is not None:
            expected_answer = str(row.get(config.column_mapping.answer_column, ""))
            judge_result = judge_evaluator.evaluate(
                question=user_prompt,
                llm_answer=llm_answer,
                expected_answer=expected_answer,
            )
            csv_handler.update_row(
                df,
                index,
                config.judge.output_column,
                json.dumps(judge_result, ensure_ascii=False),
            )

        status = "api_error" if llm_answer == "__API_ERROR__" else "success"
        conversation_logger.log(
            index=index,
            system_prompt=config.primary_llm.system_prompt,
            user_prompt=user_prompt,
            llm_response=llm_answer,
            status=status,
            duration_seconds=duration_seconds,
        )

        checkpoint.mark_done(index)

        with counter_lock:
            if llm_answer == "__API_ERROR__":
                api_error_count += 1
            else:
                success_count += 1
            processed_count += 1
            if processed_count % max(1, config.execution.batch_log_interval) == 0:
                LOGGER.info("Progress: %d/%d rows processed", processed_count, len(pending_rows))

    try:
        with ThreadPoolExecutor(max_workers=config.execution.max_workers) as executor:
            futures = {executor.submit(_process_single_row, idx): idx for idx in pending_rows}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    LOGGER.error("Unexpected error processing row %d: %s", index, exc)
    except KeyboardInterrupt:
        LOGGER.warning("KeyboardInterrupt received. Saving partial progress and exiting.")
        csv_handler.save(df)
        conversation_logger.close()
        return

    csv_handler.save(df)
    checkpoint.clear()
    conversation_logger.close()
    LOGGER.info(
        "Execution completed. total=%d, success=%d, api_errors=%d",
        len(pending_rows),
        success_count,
        api_error_count,
    )
