# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING

__all__ = ["app"]

if TYPE_CHECKING:  # pragma: no cover
    from .app import app as _app


def __getattr__(name: str):  # pragma: no cover - simple lazy import
    if name == "app":
        from .app import app as fastapi_app

        return fastapi_app
    raise AttributeError(name)
