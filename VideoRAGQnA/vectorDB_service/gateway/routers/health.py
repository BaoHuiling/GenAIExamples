# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from fastapi import APIRouter, Header
# from conf.config import Settings
# from core.common.auth import securedRoute

router = APIRouter()
# settings = Settings()


@router.get("/health", tags=["Health API"], summary="Check API health")
@router.get("/health/", include_in_schema=False)
# @securedRoute
# async def health(authorization: str = Header(default=None)):
async def health():
    """

    **Response**:

    - **status** (string): A string describing health status.
    """
    return {"status": "healthy"}