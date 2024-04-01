
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(FindPackageHandleStandardArgs)

find_path(ibverbs_INCLUDE_DIRS
  NAMES infiniband/verbs.h
  )

find_library(ibverbs_LIBRARIES
  NAMES ibverbs
  )

find_package_handle_standard_args(ibverbs DEFAULT_MSG
  ibverbs_LIBRARIES ibverbs_INCLUDE_DIRS
  )

mark_as_advanced(ibverbs_INCLUDE_DIRS ibverbs_LIBRARIES)
