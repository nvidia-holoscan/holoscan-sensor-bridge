/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_RELEASE_HPP
#define HOLOLINK_MODULE_RELEASE_HPP

namespace hololink::module {

/**
 * @brief Value of an interface's `release_version` when no release has
 * published it yet.
 *
 * Every interface reachable through the service locator declares, next to
 * its `type_id`:
 *
 * @code
 * static constexpr const char* release_version = ...;
 * @endcode
 *
 * Registering a service that doesn't include a release_version
 * field will lead to a compiler error.  This release version normally
 * includes a string that indicates when that interface was
 * frozen (e.g. "2.7.0"), but two other values are available:
 *
 * - UNRELEASED is provided to circumvent the compiler error
 *      that occurs when no release_version is declared.  This
 *      value is not allowed to appear with any interface definitions
 *      included in an official release of this code.
 * - TEST_ONLY is available for interfaces that are only allowed
 *      to be used in test code; these do not follow the backward
 *      compatible rules that application-facing services do.
 */
inline constexpr const char* UNRELEASED = "unreleased";

/**
 * @brief Value of an interface's `release_version` that marks it as
 * existing only for testing.
 *
 * A `TEST_ONLY` interface exists to exercise the framework itself, for
 * example a stub module's service or a fixture that records what a
 * callback received. Applications and examples must not use one: it may
 * change or be deleted at any time, without notice.
 *
 * `TEST_ONLY` interfaces never appear in `INTERFACES.md`.
 */
inline constexpr const char* TEST_ONLY = "test-only";

} // namespace hololink::module

#endif // HOLOLINK_MODULE_RELEASE_HPP
