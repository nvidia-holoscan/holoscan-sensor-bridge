/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_TEST_SERVICES_HPP
#define HOLOLINK_MODULE_TEST_SERVICES_HPP

#include "hololink/module/service.hpp"

namespace hololink::module::test {

class HostTestServiceV1
    : public Service<HostTestServiceV1> {
public:
    static constexpr const char* type_id = "hololink_module_test.host.v1";
    static constexpr const char* release_version = TEST_ONLY;
    virtual ~HostTestServiceV1() = default;
    virtual int answer() const = 0;
};

class TestServiceV1
    : public Service<TestServiceV1> {
public:
    static constexpr const char* type_id = "hololink_module_test.module.v1";
    static constexpr const char* release_version = TEST_ONLY;
    virtual ~TestServiceV1() = default;
    virtual int answer() const = 0;
};

} // namespace hololink::module::test

#endif /* HOLOLINK_MODULE_TEST_SERVICES_HPP */
