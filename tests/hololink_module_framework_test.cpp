/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Loads a stub module via Adapter::load_module, exchanges a no-op
 * service in both directions, and verifies that a deliberately-
 * mismatched ABI stub is rejected and that a missing path throws.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

#include "hololink/module/adapter.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/service.hpp"

// Anonymous namespace holds file-local TYPES only — function-local
// classes / anonymous namespaces are the only tools that give types
// internal linkage. File-local functions and variables use `static`
// at file scope below.
namespace {

class HostTestServiceV1
    : public hololink::module::Service<HostTestServiceV1> {
public:
    static constexpr const char* type_id = "hololink_module_test.host.v1";
    virtual ~HostTestServiceV1() = default;
    virtual int answer() const = 0;
};

class TestServiceV1
    : public hololink::module::Service<TestServiceV1> {
public:
    static constexpr const char* type_id = "hololink_module_test.module.v1";
    virtual ~TestServiceV1() = default;
    virtual int answer() const = 0;
};

class HostTestServiceImpl : public HostTestServiceV1 {
public:
    explicit HostTestServiceImpl(int value)
        : value_(value)
    {
    }
    int answer() const override { return value_; }

private:
    int value_;
};

} // namespace

static constexpr int HOST_ANSWER = 42;

static std::filesystem::path module_path(const char* basename)
{
    // The CMake build sets HOLOLINK_MODULE_TEST_MODULE_DIR via the
    // test environment to the directory holding the stub .so files.
    const char* dir = std::getenv("HOLOLINK_MODULE_TEST_MODULE_DIR");
    if (!dir) {
        return std::filesystem::path(basename);
    }
    return std::filesystem::path(dir) / basename;
}

TEST(HololinkAdapterFramework, LoadStubAndRoundTripService)
{
    using hololink::module::Adapter;
    using hololink::module::ServicePublisher;

    // Publish the host-side service the stub will look up during init.
    auto host_service = std::make_shared<HostTestServiceImpl>(HOST_ANSWER);
    ServicePublisher<HostTestServiceV1>(Adapter::get_adapter().host_publisher())
        .publish("", host_service);

    auto module = Adapter::get_adapter().load_module(
        module_path("hololink_module_stub_module.so"));
    ASSERT_NE(module, nullptr);

    auto stub_service = TestServiceV1::get_service(module, "");
    ASSERT_NE(stub_service, nullptr);
    EXPECT_EQ(stub_service->answer(), HOST_ANSWER);
}

TEST(HololinkAdapterFramework, AbiMismatchModuleIsRejected)
{
    using hololink::module::Adapter;

    EXPECT_THROW(
        Adapter::get_adapter().load_module(
            module_path("hololink_module_abi_mismatch_module.so")),
        std::runtime_error);
}

TEST(HololinkAdapterFramework, MissingModuleIsRejected)
{
    using hololink::module::Adapter;

    EXPECT_THROW(
        Adapter::get_adapter().load_module(module_path("does_not_exist.so")),
        std::runtime_error);
}
