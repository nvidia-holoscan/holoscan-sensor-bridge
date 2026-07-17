/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * IMX274 camera player built on the hololink_module API.
 *
 * Runnable companion to the "IMX274 Module Tutorial" user guide. It plays
 * one IMX274 camera behind a Holoscan Sensor Bridge board through the
 * hardware (RoCE) receiver and displays the video with Holoviz.
 * Configuration is intentionally hard-coded so the tutorial can focus on the
 * API rather than on argument parsing.
 */

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

#include <cuda.h>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "hololink/module/cuda_unique.hpp"

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/operators/csi_to_bayer_op.hpp"
#include "hololink/module/operators/image_processor_op.hpp"
#include "hololink/module/operators/roce_receiver_op.hpp"

#include "hololink/module/sensors/imx274/imx274_cam.hpp"
#include "hololink/module/sensors/imx274/imx274_mode.hpp"

// The board announces itself over bootp; this is the peer IP we wait for.
static const char* const HOLOLINK_IP = "192.168.0.2";
// Seconds to wait for that announcement before giving up.
static constexpr std::chrono::seconds DISCOVERY_TIMEOUT(30);
static constexpr auto CAMERA_MODE
    = hololink::module::sensors::imx274::Imx274_Mode::IMX274_MODE_1920X1080_60FPS;

class HoloscanApplication : public holoscan::Application {
public:
    HoloscanApplication(CUcontext cuda_context, int cuda_device_ordinal,
        hololink::module::EnumerationMetadata metadata,
        std::shared_ptr<hololink::module::sensors::imx274::Imx274Cam> camera)
        : cuda_context_(cuda_context)
        , cuda_device_ordinal_(cuda_device_ordinal)
        , metadata_(std::move(metadata))
        , camera_(std::move(camera))
    {
    }
    HoloscanApplication() = delete;

    void compose() override
    {
        camera_->set_mode(CAMERA_MODE);

        auto csi_to_bayer_pool = make_resource<holoscan::BlockMemoryPool>(
            "csi_to_bayer_pool",
            /*storage_type=*/1,
            /*block_size=*/camera_->width() * sizeof(uint16_t) * camera_->height(),
            /*num_blocks=*/4);
        auto csi_to_bayer_operator = make_operator<hololink::module::operators::CsiToBayerOp>(
            "csi_to_bayer",
            holoscan::Arg("allocator", csi_to_bayer_pool),
            holoscan::Arg("cuda_device_ordinal", cuda_device_ordinal_));
        // Configuring the converter tells us how big each received frame is.
        camera_->configure_converter(csi_to_bayer_operator);

        const size_t frame_size = csi_to_bayer_operator->get_csi_length();
        auto receiver_operator = make_operator<hololink::module::operators::RoceReceiverOp>(
            "receiver",
            holoscan::Arg("enumeration_metadata", metadata_),
            holoscan::Arg("frame_context", cuda_context_),
            holoscan::Arg("frame_size", frame_size),
            holoscan::Arg("device_start", std::function<void()>([this] { camera_->start(); })),
            holoscan::Arg("device_stop", std::function<void()>([this] { camera_->stop(); })));

        const auto bayer_format = camera_->bayer_format();
        const auto pixel_format = camera_->pixel_format();
        auto image_processor_operator = make_operator<hololink::module::operators::ImageProcessorOp>(
            "image_processor",
            // Optical black value for IMX274 is 50.
            holoscan::Arg("optical_black", 50),
            holoscan::Arg("bayer_format", static_cast<int>(bayer_format)),
            holoscan::Arg("pixel_format", static_cast<int>(pixel_format)));

        constexpr uint32_t rgba_components_per_pixel = 4;
        auto bayer_pool = make_resource<holoscan::BlockMemoryPool>(
            "bayer_pool",
            /*storage_type=*/1,
            /*block_size=*/camera_->width() * rgba_components_per_pixel * sizeof(uint16_t) * camera_->height(),
            /*num_blocks=*/4);
        auto demosaic = make_operator<holoscan::ops::BayerDemosaicOp>(
            "demosaic",
            holoscan::Arg("pool", bayer_pool),
            holoscan::Arg("generate_alpha", true),
            holoscan::Arg("alpha_value", 65535),
            holoscan::Arg("bayer_grid_pos", static_cast<int>(bayer_format)),
            holoscan::Arg("interpolation_mode", 0));

        auto visualizer = make_operator<holoscan::ops::HolovizOp>(
            "holoviz",
            holoscan::Arg("framebuffer_srgb", true));

        add_flow(receiver_operator, csi_to_bayer_operator, { { "output", "input" } });
        add_flow(csi_to_bayer_operator, image_processor_operator, { { "output", "input" } });
        add_flow(image_processor_operator, demosaic, { { "output", "receiver" } });
        add_flow(demosaic, visualizer, { { "transmitter", "receivers" } });
    }

private:
    const CUcontext cuda_context_;
    const int cuda_device_ordinal_;
    hololink::module::EnumerationMetadata metadata_;
    std::shared_ptr<hololink::module::sensors::imx274::Imx274Cam> camera_;
};

int main()
{
    HOLOLINK_MODULE_CUDA_CHECK(cuInit(0));
    const int cu_device_ordinal = 0;
    CUdevice cu_device;
    HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGet(&cu_device, cu_device_ordinal));
    CUcontext cu_context;
    HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRetain(&cu_context, cu_device));

    // Find the board via bootp enumeration and get its metadata.
    auto& adapter = hololink::module::Adapter::get_adapter();
    hololink::module::EnumerationMetadata metadata
        = adapter.wait_for_channel(HOLOLINK_IP, DISCOVERY_TIMEOUT);

    // The sensor driver is linked directly into the application; it is not a
    // service. It resolves the services it needs from the enumeration metadata.
    auto camera = std::make_shared<hololink::module::sensors::imx274::Imx274Cam>(
        metadata);

    // The control plane is a versioned service fetched by metadata.
    auto hololink = hololink::module::HololinkInterfaceV1::get_service(metadata);
    // start() opens the control-plane socket; without it, no device I/O works.
    if (hololink->start() != HOLOLINK_MODULE_OK) {
        throw std::runtime_error("HololinkInterface::start failed");
    }
    // The framework leaves device state alone unless the application asks;
    // reset() drives the board into a known state.
    if (hololink->reset() != HOLOLINK_MODULE_OK) {
        throw std::runtime_error("HololinkInterface::reset failed");
    }
    // configure() writes device registers, so it must run after reset().
    camera->configure(CAMERA_MODE);
    camera->set_digital_gain_reg(4);

    auto application = holoscan::make_application<HoloscanApplication>(
        cu_context, cu_device_ordinal, metadata, camera);
    application->run();

    // Stop streaming data and close up sockets.
    hololink->stop();
    HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));
    return 0;
}
