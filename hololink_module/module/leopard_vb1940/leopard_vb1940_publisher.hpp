/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_LEOPARD_VB1940_LEOPARD_VB1940_PUBLISHER_HPP
#define HOLOLINK_MODULE_LEOPARD_VB1940_LEOPARD_VB1940_PUBLISHER_HPP

/* The Leopard VB1940 board definition: the leaf Publisher and the service
 * impls it constructs. Module-private — the public surface this board ships
 * is include/hololink/module/leopard_vb1940/leopard_vb1940.hpp. The header
 * exists (rather than the definitions living in module_entry.cpp) because
 * both module/leopard_vb1940/ and module/leopard_vb1940_2510/ build on it;
 * the latter subclasses LeopardVb1940Publisher to swap in the pre-0x2602
 * transports. */

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/leopard_vb1940/leopard_vb1940.hpp"
#include "hololink/module/oscillator.hpp"
#include "hololink/module/publisher.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"

#include "hololink/core/hololink.hpp"
#include "hololink/sensors/camera/vb1940/renesas_bajoran_lite_ts2.hpp"

#include "hololink_default.hpp"
#include "hsb_ip_publisher.hpp"
#include "mipi_dphy/cpnx.hpp"

namespace hololink::module::leopard_vb1940 {

/* LeopardVb1940InterfaceV1 impl. Holds the per-board HololinkV1 and
 * splits the sensor bring-up into two phases:
 *   - expect_sensor(N): called from Vb1940Cam::configure. Adds N to
 *     the expected_sensors_ mask. No hardware write.
 *   - enable_sensor(N): called from Vb1940Cam::start. Validates N is
 *     in expected_sensors_, then commits the cumulative
 *     expected_sensors_ mask to the FPGA sensor-reset register if
 *     the hardware doesn't already match. The two-phase split lets
 *     the board see the full expected-sensors set (across all
 *     stereo channels) before any camera commits to hardware, so we
 *     never write a partial mask the hardware might not honour.
 *
 * Idempotency contract mirrors HsbLiteOscillatorV1::enable: first
 * commit programs the register, subsequent commits where the mask
 * matches no-op, an on_reset listener clears committed_mask_ so a
 * board reset re-commits on the next start(). */
class LeopardVb1940V1
    : public LeopardVb1940InterfaceV1,
      public hololink::module::Service<LeopardVb1940V1> {
public:
    static constexpr const char* type_id = "leopard_vb1940.impl.v1";
    using hololink::module::Service<LeopardVb1940V1>::get_service;
    using hololink::module::Service<LeopardVb1940V1>::for_each_type_id;
    using ServiceAlias = LeopardVb1940InterfaceV1;

    LeopardVb1940V1() = default;

    void configure(const hololink::module::EnumerationMetadata& metadata) override
    {
        const std::string hololink_id
            = hololink::module::HololinkInterfaceV1::locator_id(metadata);
        hololink_ = hololink::module::module_core::HololinkV1::get_service(
            this->module(), hololink_id.c_str());
        hololink_->legacy_access()->on_reset(
            std::make_shared<ResetCommittedMask>(&committed_mask_));
    }

    void expect_sensor(int64_t sensor_number) override
    {
        validate_sensor_number(sensor_number);
        expected_sensors_ |= (1u << sensor_number);
    }

    void enable_sensor(int64_t sensor_number) override
    {
        validate_sensor_number(sensor_number);
        const uint32_t bit = 1u << sensor_number;
        if ((expected_sensors_ & bit) == 0) {
            throw std::runtime_error(
                "While enabling a Leopard VB1940 sensor: sensor "
                + std::to_string(sensor_number)
                + " was not previously expected — call expect_sensor "
                  "before enable_sensor (Vb1940Cam::configure normally "
                  "handles this)");
        }
        if (committed_mask_ == expected_sensors_) {
            return; // Hardware already matches the expected set — no-op.
        }
        const std::vector<uint32_t> reset_reg { SENSOR_RESET_REG };
        if (committed_mask_ == 0) {
            // First commit after a fresh board (or after the on_reset
            // listener cleared the cache): zero the register defensively
            // so the cumulative-mask write that follows is the only
            // sensor-reset transition the hardware sees.
            const std::vector<uint32_t> zero { 0 };
            if (hololink_->write_uint32(reset_reg, zero) != HOLOLINK_MODULE_OK) {
                throw std::runtime_error(
                    "While enabling Leopard VB1940 sensors: failed to "
                    "zero the sensor-reset register");
            }
        }
        const std::vector<uint32_t> mask_value { expected_sensors_ };
        if (hololink_->write_uint32(reset_reg, mask_value) != HOLOLINK_MODULE_OK) {
            throw std::runtime_error(
                "While enabling Leopard VB1940 sensors: failed to "
                "write the cumulative expected-sensor mask");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        committed_mask_ = expected_sensors_;
    }

private:
    // The Leopard VB1940 carrier wires three camera positions
    // (sensor_number = 0, 1, 2); the FPGA's sensor-reset register
    // mirrors those three positions one-bit-per-sensor. Anything
    // outside [0, MAX_SENSOR_NUMBER] is a configuration bug, not a
    // hardware capability we can address.
    static constexpr int64_t MAX_SENSOR_NUMBER = 2;

    static void validate_sensor_number(int64_t sensor_number)
    {
        if (sensor_number < 0 || sensor_number > MAX_SENSOR_NUMBER) {
            throw std::runtime_error(
                "While operating on a Leopard VB1940 sensor: sensor_number "
                + std::to_string(sensor_number)
                + " is out of range; Leopard VB1940 supports sensor_number "
                  "in [0, "
                + std::to_string(MAX_SENSOR_NUMBER) + "]");
        }
    }

    // FPGA register that holds one bit per sensor; a 1 means the
    // sensor is out of reset. The stereo transition 0x8: 0x0 -> 0x3 is
    // written here and nowhere else.
    static constexpr uint32_t SENSOR_RESET_REG = 0x8;

    class ResetCommittedMask : public hololink::Hololink::ResetController {
    public:
        explicit ResetCommittedMask(uint32_t* committed)
            : committed_(committed)
        {
        }
        // Only the committed-mask cache is cleared here.
        // expected_sensors_ persists across board reset — it's the
        // application's stated intent (which sensors the cameras
        // expect), not a mirror of hardware state. The next enable_sensor
        // call will re-commit the cumulative expected mask.
        void reset() override { *committed_ = 0; }

    private:
        uint32_t* committed_;
    };

    std::shared_ptr<hololink::module::module_core::HololinkV1> hololink_;
    // Sensors the application has declared via expect_sensor — the
    // intent, accumulated across every Vb1940Cam::configure call on
    // this board.
    uint32_t expected_sensors_ = 0;
    // The mask currently written to the sensor-reset register. Cleared
    // on board reset so the next enable_sensor re-commits.
    uint32_t committed_mask_ = 0;
};

/* OscillatorInterfaceV1 impl for Leopard VB1940 carriers. On-board
 * Renesas Bajoran TS2 is shared across the board's data planes, so
 * this impl is a per-board singleton: HsbIpPublisher's
 * construct_shared_oscillator publishes the same shared_ptr under every
 * data-plane instance_id for a given serial. That sharing is what
 * makes the in-impl rate cache correct — `enable()` programs the
 * chip on the first call and short-circuits subsequent same-rate
 * calls instead of re-running setup_clock, which would otherwise
 * toggle the shared sensor-reset register 0x8 = 0x30 -> 0x3 and
 * disrupt the sibling channel that's already streaming.
 *
 * Cache is dropped via an on_reset listener registered against the
 * underlying LegacyHololinkAccess, so a post-reset enable()
 * reprograms the chip. */
class LeopardVb1940OscillatorV1
    : public hololink::module::OscillatorInterfaceV1,
      public hololink::module::Service<LeopardVb1940OscillatorV1> {
public:
    static constexpr const char* type_id = "oscillator.leopard_vb1940.impl.v1";
    using hololink::module::Service<LeopardVb1940OscillatorV1>::get_service;
    using hololink::module::Service<LeopardVb1940OscillatorV1>::for_each_type_id;
    using ServiceAlias = hololink::module::OscillatorInterfaceV1;

    LeopardVb1940OscillatorV1() = default;

    void configure(const hololink::module::EnumerationMetadata& metadata) override
    {
        const std::string hololink_id
            = hololink::module::HololinkInterfaceV1::locator_id(metadata);
        hololink_ = hololink::module::module_core::HololinkV1::get_service(
            this->module(), hololink_id.c_str());
        hololink_->legacy_access()->on_reset(
            std::make_shared<ResetClockRate>(&committed_clock_rate_));
    }

    bool enable(uint64_t clocks_per_second) override
    {
        if (committed_clock_rate_.has_value()) {
            // Cache hit (same rate already programmed) → no-op success.
            // Cache miss (different rate already committed) → fail; the
            // on-board clock generator can serve only one rate at a
            // time.
            return *committed_clock_rate_ == clocks_per_second;
        }
        hololink_->legacy_access()->setup_clock(
            hololink::renesas_ts2::DEVICE_CONFIGURATION);
        committed_clock_rate_ = clocks_per_second;
        return true;
    }

    std::map<std::string, std::string> get_caps() override { return {}; }

    bool set_caps(const std::map<std::string, std::string>& /*caps*/) override
    {
        return false;
    }

private:
    class ResetClockRate : public hololink::Hololink::ResetController {
    public:
        explicit ResetClockRate(std::optional<uint64_t>* committed)
            : committed_(committed)
        {
        }
        void reset() override { committed_->reset(); }

    private:
        std::optional<uint64_t>* committed_;
    };

    std::shared_ptr<hololink::module::module_core::HololinkV1> hololink_;
    std::optional<uint64_t> committed_clock_rate_;
};

/* Leopard VB1940 channel configuration.
 *
 * Leopard VB1940-AIO puts all sensors on one physical data plane (0)
 * with one SIF per sensor, and adds a per-sensor camera I2C bus the
 * shared HSB-IP formula doesn't cover. use_sensor stamps the shared
 * module_core::hsb_ip_sensor_metadata fields with that layout, then
 * the i2c_bus.
 *
 * HsbIpPublisher::update_metadata delegates to this same use_sensor
 * for the bootp data_plane, so the enumerate and stereo-clone paths
 * produce identical metadata and the per-sensor formula lives in
 * exactly one place. The config is self-contained — no back-pointer to
 * the publisher. */
class LeopardVb1940ChannelConfigurationV1
    : public hololink::module::module_core::HsbIpChannelConfigurationV1 {
public:
    // Three camera positions on the AIO carrier; the single source of
    // truth for the valid range, shared by use_sensor's bounds check
    // and is_sensor_valid().
    static constexpr int64_t TOTAL_SENSORS = 3;

    bool is_sensor_valid(int64_t sensor_number) const override
    {
        return sensor_number >= 0 && sensor_number < TOTAL_SENSORS;
    }

    void use_sensor(hololink::module::EnumerationMetadata& metadata,
        int64_t sensor_number) override
    {
        if (!is_sensor_valid(sensor_number)) {
            throw std::runtime_error(
                "While selecting a Leopard VB1940 sensor: sensor_number "
                + std::to_string(sensor_number)
                + " is out of range (Leopard VB1940 supports 0.."
                + std::to_string(TOTAL_SENSORS - 1) + ")");
        }
        // One physical data plane (0) shared by every sensor, one SIF
        // per sensor; the SIF / VP addresses come from sensor_number.
        hololink::module::module_core::hsb_ip_sensor_metadata(
            metadata, sensor_number, /*data_plane=*/0, /*sifs_per_sensor=*/1);
        // Per-sensor camera I2C bus — the one field the shared formula
        // doesn't cover.
        constexpr int64_t CAM_I2C_BUS = 1;
        metadata["i2c_bus"] = CAM_I2C_BUS + sensor_number;
    }
};

/* Leopard VB1940 leaf Publisher. Inherits the canonical HSB-IP service
 * chain from module_core::HsbIpPublisher and adds what makes the board
 * itself:
 *   - module_name() -> "leopard_vb1940"
 *   - the single-FPGA CertusPro-NX D-PHY, both halves of the pairing
 *   - construct_oscillator -> LeopardVb1940OscillatorV1 (Bajoran TS2)
 *   - construct_board_services -> LeopardVb1940V1 on the
 *     leopard_vb1940.v1 type_id, plus the D-PHY above
 *   - publish_channel_configuration -> a
 *     LeopardVb1940ChannelConfigurationV1 whose use_sensor stamps the
 *     Leopard layout (data_plane=0, one SIF per sensor, camera i2c_bus)
 *     via the shared hsb_ip_sensor_metadata formula.
 *
 * The per-sensor formula lives only in that ChannelConfiguration:
 * Leopard does NOT override update_metadata — the inherited base
 * delegates to use_sensor for the bootp data_plane, so enumerate-time
 * and stereo-clone metadata are produced by the same routine.
 *
 * Every other canonical service (Hololink, DataChannel,
 * RoceDataChannel, RoceReceiver, Sequencer, I2C) uses the base's
 * default behavior unchanged. */
class LeopardVb1940Publisher
    : public hololink::module::module_core::HsbIpPublisher {
protected:
    /* Single-FPGA CertusPro-NX; pairs with the CPNX layout
     * construct_board_services publishes. */
    hololink_module_status_t update_board_metadata(
        hololink::module::EnumerationMetadata& metadata) override
    {
        hololink::module::module_core::stamp_cpnx_mipi_dphy_metadata(metadata, 60);
        return HOLOLINK_MODULE_OK;
    }

    /* What this board adds to the canonical set: its own supplement and
     * the single-FPGA CertusPro-NX D-PHY. Neither substitutes for a
     * canonical branch, so both belong on the tail hook rather than
     * construct_overrides. */
    bool construct_board_services(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        return construct_leopard_vb1940(instance_id, type_id)
            || hololink::module::module_core::construct_cpnx_mipi_dphy(
                *this, instance_id, type_id);
    }

    virtual bool construct_leopard_vb1940(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<LeopardVb1940V1>(type_id)) {
            return false;
        }
        auto impl = std::make_shared<LeopardVb1940V1>();
        hololink::module::ServicePublisher<LeopardVb1940V1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    std::string module_name() const override { return "leopard_vb1940"; }

    void publish_channel_configuration() override
    {
        auto impl = std::make_shared<LeopardVb1940ChannelConfigurationV1>();
        hololink::module::ServicePublisher<
            hololink::module::module_core::HsbIpChannelConfigurationV1>(
            shared_from_this())
            .publish("", impl);
    }

    /* The Bajoran TS2 is a per-board resource shared across both stereo
     * channels, so the base's share-by-serial helper publishes one impl
     * per serial under every (serial, data_plane) instance_id the
     * framework asks for. The impl's in-class rate cache only works if
     * both data planes resolve to the same shared_ptr. */
    bool construct_oscillator(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        return construct_shared_oscillator<LeopardVb1940OscillatorV1>(
            instance_id, type_id);
    }
};

} // namespace hololink::module::leopard_vb1940

#endif // HOLOLINK_MODULE_LEOPARD_VB1940_LEOPARD_VB1940_PUBLISHER_HPP
