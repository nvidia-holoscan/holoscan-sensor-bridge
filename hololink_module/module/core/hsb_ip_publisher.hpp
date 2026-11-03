/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_HSB_IP_PUBLISHER_HPP
#define HOLOLINK_MODULE_CORE_HSB_IP_PUBLISHER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "hololink/module/channel_configuration.hpp"
#include "hololink/module/coe_data_channel.hpp"
#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/frame_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/i2c.hpp"
#include "hololink/module/logging.hpp"
#include "hololink/module/name_value_pairs.hpp"
#include "hololink/module/publisher.hpp"
#include "hololink/module/roce_data_channel.hpp"
#include "hololink/module/roce_receiver.hpp"
#include "hololink/module/sequencer.hpp"
#include "hololink/module/status.h"

#include "coe_data_channel_default.hpp"
#include "data_channel_default.hpp"
#include "frame_metadata_default.hpp"
#include "hololink_default.hpp"
#include "i2c_default.hpp"
#include "linux_data_channel_default.hpp"
#include "null_vsync_default.hpp"
#include "ptp_pps_output_default.hpp"
#include "roce_data_channel_default.hpp"
#include "sequencer_default.hpp"

namespace hololink::module::module_core {

/* Canonical HSB IP per-sensor addressing formula, and the single
 * source of truth for it. Stamps the HSB-shape address fields a
 * sensor occupies, given the logical channel (sensor_number), the
 * physical HIF index (data_plane), and how many SIFs each sensor
 * spans (sifs_per_sensor).
 *
 * The HIF register space is per-physical-data-plane, and how sensors
 * map onto data planes is a board property: some boards map each
 * sensor 1:1 onto its own data plane, others put several sensors on
 * one. Both the enumerate-time path (HsbIpPublisher::update_metadata,
 * via the ChannelConfiguration it delegates to) and the application
 * path (ChannelConfigurationInterfaceV1::use_sensor) route through
 * this function, so the two can't drift. In particular hif_address is
 * recomputed on every sensor selection, so switching to a sensor on a
 * different data plane re-points the HIF correctly. */
inline void hsb_ip_sensor_metadata(
    EnumerationMetadata& metadata,
    int64_t sensor_number,
    int64_t data_plane,
    uint32_t sifs_per_sensor)
{
    const uint32_t sensor = sensor_number * sifs_per_sensor;
    const uint32_t vp_mask = 1 << sensor;
    // yes, "sensor_number" not "sensor"
    const uint32_t sif_address = 0x01000000 + 0x10000 * sensor_number;
    const uint32_t vp_address = 0x1000 + 0x40 * sensor;
    const uint32_t hif_address = 0x02000300 + 0x10000 * data_plane;
    metadata["sensor_number"] = sensor_number;
    metadata["sensor"] = sensor;
    metadata["vp_mask"] = vp_mask;
    metadata["sif_address"] = sif_address;
    metadata["vp_address"] = vp_address;
    metadata["hif_address"] = hif_address;
    // Per-sensor channel discriminator: the locator key every
    // per-channel service (DataChannelInterfaceV1,
    // RoceDataChannelInterfaceV1, RoceReceiverInterfaceV1) reads off
    // the metadata. Distinct from `data_plane`, the bootp-side
    // per-data-plane index — on N-sensors-per-data-plane boards the
    // same data_plane carries multiple sensors and only
    // `data_channel` separates them. Single-channel flows that never
    // call Adapter::use_sensor still get a usable default here;
    // multi-channel flows overwrite it by calling use_sensor(n).
    metadata["data_channel"] = sensor_number;
    // Hololink::Event::SIF_0_FRAME_END = 16; SIF_1_FRAME_END = 17.
    switch (sensor_number) {
    case 0:
        metadata["frame_end_event"] = 16; // Hololink::Event::SIF_0_FRAME_END
        break;
    case 1:
        metadata["frame_end_event"] = 17; // Hololink::Event::SIF_1_FRAME_END
        break;
    default:
        // Frame end events are only defined for sensors 0 and 1.
        // Clear any value carried over from a prior sensor selection so
        // re-stamping the same metadata for sensor >1 can't retain a
        // stale frame_end_event.
        metadata.erase("frame_end_event");
        break;
    }
}

/* Base ChannelConfigurationInterfaceV1 impl for HSB-IP-shaped boards.
 * Carries the two board-independent halves of the interface — use_mtu
 * and use_multicast — and holds the canonical type_id every board
 * publishes its configuration under. The sensor layout itself is
 * per-board, so use_sensor / is_sensor_valid stay abstract: a board
 * derives from this, states its sensor count, and routes use_sensor
 * through hsb_ip_sensor_metadata with its own data-plane and
 * SIFs-per-sensor mapping. Implementations are stateless — one shared
 * instance per loaded module .so is enough. */
class HsbIpChannelConfigurationV1
    : public ChannelConfigurationInterfaceV1,
      public Service<HsbIpChannelConfigurationV1> {
public:
    static constexpr const char* type_id = "channel_configuration.module_core.v1";
    using Service<HsbIpChannelConfigurationV1>::get_service;
    using Service<HsbIpChannelConfigurationV1>::for_each_type_id;
    using ServiceAlias = ChannelConfigurationInterfaceV1;

    HsbIpChannelConfigurationV1() = default;

    void use_mtu(EnumerationMetadata& metadata, uint32_t mtu) override
    {
        metadata["mtu"] = static_cast<int64_t>(mtu);
    }

    void use_multicast(EnumerationMetadata& metadata, std::string address,
        uint16_t port) override
    {
        // Same keys the legacy DataChannel reads at configure time;
        // RoceDataChannelV1::configure copies them onto the legacy
        // channel, which programs the FPGA's multicast MAC/IP/port.
        metadata["multicast"] = std::move(address);
        metadata["multicast_port"] = static_cast<int64_t>(port);
    }
};

/* Canonical Publisher for HSB-IP-shaped boards — every board whose
 * control plane, data planes, sequencer and I2C speak the HSB IP
 * register map, whatever carrier they are on. construct_service is a
 * short-circuit OR over construct_overrides (the head hook), one
 * per-service virtual per canonical type_id, then
 * construct_board_services (the tail hook). Both hooks are blank on
 * the base, and which one a board reaches for follows from what it is
 * doing: substituting its own impl for a service the chain would
 * otherwise build goes at the head (construct_overrides, or the
 * individual construct_<service> virtual when only one branch
 * changes), while publishing a service the chain knows nothing about
 * goes at the tail (construct_board_services). Anything
 * construct_overrides handles preempts the canonical chain.
 *
 * Each per-service branch follows the same shape:
 *   if !Publisher::has_type_id<X>(type_id) return false;
 *   construct the impl (parsing instance_id and fetching siblings
 *       or anchors as needed);
 *   ServicePublisher<X>(shared_from_this()).publish(instance_id, impl);
 *   return true;
 *
 * What this base deliberately does NOT name is any board-specific
 * behavior or resource — the peripherals a particular carrier wires
 * up, the sensor layout it exposes, and the board's own service
 * types. Those arrive from the subclass. */
class HsbIpPublisher : public Publisher,
                       public EnumerationInterfaceV1 {
public:
    bool construct_service(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        return construct_overrides(instance_id, type_id)
            || construct_hololink(instance_id, type_id)
            || construct_ptp_pps_output(instance_id, type_id)
            || construct_null_vsync(instance_id, type_id)
            || construct_data_channel(instance_id, type_id)
            || construct_roce_data_channel(instance_id, type_id)
            || construct_coe_data_channel(instance_id, type_id)
            || construct_roce_receiver(instance_id, type_id)
            || construct_linux_data_channel(instance_id, type_id)
            || construct_linux_receiver(instance_id, type_id)
            || construct_oscillator(instance_id, type_id)
            || construct_sequencer(instance_id, type_id)
            || construct_i2c(instance_id, type_id)
            || construct_board_services(instance_id, type_id);
    }

    /* EnumerationInterfaceV1 contract. The Publisher is its own
     * EnumerationInterfaceV1 impl; module_name() is the identity hook,
     * and the per-sensor addressing is owned by this module's
     * ChannelConfigurationInterfaceV1. update_metadata stamps the
     * module name and then delegates to that ChannelConfiguration's
     * use_sensor for the default (bootp) sensor — the same routine the
     * application calls to retarget a sensor — so the enumerate-time
     * metadata is exactly what use_sensor would produce and the two
     * paths can't drift. The default sensor is the bootp data_plane.
     * Virtual dispatch selects the board's own use_sensor, so boards
     * don't override update_metadata; they publish their own
     * ChannelConfiguration instead. The
     * ChannelConfiguration is published in setup() before any
     * enumerate, so the lookup always resolves. */
    hololink_module_status_t update_metadata(
        EnumerationMetadata& metadata,
        const uint8_t* /*raw_packet*/,
        size_t /*raw_packet_len*/) override
    {
        metadata["module_name"] = module_name();
        auto config = ChannelConfigurationInterfaceV1::get_service(
            this->self_module());
        // The default sensor is the bootp data_plane, which comes
        // straight from the untrusted bootp payload. Validate it against
        // the board's sensor range up front: a missing field reads back
        // as -1, and a plane the board doesn't have is out of range.
        // Either way, decline the device cleanly instead of forwarding a
        // bad value into use_sensor (which would throw out of the
        // reactor's bootp callback and abort enumeration of every
        // device).
        const int64_t data_plane = metadata.get<int64_t>("data_plane", -1);
        if (!config->is_sensor_valid(data_plane)) {
            return HOLOLINK_MODULE_ENUMERATION_SKIPPED;
        }
        config->use_sensor(metadata, data_plane);
        return update_board_metadata(metadata);
    }

    /* Last step of update_metadata, mirroring construct_board_services at
     * the tail of the construct_service chain. A board stamps whatever its
     * board-specific behaviors and resources need from the enumeration
     * metadata here, and the return value is update_metadata's — so a board
     * that can tell from the metadata that it must not drive this device
     * declines it with HOLOLINK_MODULE_ENUMERATION_SKIPPED. Base stamps
     * nothing and accepts.
     *
     * What a board stamps here must agree with what it publishes from
     * construct_board_services: a service configured from fields describing
     * some other hardware than the one published stays unconfigured, and
     * every call into it reports NOT_FOUND. Stamping nothing at all is
     * inert rather than broken, with the same NOT_FOUND result. */
    virtual hololink_module_status_t update_board_metadata(
        EnumerationMetadata& /*metadata*/)
    {
        return HOLOLINK_MODULE_OK;
    }

    /* Registers this Publisher as the EnumerationInterfaceV1
     * singleton in its own service registry. Called from
     * hololink_module_init after the Publisher is constructed.
     * Default publishes `this` (the Publisher serves as its own
     * EnumerationInterfaceV1 via multi-inheritance). A board
     * subclass overrides to publish a different EnumerationInterfaceV1
     * impl instead — e.g. one whose metadata stamping doesn't fit
     * the module_name + data-planes shape. */
    virtual void publish_enumeration()
    {
        auto self = shared_from_this();
        ServicePublisher<EnumerationInterfaceV1>(self).publish(
            "",
            std::shared_ptr<EnumerationInterfaceV1>(self, this));
    }

    /* Constructs the canonical FrameMetadataV1 and publishes it as
     * FrameMetadataInterfaceV1 in this Publisher's own registry.
     * Called from setup() during module init. Default registers
     * module_core::FrameMetadataV1; a board subclass overrides
     * to publish a different impl. */
    virtual void publish_frame_metadata()
    {
        auto impl = std::make_shared<FrameMetadataV1>();
        ServicePublisher<FrameMetadataInterfaceV1>(shared_from_this())
            .publish("", impl);
    }

    /* Publishes this board's ChannelConfigurationInterfaceV1 in the
     * Publisher's own registry. Called from setup() during module init.
     * Pure virtual: the sensor layout — how many sensors, which data
     * plane each sits on, how many SIFs each spans — is a property of
     * the board, so every board states it rather than inheriting a
     * default that may not fit. Implementations publish a
     * HsbIpChannelConfigurationV1 subclass via
     * ServicePublisher<HsbIpChannelConfigurationV1>. */
    virtual void publish_channel_configuration() = 0;

    /* Overrides Publisher::setup to publish the canonical singletons
     * (FrameMetadataInterfaceV1, EnumerationInterfaceV1,
     * ChannelConfigurationInterfaceV1) after the base's
     * init-validation + bootstrap succeeds. Boards that subclass
     * this publisher don't override setup themselves — they
     * override the publish_* hooks. */
    hololink_module_services_t setup(
        const hololink_module_init_t* init) override
    {
        auto result = Publisher::setup(init);
        if (result.status != HOLOLINK_MODULE_OK) {
            return result;
        }
        publish_frame_metadata();
        publish_enumeration();
        publish_channel_configuration();
        return result;
    }

    /* The configured HololinkV1 named by a "serial=<n>" instance_id. Throws
     * rather than returning null: the instance-id form does not configure, so
     * a parent fetched before the application resolved it by metadata has no
     * backing and would fault on the first register access. `what` names the
     * service being constructed, for the error text. Public so the publish
     * helpers that live next to the board-specific impls they construct can
     * resolve a parent the same way the in-class branches do. */
    std::shared_ptr<HololinkV1> configured_hololink(
        const std::string& instance_id, const char* what)
    {
        const auto fields = parse_name_value_pairs(instance_id);
        const auto serial_it = fields.find("serial");
        if (serial_it == fields.end()) {
            throw std::runtime_error(
                std::string("While constructing ") + what + " for instance_id '"
                + instance_id + "': missing 'serial'");
        }
        const std::string hololink_id = "serial=" + serial_it->second;
        auto hololink_impl = HololinkV1::get_service(
            this->self_module(), hololink_id.c_str());
        if (!hololink_impl->legacy_access()) {
            throw std::runtime_error(
                std::string("While constructing ") + what + " for instance_id '"
                + instance_id + "': the parent HololinkInterface (serial='"
                + serial_it->second
                + "') has not been configured yet — fetch it via "
                  "HololinkInterfaceV1::get_service(metadata) first");
        }
        return hololink_impl;
    }

protected:
    /* Identifies the board in enumeration metadata. Every board names
     * itself here. Pure virtual — the canonical Publisher does not
     * declare a default name. */
    virtual std::string module_name() const = 0;

    /* First in the chain. Base leaves blank. A board subclass
     * overrides this to substitute its own impl class on a canonical
     * type_id — the same effect as overriding that service's virtual,
     * but in one place when several branches change together.
     * Anything this method handles short-circuits the rest of the
     * chain. A service the canonical chain would never build has
     * nothing to preempt, so it belongs on construct_board_services
     * instead. */
    virtual bool construct_overrides(
        const std::string& /*instance_id*/,
        const std::string& /*type_id*/)
    {
        return false;
    }

    /* Last in the chain, mirroring construct_overrides at the head.
     * Board-specific behaviors and resources are published here: the
     * board's own service types, and the peripherals whose
     * implementation is a property of that carrier rather than of the
     * HSB IP. Neither substitutes for a canonical branch — there is
     * nothing to preempt — so they run after the canonical services
     * rather than ahead of them. Where several implementations of one
     * interface exist, the board picks the one matching its hardware;
     * only one can be effective, since they report the same interface
     * alias and the first to claim the type_id wins. Base leaves
     * blank; a subclass of a board publisher that already overrides
     * this chains to it. */
    virtual bool construct_board_services(
        const std::string& /*instance_id*/,
        const std::string& /*type_id*/)
    {
        return false;
    }

    virtual bool construct_hololink(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<HololinkV1>(type_id)) {
            return false;
        }
        auto impl = std::make_shared<HololinkV1>();
        ServicePublisher<HololinkV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    /* instance_id is "serial=<n>". The PtpPpsOutput is one physical
     * FPGA resource per board, parented to the per-board HololinkV1;
     * the impl needs the legacy hololink::Hololink to call
     * ptp_pps_output(freq) / setup() / shutdown() on its singleton
     * synchronizer, so we fetch the sibling HololinkV1 (must already
     * be configured) and hand its legacy_access() to the impl.
     *
     * NullVsyncV1 also publishes under "vsync.v1", differentiated by
     * its own "serial=<n>;kind=null" instance_id; skip those here so
     * construct_null_vsync can claim them. */
    virtual bool construct_ptp_pps_output(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<PtpPpsOutputInterfaceV1>(type_id)) {
            return false;
        }
        const auto fields = parse_name_value_pairs(instance_id);
        if (fields.find("kind") != fields.end()) {
            return false;
        }
        auto hololink_impl
            = configured_hololink(instance_id, "PtpPpsOutputInterface");
        auto impl = std::make_shared<PtpPpsOutputV1>(
            hololink_impl, hololink_impl->legacy_access());
        ServicePublisher<PtpPpsOutputInterfaceV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    /* instance_id is "serial=<n>;kind=null". A no-op Vsync stateless
     * impl is constructed for every request; no parent fetch needed.
     * Published under "vsync.v1" only (NullVsyncV1 has no own
     * type_id), differentiated from PtpPpsOutput by the kind=null
     * field in instance_id. */
    virtual bool construct_null_vsync(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<NullVsyncV1>(type_id)) {
            return false;
        }
        const auto fields = parse_name_value_pairs(instance_id);
        const auto kind_it = fields.find("kind");
        if (kind_it == fields.end() || kind_it->second != "null") {
            return false;
        }
        const auto serial_it = fields.find("serial");
        if (serial_it == fields.end()) {
            throw std::runtime_error(
                std::string("While constructing NullVsync for instance_id '")
                + instance_id + "': missing 'serial'");
        }
        auto hololink_impl = HololinkV1::get_service(
            this->self_module(),
            ("serial=" + serial_it->second).c_str());
        auto impl = std::make_shared<NullVsyncV1>(hololink_impl);
        ServicePublisher<NullVsyncV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    virtual bool construct_data_channel(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<DataChannelV1>(type_id)) {
            return false;
        }
        auto impl = std::make_shared<DataChannelV1>();
        ServicePublisher<DataChannelV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    virtual bool construct_roce_data_channel(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<RoceDataChannelV1>(type_id)) {
            return false;
        }
        auto anchor = DataChannelInterfaceV1::get_service(
            this->self_module(), instance_id.c_str());
        auto impl = std::make_shared<RoceDataChannelV1>(std::move(anchor));
        ServicePublisher<RoceDataChannelV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    virtual bool construct_coe_data_channel(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<CoeDataChannelV1>(type_id)) {
            return false;
        }
        auto anchor = DataChannelInterfaceV1::get_service(
            this->self_module(), instance_id.c_str());
        auto impl = std::make_shared<CoeDataChannelV1>(std::move(anchor));
        ServicePublisher<CoeDataChannelV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    /* Always declared and always called from the construct_service
     * chain; defined out-of-line in roce_receiver_construct.cpp so this
     * header never names the legacy ibverbs RoceReceiver. The body
     * publishes a functional RoceReceiverV1 when the build has RoCE,
     * and is empty (publishes nothing) otherwise. */
    virtual bool construct_roce_receiver(
        const std::string& instance_id,
        const std::string& type_id);

    /* Software-transport sibling of construct_roce_data_channel. The
     * LinuxDataChannelV1 wraps the same legacy hololink::DataChannel
     * (no ibverbs), so this branch is inline and unconditional. */
    virtual bool construct_linux_data_channel(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<LinuxDataChannelV1>(type_id)) {
            return false;
        }
        auto anchor = DataChannelInterfaceV1::get_service(
            this->self_module(), instance_id.c_str());
        auto impl = std::make_shared<LinuxDataChannelV1>(std::move(anchor));
        ServicePublisher<LinuxDataChannelV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    /* Always declared and always called from the construct_service
     * chain; defined out-of-line in linux_receiver_construct.cpp to keep
     * this header free of the legacy receiver's <cuda.h> dependency
     * (the impl header pulls it in). Unlike construct_roce_receiver the
     * body is unconditional — the software receiver needs no ibverbs,
     * so it always publishes a functional LinuxReceiverV1. */
    virtual bool construct_linux_receiver(
        const std::string& instance_id,
        const std::string& type_id);

    /* The board's clock generator. Base publishes none — which chip is
     * on the carrier, and therefore which OscillatorInterfaceV1 impl
     * serves it, is a board property. Boards override this and route
     * through construct_shared_oscillator with their own impl. */
    virtual bool construct_oscillator(
        const std::string& /*instance_id*/,
        const std::string& /*type_id*/)
    {
        return false;
    }

    /* The on-board clock generator is a per-board resource shared
     * across every data plane on the board, so the Oscillator impl
     * is constructed once per (serial) and published under every
     * (serial, data_plane) instance_id the framework asks for. The
     * impl's idempotent rate cache is what keeps a second channel's
     * `enable()` from reprogramming the shared chip and disrupting
     * the first channel that's already streaming; that cache only
     * works if the second `enable()` reaches the same impl instance,
     * which is what this share-by-serial does. */
    template <typename Impl>
    bool construct_shared_oscillator(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<Impl>(type_id)) {
            return false;
        }
        const auto fields = parse_name_value_pairs(instance_id);
        const auto serial_it = fields.find("serial");
        if (serial_it == fields.end()) {
            throw std::runtime_error(
                std::string("While constructing Oscillator for instance_id '")
                + instance_id + "': missing 'serial'");
        }
        // operator[] default-inserts a null shared_ptr for a new
        // serial; `slot` is a *reference* to that map slot, so the
        // assignment below writes the new oscillator back into
        // oscillator_by_serial_ (it would be lost if `slot` were a
        // copy). A pre-existing entry is reused as-is.
        auto& slot = oscillator_by_serial_[serial_it->second];
        if (!slot) {
            slot = std::make_shared<Impl>();
        }
        ServicePublisher<Impl>(shared_from_this())
            .publish(instance_id, std::static_pointer_cast<Impl>(slot));
        return true;
    }

    /* instance_id is "serial=<n>;data_channel=<c>;kind=<k>". Only
     * kind=frame_end is published today; other kinds fall through
     * so the cache miss surfaces to the caller. */
    virtual bool construct_sequencer(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<SequencerInterfaceV1>(type_id)) {
            return false;
        }
        const auto fields = parse_name_value_pairs(instance_id);
        const auto serial_it = fields.find("serial");
        const auto data_channel_it = fields.find("data_channel");
        const auto kind_it = fields.find("kind");
        if (serial_it == fields.end() || data_channel_it == fields.end()
            || kind_it == fields.end()) {
            throw std::runtime_error(
                std::string("While constructing SequencerInterface for instance_id '")
                + instance_id
                + "': missing 'serial', 'data_channel', or 'kind'");
        }
        if (kind_it->second != "frame_end") {
            return false;
        }
        // The frame-end Sequencer is a RoCE-channel concept — its
        // source is the legacy hololink::DataChannel that
        // RoceDataChannelV1 holds. Resolve the RoCE impl (via its
        // impl type_id, no cast needed) — its legacy_data_channel()
        // is null until the RoCE impl's configure() has run, in
        // which case we throw.
        const std::string channel_id
            = "serial=" + serial_it->second
            + ";data_channel=" + data_channel_it->second;
        auto roce_impl = RoceDataChannelV1::get_service(
            this->self_module(), channel_id.c_str());
        auto legacy_data_channel = roce_impl->legacy_data_channel();
        if (!legacy_data_channel) {
            throw std::runtime_error(
                std::string("While constructing SequencerInterface for instance_id '")
                + instance_id
                + "': the parent RoceDataChannelInterface has not been "
                  "configured yet — fetch it via "
                  "RoceDataChannelInterfaceV1::get_service(metadata) first");
        }
        auto hololink_impl = HololinkV1::get_service(
            this->self_module(),
            ("serial=" + serial_it->second).c_str());
        auto impl = std::make_shared<SequencerV1>(
            hololink_impl, legacy_data_channel->frame_end_sequencer());
        ServicePublisher<SequencerInterfaceV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    /* instance_id is "serial=<n>;bus=<b>;address=<a>". The per-bus
     * legacy I2c handle comes from this board's already-configured
     * HololinkImpl — fetched directly via HololinkV1::get_service,
     * no cast needed. */
    virtual bool construct_i2c(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<I2cInterfaceV1>(type_id)) {
            return false;
        }
        const auto fields = parse_name_value_pairs(instance_id);
        const auto bus_it = fields.find("bus");
        if (bus_it == fields.end()) {
            throw std::runtime_error(
                std::string("While constructing I2cInterface for instance_id '")
                + instance_id + "': missing 'bus'");
        }
        const uint32_t bus
            = static_cast<uint32_t>(std::stoul(bus_it->second));
        auto hololink_impl = configured_hololink(instance_id, "I2cInterface");
        auto impl = std::make_shared<I2cV1>(
            hololink_impl, hololink_impl->legacy_access()->get_i2c(bus));
        ServicePublisher<I2cInterfaceV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

private:
    // Per-board cache of the board's oscillator impl — see
    // construct_shared_oscillator for why share-by-serial is required.
    // Type-erased so one cache serves whichever impl the board names.
    std::unordered_map<std::string, std::shared_ptr<void>>
        oscillator_by_serial_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_HSB_IP_PUBLISHER_HPP
