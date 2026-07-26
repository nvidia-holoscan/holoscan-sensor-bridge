/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_HSB_LITE_PUBLISHER_HPP
#define HOLOLINK_MODULE_CORE_HSB_LITE_PUBLISHER_HPP

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
#include "hsb_lite_default.hpp"
#include "i2c_default.hpp"
#include "linux_data_channel_default.hpp"
#include "null_vsync_default.hpp"
#include "ptp_pps_output_default.hpp"
#include "roce_data_channel_default.hpp"
#include "sequencer_default.hpp"

namespace hololink::module::module_core {

/* Canonical HSB-Lite per-sensor addressing formula, and the single
 * source of truth for it. Stamps the HSB-shape address fields a
 * sensor occupies, given the logical channel (sensor_number), the
 * physical HIF index (data_plane), and how many SIFs each sensor
 * spans (sifs_per_sensor).
 *
 * The HIF register space is per-physical-data-plane; multiple sensors
 * may share a data plane (Leopard VB1940-AIO does, with data_plane
 * fixed at 0), while HSB-Lite maps each sensor 1:1 onto its own data
 * plane. Both the enumerate-time path (HsbLitePublisher::update_metadata,
 * via the ChannelConfiguration it delegates to) and the application
 * path (ChannelConfigurationInterfaceV1::use_sensor) route through
 * this function, so the two can't drift. In particular hif_address is
 * recomputed on every sensor selection, so switching to a sensor on a
 * different data plane re-points the HIF correctly. */
inline void hsb_lite_sensor_metadata(
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
    // per-data-plane index — on N-sensors-per-data-plane boards
    // (Leopard VB1940-AIO) the same data_plane carries multiple
    // sensors and only `data_channel` separates them. Single-camera
    // flows that never call Adapter::use_sensor still get a usable
    // default here; stereo flows overwrite it by calling use_sensor(n).
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

/* Canonical ChannelConfigurationInterfaceV1 impl for HSB-Lite-shaped
 * boards. use_sensor re-stamps the per-sensor VP / SIF address fields
 * through the shared hsb_lite_sensor_metadata formula (two SIFs per
 * sensor) while leaving the data plane on the enumerated one — sensors
 * are multiplexed onto a data plane by VP, not 1:1 with data planes;
 * use_mtu records the MTU. The class is stateless — one shared instance
 * per loaded module .so is enough. */
class HsbLiteChannelConfigurationV1
    : public ChannelConfigurationInterfaceV1,
      public Service<HsbLiteChannelConfigurationV1> {
public:
    static constexpr const char* type_id = "channel_configuration.module_core.v1";
    using Service<HsbLiteChannelConfigurationV1>::get_service;
    using Service<HsbLiteChannelConfigurationV1>::for_each_type_id;
    using ServiceAlias = ChannelConfigurationInterfaceV1;

    // HSB-Lite carries 3 sensors (sensor 2 is the I2S audio interface)
    // mapped 1:1 onto their own data planes, two SIFs each. The single
    // source of truth for the valid sensor range, shared by use_sensor's
    // bounds check and is_sensor_valid().
    static constexpr int64_t TOTAL_SENSORS = 3;

    HsbLiteChannelConfigurationV1() = default;

    bool is_sensor_valid(int64_t sensor_number) const override
    {
        return sensor_number >= 0 && sensor_number < TOTAL_SENSORS;
    }

    void use_sensor(
        EnumerationMetadata& metadata, int64_t sensor_number) override
    {
        // data_plane == sensor_number, so selecting a sensor also
        // re-points hif_address at that sensor's data plane.
        if (!is_sensor_valid(sensor_number)) {
            throw std::runtime_error(
                "While selecting an HSB-Lite sensor: sensor_number "
                + std::to_string(sensor_number)
                + " is out of range (HSB-Lite supports 0.."
                + std::to_string(TOTAL_SENSORS - 1) + ")");
        }
        const int64_t data_plane = metadata.get<int64_t>("data_plane", int64_t { 0 });
        hsb_lite_sensor_metadata(metadata, sensor_number,
            /*data_plane=*/data_plane,
            /*sifs_per_sensor=*/2);
        // Diagnostic: confirms which use_sensor code path is loaded. The
        // fixed path keeps hif_address on the enumerated data plane (so
        // sensor 1 stays on data plane 0); the old data_plane==sensor_number
        // path would log hif_address=0x2010300 for sensor 1.
        HSB_LOG_DEBUG(
            "use_sensor sensor_number={} data_plane={} hif_address={:#x} "
            "vp_address={:#x} data_channel={}",
            sensor_number, data_plane,
            metadata.get<int64_t>("hif_address", int64_t { 0 }),
            metadata.get<int64_t>("vp_address", int64_t { 0 }),
            metadata.get<int64_t>("data_channel", int64_t { 0 }));
    }

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

/* Canonical Publisher for HSB-Lite-shaped boards. construct_service
 * is a short-circuit OR over construct_overrides (a board-extension
 * hook the base leaves blank) plus one per-service virtual per
 * canonical type_id. Boards subclass and override individual
 * construct_<service> methods (targeted substitution of one
 * canonical branch) or construct_overrides (centralized: substitute
 * impl classes on canonical type_ids and/or add bespoke type_ids
 * such as SPI). Anything construct_overrides handles preempts the
 * canonical chain.
 *
 * Each per-service branch follows the same shape:
 *   if !Publisher::has_type_id<X>(type_id) return false;
 *   construct the impl (parsing instance_id and fetching siblings
 *       or anchors as needed);
 *   ServicePublisher<X>(shared_from_this()).publish(instance_id, impl);
 *   return true;
 */
class HsbLitePublisher : public Publisher,
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
            || construct_hsb_lite(instance_id, type_id)
            || construct_oscillator(instance_id, type_id)
            || construct_sequencer(instance_id, type_id)
            || construct_i2c(instance_id, type_id);
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
     * Virtual dispatch selects the board's own use_sensor (e.g.
     * Leopard's), so supplements don't override update_metadata; they
     * publish their own ChannelConfiguration instead. The
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
        // the board's sensor range up front: a missing/mistyped field
        // reads back as -1 via the non-throwing get, and a plane the
        // board doesn't have is out of range. Either way, decline the
        // device cleanly instead of forwarding a bad value into
        // use_sensor (which would throw out of the reactor's bootp
        // callback and abort enumeration of every device).
        const int64_t data_plane = metadata.get<int64_t>("data_plane", -1);
        if (!config->is_sensor_valid(data_plane)) {
            return HOLOLINK_MODULE_ENUMERATION_SKIPPED;
        }
        config->use_sensor(metadata, data_plane);
        return HOLOLINK_MODULE_OK;
    }

    /* Registers this Publisher as the EnumerationInterfaceV1
     * singleton in its own service registry. Called from
     * hololink_module_init after the Publisher is constructed.
     * Default publishes `this` (the Publisher serves as its own
     * EnumerationInterfaceV1 via multi-inheritance). A supplement
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
     * module_core::FrameMetadataV1; a supplement subclass overrides
     * to publish a different impl. */
    virtual void publish_frame_metadata()
    {
        auto impl = std::make_shared<FrameMetadataV1>();
        ServicePublisher<FrameMetadataInterfaceV1>(shared_from_this())
            .publish("", impl);
    }

    /* Constructs the canonical HsbLiteChannelConfigurationV1 and
     * publishes it as ChannelConfigurationInterfaceV1 in this
     * Publisher's own registry. Called from setup() during module
     * init. Default stamps per-sensor VP / SIF addressing via the shared
     * hsb_lite_sensor_metadata formula (sensors multiplexed onto a data
     * plane by VP); a supplement subclass overrides to publish a
     * different impl when the board layout doesn't fit. */
    virtual void publish_channel_configuration()
    {
        auto impl = std::make_shared<HsbLiteChannelConfigurationV1>();
        ServicePublisher<HsbLiteChannelConfigurationV1>(shared_from_this())
            .publish("", impl);
    }

    /* Overrides Publisher::setup to publish the canonical singletons
     * (FrameMetadataInterfaceV1, EnumerationInterfaceV1,
     * ChannelConfigurationInterfaceV1) after the base's
     * init-validation + bootstrap succeeds. Supplements that subclass
     * HsbLitePublisher don't override setup themselves — they
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

protected:
    /* Identifies the supplement in enumeration metadata. Every
     * partner supplement names itself here. Pure virtual — the
     * canonical Publisher does not declare a default name. */
    virtual std::string module_name() const = 0;

    /* First in the chain. Base leaves blank. A board subclass
     * overrides this to handle anything ahead of the canonical
     * branches — substituting an impl class on a canonical type_id
     * (same effect as overriding the per-service virtual, but in one
     * place), or adding bespoke type_ids (SPI, UART, a board-specific
     * service). Anything this method handles short-circuits the rest
     * of the chain. */
    virtual bool construct_overrides(
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
        const auto serial_it = fields.find("serial");
        if (serial_it == fields.end()) {
            throw std::runtime_error(
                std::string("While constructing PtpPpsOutputInterface for instance_id '")
                + instance_id + "': missing 'serial'");
        }
        const std::string hololink_id = "serial=" + serial_it->second;
        auto hololink_impl = HololinkV1::get_service(
            this->self_module(), hololink_id.c_str());
        auto legacy = hololink_impl->legacy_access();
        if (!legacy) {
            throw std::runtime_error(
                std::string("While constructing PtpPpsOutputInterface for instance_id '")
                + instance_id
                + "': the parent HololinkInterface (serial='" + serial_it->second
                + "') has not been configured yet — fetch it via "
                  "HololinkInterfaceV1::get_service(metadata) first");
        }
        auto impl = std::make_shared<PtpPpsOutputV1>(
            hololink_impl, std::move(legacy));
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

    virtual bool construct_hsb_lite(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<HsbLiteV1>(type_id)) {
            return false;
        }
        auto impl = std::make_shared<HsbLiteV1>();
        ServicePublisher<HsbLiteV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    /* The on-board clock generator is a per-board resource shared
     * across every data plane on the board, so the Oscillator impl
     * is constructed once per (serial) and published under every
     * (serial, data_plane) instance_id the framework asks for. The
     * impl's idempotent rate cache is what keeps a second stereo
     * channel's `enable()` from re-running setup_clock and
     * disrupting the first channel that's already streaming; that
     * cache only works if the second `enable()` reaches the same
     * impl instance, which is what this share-by-serial does. */
    virtual bool construct_oscillator(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<HsbLiteOscillatorV1>(type_id)) {
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
        // serial; `impl` is a *reference* to that map slot, so the
        // assignment below writes the new oscillator back into
        // oscillator_by_serial_ (it would be lost if `impl` were a
        // copy). A pre-existing entry is reused as-is.
        auto& impl = oscillator_by_serial_[serial_it->second];
        if (!impl) {
            impl = std::make_shared<HsbLiteOscillatorV1>();
        }
        ServicePublisher<HsbLiteOscillatorV1>(shared_from_this())
            .publish(instance_id, impl);
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
        const auto serial_it = fields.find("serial");
        const auto bus_it = fields.find("bus");
        if (serial_it == fields.end() || bus_it == fields.end()) {
            throw std::runtime_error(
                std::string("While constructing I2cInterface for instance_id '")
                + instance_id + "': missing 'serial' or 'bus'");
        }
        const std::string& serial = serial_it->second;
        const uint32_t bus
            = static_cast<uint32_t>(std::stoul(bus_it->second));
        const std::string hololink_id = "serial=" + serial;
        auto hololink_impl = HololinkV1::get_service(
            this->self_module(), hololink_id.c_str());
        auto legacy = hololink_impl->legacy_access();
        if (!legacy) {
            throw std::runtime_error(
                std::string("While constructing I2cInterface for instance_id '")
                + instance_id
                + "': the parent HololinkInterface (serial='" + serial
                + "') has not been configured yet — fetch it via "
                  "HololinkInterfaceV1::get_service(metadata) first");
        }
        auto impl = std::make_shared<I2cV1>(
            hololink_impl, legacy->get_i2c(bus));
        ServicePublisher<I2cInterfaceV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

private:
    // Per-board cache of the canonical HsbLiteOscillatorV1 — see
    // construct_oscillator for why share-by-serial is required.
    std::unordered_map<std::string, std::shared_ptr<HsbLiteOscillatorV1>>
        oscillator_by_serial_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_HSB_LITE_PUBLISHER_HPP
