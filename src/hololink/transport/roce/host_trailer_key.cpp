/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "host_trailer_key.hpp"

#if !defined(HOLOLINK_HAS_MLX5DV)

namespace hololink::transport::roce::detail {

std::unique_ptr<HostTrailerKey> try_host_trailer_key(ibv::Context&,
    ibv::ProtectionDomain&, std::uint8_t, ::ibv_mtu, const LocalRoceV2Ipv4Gid&,
    const ibv::MemoryRegion&, std::uint64_t, std::uint32_t,
    std::uint32_t) noexcept
{
    return nullptr;
}

} // namespace hololink::transport::roce::detail

#else

#include <chrono>
#include <optional>
#include <type_traits>
#include <vector>

#include <dlfcn.h>

#include <infiniband/mlx5dv.h>

namespace hololink::transport::roce::detail {

namespace {

    // libibverbs has already loaded this provider for the context, so dlopen
    // returns that instance. Loading it at run time keeps libmlx5 out of the
    // exported static archive's link interface.
    struct Mlx5 {
        decltype(&::mlx5dv_create_mkey) create_mkey;
        decltype(&::mlx5dv_destroy_mkey) destroy_mkey;
        decltype(&::mlx5dv_create_qp) create_qp;
        decltype(&::mlx5dv_qp_ex_from_ibv_qp_ex) qp_ex_from;
    };

    const Mlx5* mlx5() noexcept
    {
        static const std::optional<Mlx5> api = []() -> std::optional<Mlx5> {
            void* handle = dlopen("libmlx5.so.1", RTLD_NOW | RTLD_LOCAL);
            if (handle == nullptr) {
                return std::nullopt;
            }
            const auto symbol = [handle](auto& function, const char* name) {
                function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
                    dlsym(handle, name));
                return function != nullptr;
            };
            Mlx5 loaded {};
            if (!symbol(loaded.create_mkey, "mlx5dv_create_mkey")
                || !symbol(loaded.destroy_mkey, "mlx5dv_destroy_mkey")
                || !symbol(loaded.create_qp, "mlx5dv_create_qp")
                || !symbol(loaded.qp_ex_from, "mlx5dv_qp_ex_from_ibv_qp_ex")) {
                return std::nullopt;
            }
            return loaded;
        }();
        return api ? &*api : nullptr;
    }

    // MR_LIST is an RC-only send op, so the setup QP is RC looped back to
    // itself, and RC RTR and RTS need the attributes the UC helpers omit.
    bool to_rts(ibv::QueuePair& qp, std::uint8_t port_num, ::ibv_mtu path_mtu,
        const LocalRoceV2Ipv4Gid& gid)
    {
        if (qp.reset_to_init(port_num) != 0) {
            return false;
        }
        ::ibv_qp_attr attr {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = path_mtu;
        attr.dest_qp_num = qp.ptr_->qp_num;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = port_num;
        attr.ah_attr.grh.dgid = gid.gid;
        attr.ah_attr.grh.sgid_index = static_cast<std::uint8_t>(gid.index);
        attr.ah_attr.grh.hop_limit = 0xFF;
        if (ibv_modify_qp(qp.ptr_, &attr,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN
                    | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC
                    | IBV_QP_MIN_RNR_TIMER)
            != 0) {
            return false;
        }
        attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.max_rd_atomic = 1;
        return ibv_modify_qp(qp.ptr_, &attr,
                   IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT
                       | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY
                       | IBV_QP_MAX_QP_RD_ATOMIC)
            == 0;
    }

} // namespace

std::unique_ptr<HostTrailerKey> try_host_trailer_key(ibv::Context& context,
    ibv::ProtectionDomain& protection_domain, std::uint8_t port_num,
    ::ibv_mtu path_mtu, const LocalRoceV2Ipv4Gid& gid,
    const ibv::MemoryRegion& payload, std::uint64_t stride,
    std::uint32_t trailer_bytes, std::uint32_t slots) noexcept
{
    const Mlx5* api = mlx5();
    // Two KLM entries per slot must fit a uint16_t, and rdma-core before v61
    // sums the whole KLM list in an int.
    if (api == nullptr || !payload || slots == 0 || slots > 0x7FFF
        || trailer_bytes == 0 || trailer_bytes >= stride
        || stride * slots >= (std::uint64_t { 1 } << 31)) {
        return nullptr;
    }
    try {
        const auto entries = static_cast<std::uint16_t>(2 * slots);
        // Declared first, so the setup QP and CQ are destroyed before it.
        auto key = std::make_unique<HostTrailerKey>();
        // The QP comes first: unlike older mlx5dv_create_mkey, it rejects a
        // non-mlx5 device, and the inline size is what caps the page count.
        ibv::CompletionQueue completion_queue
            = context.create_completion_queue(1);
        ::ibv_qp_init_attr_ex qp_attr {};
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.send_cq = completion_queue.ptr_;
        qp_attr.recv_cq = completion_queue.ptr_;
        qp_attr.cap.max_send_wr = 1;
        qp_attr.cap.max_recv_wr = 1;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        // rdma-core reserves the UMR control segments. Each KLM is 16 bytes.
        qp_attr.cap.max_inline_data = 16u * entries;
        qp_attr.pd = protection_domain.ptr_;
        qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        qp_attr.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;
        ::mlx5dv_qp_init_attr mlx5_attr {};
        mlx5_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS;
        mlx5_attr.send_ops_flags = MLX5DV_QP_EX_WITH_MR_LIST;
        ibv::QueuePair setup_qp(api->create_qp(context.ptr_, &qp_attr, &mlx5_attr));
        if (!setup_qp || !to_rts(setup_qp, port_num, path_mtu, gid)) {
            return nullptr;
        }

        const std::size_t host_bytes = std::size_t { slots } * trailer_bytes;
        key->trailers = HostAllocation(host_bytes, host_page_size());
        key->region = protection_domain.register_memory_region(key->trailers.get(),
            host_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        ::mlx5dv_mkey_init_attr mkey_attr {};
        mkey_attr.pd = protection_domain.ptr_;
        mkey_attr.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT;
        mkey_attr.max_entries = entries;
        key->mkey = { api->create_mkey(&mkey_attr), api->destroy_mkey };
        if (!key->mkey) {
            return nullptr;
        }

        const auto host = reinterpret_cast<std::uint64_t>(key->trailers.get());
        const auto payload_bytes = static_cast<std::uint32_t>(stride - trailer_bytes);
        std::vector<::ibv_sge> klm(entries);
        for (std::uint32_t slot = 0; slot < slots; ++slot) {
            klm[2 * slot] = { slot * stride, payload_bytes, payload.ptr_->lkey };
            klm[2 * slot + 1] = { host + std::uint64_t { slot } * trailer_bytes,
                trailer_bytes, key->region.ptr_->lkey };
        }
        ::ibv_qp_ex* extended = ibv_qp_to_qp_ex(setup_qp.ptr_);
        ::mlx5dv_qp_ex* mlx5_qp = extended != nullptr ? api->qp_ex_from(extended) : nullptr;
        if (mlx5_qp == nullptr) {
            return nullptr;
        }
        ibv_wr_start(extended);
        extended->wr_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        mlx5dv_wr_mr_list(mlx5_qp, key->mkey.get(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE, entries, klm.data());
        if (ibv_wr_complete(extended) != 0) {
            return nullptr;
        }

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        std::optional<::ibv_wc> completion;
        while (!(completion = completion_queue.poll())
            && std::chrono::steady_clock::now() < deadline) {
            cpu_relax();
        }
        if (!completion || completion->status != IBV_WC_SUCCESS) {
            return nullptr;
        }
        key->rkey = key->mkey->rkey;
        return key;
    } catch (...) {
        return nullptr;
    }
}

} // namespace hololink::transport::roce::detail

#endif
