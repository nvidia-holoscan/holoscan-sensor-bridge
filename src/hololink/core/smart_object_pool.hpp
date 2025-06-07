/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#ifndef SRC_HOLOLINK_SMART_OBJECT_POOL_HPP
#define SRC_HOLOLINK_SMART_OBJECT_POOL_HPP

#include <memory>
#include <stack>
#include <stdexcept>

namespace hololink {

/**
 * The SmartObjectPool is a pool of smart pointers to T objects.
 * When an object is acquired from the pool, a smart pointer to the object is returned.
 * When the smart pointer is destructed, the object is automatically returns to the queue.
 * If the queue is already destructed, the object is destructed as well.
 */
template <class T, class D = std::default_delete<T>>
class SmartObjectPool {
private:
    // The deleter that is stored in the smart pointer
    struct return_to_pool_deleter {
        explicit return_to_pool_deleter(std::weak_ptr<SmartObjectPool<T, D>*> pool)
            : pool_(pool)
        {
        }

        void operator()(T* ptr)
        {
            // The pointer was already released
            if (!ptr)
                return;
            if (auto pool = pool_.lock()) // the pool is till alive, return the object to the pool
                (*pool)->stack_.push(std::unique_ptr<T, D> { ptr });
            else // the poool is already destructed, destruct the object
                D {}(ptr);
        }

    private:
        std::weak_ptr<SmartObjectPool<T, D>*> pool_;
    };

public:
    using Deleter = D;
    // The smart pointer type
    using Pointer = std::unique_ptr<T, return_to_pool_deleter>;

    SmartObjectPool()
        : self_(new SmartObjectPool<T, D>*(this))
    {
    }

    // Insert a new object to the pool and returns a smart pointer to it.
    template <typename... Args>
    Pointer emplace(Args&&... args)
    {
        stack_.push(std::make_unique<T>(std::forward<Args>(args)...));
        return acquire();
    }

    // acquire an object from the pool and returns a smart pointer to it.
    // If pool is empty, returns an empty smart pointer
    Pointer acquire()
    {
        if (empty())
            return Pointer(nullptr, return_to_pool_deleter(std::weak_ptr<SmartObjectPool<T, D>*>()));

        Pointer pointer(stack_.top().release(),
            return_to_pool_deleter {
                std::weak_ptr<SmartObjectPool<T, D>*>(self_) });
        stack_.pop();
        return pointer;
    }

    // Returns true if pool is empty
    bool empty() const
    {
        return stack_.empty();
    }

    // Returns the number of objects in the pool
    size_t size() const
    {
        return stack_.size();
    }

    // Removes one object from the pool. If pool empty, do nothing.
    void pop()
    {
        if (empty())
            return;
        stack_.pop();
    }

private:
    std::shared_ptr<SmartObjectPool<T, D>*> self_; // A pointer to this so we can keep track of the pool's life
    std::stack<std::unique_ptr<T, D>> stack_; // Where the objects are stored
};

} // namespace hololink

#endif /* SRC_HOLOLINK_SMART_OBJECT_POOL_HPP */
