/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * The Holoargs work was inspired by the Boost.Program_options library.
 * Although it was implemented from scratch, an effort was made to keep the
 * API similar to Boost.Program_options' for whose those who are familiar with.
 */
#ifndef SRC_HOLOLINK_HOLOARGS_HPP
#define SRC_HOLOLINK_HOLOARGS_HPP

#include <argp.h>

#include <any>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace hololink {
namespace args {

    class OptionsDescription;
    class Parser;

    class RequiredOption : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    class ValueSemantic : public std::enable_shared_from_this<ValueSemantic> {
    public:
        using Pointer = std::shared_ptr<ValueSemantic>;
        virtual bool is_required() const = 0;
        virtual std::string name() const = 0;
        virtual bool apply_default(std::any& value) const = 0;
        virtual void parse(std::any& value_store, const std::string& token) const = 0;
        virtual const std::type_info& value_type() const = 0;
    };

    template <typename T>
    class TypedValue : public ValueSemantic {
    public:
        using Pointer = std::shared_ptr<TypedValue<T>>;

        virtual ~TypedValue() = default;

        Pointer required()
        {
            required_ = true;
            return std::static_pointer_cast<TypedValue<T>>(shared_from_this());
        }

        Pointer value_name(const std::string& name)
        {
            name_ = name;
            return std::static_pointer_cast<TypedValue<T>>(shared_from_this());
        }

        Pointer default_value(const T& default_value)
        {
            default_value_ = default_value;
            return std::static_pointer_cast<TypedValue<T>>(shared_from_this());
        }

        bool is_required() const override
        {
            return required_;
        }

        std::string name() const override
        {
            return name_;
        }

        bool apply_default(std::any& value) const override
        {
            if (default_value_.has_value())
                value = default_value_;
            return default_value_.has_value();
        }

        void parse(std::any& value_store, const std::string& token) const override
        {
            T value {};
            std::istringstream iss(token);
            iss >> value;
            value_store = value;
        }

        const std::type_info& value_type() const override
        {
            return typeid(T);
        }

    private:
        bool required_ = false;
        std::string name_;
        std::any default_value_;
    };

    template <typename T>
    typename TypedValue<T>::Pointer value()
    {
        return std::make_shared<TypedValue<T>>();
    }

    TypedValue<bool>::Pointer bool_switch();

    class OptionDescription {
    public:
        OptionDescription(const std::string& name, int key, const std::string& doc, std::shared_ptr<const ValueSemantic> value_semantic = nullptr);

    private:
        std::string name_;
        int key_;
        std::string doc_;
        std::shared_ptr<const ValueSemantic> value_semantic_;

        friend class OptionsDescription;
        friend class Parser;
    };

    class OptionsDescriptionEasyInit {
    public:
        OptionsDescriptionEasyInit(OptionsDescription& options_descripton);

        OptionsDescriptionEasyInit& operator()(const std::string& name, const std::string& doc);
        OptionsDescriptionEasyInit& operator()(const std::string& name, std::shared_ptr<const ValueSemantic> value_semantic, const std::string& doc);

    private:
        OptionsDescription& options_description_;
    };

    class OptionsDescription {
    public:
        OptionsDescription(const std::string& caption = std::string());
        void add(const OptionDescription& option_description);
        OptionsDescriptionEasyInit add_options();

    private:
        std::string caption_;
        int long_form_only_index_;
        std::unordered_map<int, OptionDescription> options_;

        friend class OptionsDescriptionEasyInit;
        friend class Parser;
    };

    class VariableValue {
    public:
        VariableValue() = default;
        VariableValue(const std::any& value);
        std::any& value();
        const std::any& value() const;
        bool empty() const;

        template <typename T>
        T& as()
        {
            return std::any_cast<T&>(value_);
        }

        template <typename T>
        const T& as() const
        {
            return std::any_cast<const T&>(value_);
        }

    private:
        std::any value_;
    };

    using VariablesMap = std::unordered_map<std::string, VariableValue>;

    class Parser {
    public:
        VariablesMap parse_command_line(int argc, char* argv[], const OptionsDescription& od) const;

    private:
        static error_t parse_option(int key, char* arg, ::argp_state* state);
    };

} // namespace args
} // namespace hololink

#endif /* SRC_HOLOLINK_HOLOARGS_HPP */
