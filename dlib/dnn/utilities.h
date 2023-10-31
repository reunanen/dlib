// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_UTILITIES_H_
#define DLIB_DNn_UTILITIES_H_

#include "../cuda/tensor.h"
#include "utilities_abstract.h"
#include "../geometry.h"
#include <fstream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    inline void randomize_parameters (
        tensor& params,
        unsigned long num_inputs_and_outputs,
        dlib::rand& rnd
    )
    {
        for (auto& val : params)
        {
            // Draw a random number to initialize the layer according to formula (16)
            // from Understanding the difficulty of training deep feedforward neural
            // networks by Xavier Glorot and Yoshua Bengio.
            val = 2*rnd.get_random_float()-1;
            val *= std::sqrt(6.0/(num_inputs_and_outputs));
        }
    }

// ----------------------------------------------------------------------------------------

    template <typename label_type>
    struct weighted_label
    {
        weighted_label()
        {}

        weighted_label(label_type label, float weight = 1.f)
            : label(label), weight(weight)
        {}

        label_type label{};
        float weight = 1.f;
    };

// ----------------------------------------------------------------------------------------

    inline double log1pexp(double x)
    {
        using std::exp;
        using namespace std; // Do this instead of using std::log1p because some compilers
                             // error out otherwise (E.g. gcc 4.9 in cygwin)
        if (x <= -37)
            return exp(x);
        else if (-37 < x && x <= 18)
            return log1p(exp(x));
        else if (18 < x && x <= 33.3)
            return x + exp(-x);
        else
            return x;
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    T safe_log(T input, T epsilon = 1e-10)
    {
        // Prevent trying to calculate the logarithm of a very small number (let alone zero)
        return std::log(std::max(input, epsilon));
    }

// ----------------------------------------------------------------------------------------

    inline size_t tensor_index(
        const tensor& t,
        const long sample,
        const long k,
        const long r,
        const long c
    )
    {
        return ((sample * t.k() + k) * t.nr() + r) * t.nc() + c;
    }

// ----------------------------------------------------------------------------------------

    class cost_weight_matrix_index_based
    {
    public:
        using index_type = size_t;

        cost_weight_matrix_index_based() {}

        cost_weight_matrix_index_based(index_type index_count)
        {
            resize(index_count);
        }

        void resize(index_type index_count)
        {
            const auto default_cost_weight = get_default_cost_weight();

            cost_weights.resize(index_count, std::vector<float>(index_count, default_cost_weight));

            for (auto& column : cost_weights)
            {
                column.resize(index_count, default_cost_weight);
            }

            cost_weights_flat.clear();
        }

        void set_cost_weight(index_type truth, index_type prediction, float cost)
        {
            cost_weights[truth][prediction] = cost;
            cost_weights_flat.clear();
        }

        const std::vector<float>& get_cost_weights(const index_type& truth) const
        {
            return cost_weights[truth];
        }

        const std::vector<float>& get_cost_weights_flat() const
        {
            if (cost_weights_flat.empty())
            {
                const auto index_count = cost_weights.size();
                cost_weights_flat.resize(index_count * index_count);
                for (index_type truth = 0; truth < index_count; ++truth)
                {
                    for (index_type prediction = 0; prediction < index_count; ++prediction)
                    {
                        cost_weights_flat[truth * index_count + prediction] = cost_weights[truth][prediction];
                    }
                }
            }

            DLIB_ASSERT(cost_weights.size() * cost_weights.size() == cost_weights_flat.size());

            return cost_weights_flat;
        }

        bool empty() const
        {
            return cost_weights.empty();
        }

        void clear()
        {
            cost_weights.clear();
            cost_weights_flat.clear();
        }

        friend void serialize(const cost_weight_matrix_index_based& item, std::ostream& out)
        {
            serialize("cost_weight_matrix_index_based", out);
            serialize(item.cost_weights, out);
        }

        friend void deserialize(cost_weight_matrix_index_based& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "cost_weight_matrix_index_based")
                throw serialization_error("Unexpected version found while deserializing dlib::cost_weight_matrix_index_based.");
            deserialize(item.cost_weights, in);
            
            item.cost_weights_flat.clear();
        }

        static float get_default_cost_weight()
        {
            return 1.f;
        }

    private:
        std::vector<std::vector<float>> cost_weights;

        mutable std::vector<float> cost_weights_flat; // evaluated lazily
    };

// ----------------------------------------------------------------------------------------

    class cost_weight_matrix_label_based
    {
    public:
        using label_type = std::string;

        void set_cost_weight(const label_type& truth, const label_type& prediction, float cost_weight)
        {
            cost_weights[truth][prediction] = cost_weight;
        }

        float get_cost_weight(const label_type& truth, const label_type& prediction) const
        {
            const auto i = cost_weights.find(truth);
            if (i == cost_weights.end())
            {
                return get_default_cost_weight(truth, prediction);
            }
            const auto j = i->second.find(prediction);
            if (j == i->second.end())
            {
                return get_default_cost_weight(truth, prediction);
            }
            return j->second;
        }

        bool empty() const
        {
            return cost_weights.empty();
        }

        void clear()
        {
            cost_weights.clear();
        }

        friend void serialize(const cost_weight_matrix_label_based& costs, std::ostream& out)
        {
            serialize("cost_weight_matrix_label_based", out);
            serialize(costs.cost_weights, out);
        }

        friend void deserialize(cost_weight_matrix_label_based& costs, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "cost_weight_matrix_label_based")
                throw serialization_error("Unexpected version found while deserializing dlib::cost_weight_matrix_label_based.");
            deserialize(costs.cost_weights, in);
        }

    private:
        static float get_default_cost_weight(const label_type& truth, const label_type& prediction)
        {
            return 1.f;
        }

        std::unordered_map<label_type, std::unordered_map<label_type, float>> cost_weights;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_UTILITIES_H_ 



