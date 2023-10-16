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

    class cost_matrix
    {
    public:
        using label_type = size_t;

        cost_matrix() {}

        cost_matrix(label_type max_label_count)
        {
            resize(max_label_count);
        }

        void resize(label_type max_label_count)
        {
            costs.resize(max_label_count);

            for (auto& column : costs)
            {
                column.resize(max_label_count);
            }
        }

        void set_cost(label_type truth, label_type prediction, float cost)
        {
            DLIB_CASSERT(truth != prediction || cost == 0.f);
            costs[truth][prediction] = cost;
        }

        float get_cost(label_type truth, label_type prediction) const
        {
            DLIB_CASSERT(truth != prediction);
            return costs[truth][prediction];
        }

        // The returned tensor is valid only until the next call is made.
        template <
            typename const_label_iterator
        >
        const dlib::tensor& add_cost(const_label_iterator truth, const tensor& input) const
        {
            if (empty())
            {
                output_buffer.clear();
                return input;
            }

            output_buffer.copy_size(input);

            const float* inp = input.host();
            float* outp = output_buffer.host();

            for (long i = 0; i < input.num_samples(); ++i, ++truth)
            {
                for (long r = 0; r < input.nr(); ++r)
                {
                    for (long c = 0; c < input.nc(); ++c)
                    {
                        const auto& weighted_label = truth->operator()(r, c);
                        const auto y = weighted_label.label;

                        if (y < costs.size())
                        {
                            const std::vector<float>& costs_column = costs[y];

                            for (long k = 0; k < input.k(); ++k)
                            {
                                DLIB_ASSERT(y != k || costs_column[k] == 0.f);

                                const size_t idx = tensor_index(input, i, k, r, c);
                                DLIB_ASSERT(idx == tensor_index(output_buffer, i, k, r, c));

                                outp[idx] = inp[idx] + costs_column[k];
                            }
                        }
                        else
                        {
                            DLIB_ASSERT(y == std::numeric_limits<decltype(y)>::max());

                            for (long k = 0; k < input.k(); ++k)
                            {
                                const size_t idx = tensor_index(input, i, k, r, c);
                                DLIB_ASSERT(idx == tensor_index(output_buffer, i, k, r, c));

                                outp[idx] = inp[idx];
                            }
                        }
                    }
                }
            }

            return output_buffer;
        }

        bool empty() const
        {
            return costs.empty();
        }

        void clear()
        {
            return costs.clear();
        }

        friend void serialize(const cost_matrix& costs, std::ostream& out)
        {
            serialize("cost_matrix", out);
            serialize(costs.costs, out);
        }

        friend void deserialize(cost_matrix& costs, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "cost_matrix")
                throw serialization_error("Unexpected version found while deserializing dlib::cost_matrix.");
            deserialize(costs.costs, in);
        }

    private:
        std::vector<std::vector<float>> costs;

        mutable dlib::resizable_tensor output_buffer;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_UTILITIES_H_ 



