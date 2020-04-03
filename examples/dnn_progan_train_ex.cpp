// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  I'm assuming you have already read the dnn_dcgan_train_ex.cpp example.
    This example program extends that one by allowing to generate large images.  It
    implements the "progressive GAN" (or "ProGAN") idea from this paper:
    "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
    by Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen.
    (Note that not everything described in the paper is implemented, at least just yet.)

    The main idea is that we first train a generator and a discriminator using highly
    downsampled images (4x4).  When that works, we increase the resolution, and continue
    training.
*/

#include <algorithm>
#include <iostream>
#include <filesystem>

#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/matrix.h>

using namespace std;
using namespace dlib;

// We start by defining a simple visitor to disable bias learning in a network.  By default,
// biases are initialized to 0, so setting the multipliers to 0 disables bias learning.
class visitor_no_bias
{
public:
    template <typename input_layer_type>
    void operator()(size_t , input_layer_type& ) const
    {
        // ignore other layers
    }

    template <typename T, typename U, typename E>
    void operator()(size_t , add_layer<T, U, E>& l) const
    {
        set_bias_learning_rate_multiplier(l.layer_details(), 0);
        set_bias_weight_decay_multiplier(l.layer_details(), 0);
    }
};

// Some helper definitions for the noise generation
const size_t noise_size = 500;
using noise_t = std::array<matrix<float, 1, 1>, noise_size>;

noise_t make_noise(dlib::rand& rnd)
{
    noise_t noise;
    for (auto& n : noise)
    {
        n = rnd.get_random_gaussian();
    }
    return noise;
}

// A convolution with custom padding
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using conp = add_layer<con_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// A transposed convolution with custom padding
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using contp = add_layer<cont_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// Average pooling, 2x2
template<typename SUBNET>
using avg2 = add_layer<avg_pool_<2, 2, 2, 2, 0, 0>, SUBNET>;

// The generator is made of a bunch of deconvolutional layers.  Its input is a 1 x 1 x k noise
// tensor, and the output is the generated image.  The loss layer does not matter for the
// training, we just stack a compatible one on top to be able to have a () operator on the
// generator.

// TODO CLEANUP

constexpr auto max_k = 1024;
constexpr auto min_size = 4;
constexpr auto fc_k = max_k * min_size * min_size;

template <typename SUBNET>
using reshape_to_min_size =
    extract<0, max_k, min_size, min_size, SUBNET>;

// TODO: add some remarks here (why 513, 514, etc, and not 512 as in the ProGAN paper?)
constexpr auto fmaps1 = 64;
constexpr auto fmaps2 = 128;
constexpr auto fmaps3 = 256;
constexpr auto fmaps4 = 512;
constexpr auto fmaps5 = 513;
constexpr auto fmaps6 = 514;
constexpr auto fmaps7 = 515;

using generator_type =
    loss_multiclass_log_per_pixel<sig<
    skip7<conp<3, 1, 1, 0, tag7<mish<bn_con<conp<fmaps1, 3, 1, 1, mish<bn_con<conp<fmaps1, 3, 1, 1, upsample<2, // 2...
    skip6<conp<3, 1, 1, 0, tag6<mish<bn_con<conp<fmaps2, 3, 1, 1, mish<bn_con<conp<fmaps2, 3, 1, 1, upsample<2, // 12...
    skip5<conp<3, 1, 1, 0, tag5<mish<bn_con<conp<fmaps3, 3, 1, 1, mish<bn_con<conp<fmaps3, 3, 1, 1, upsample<2, // 22...
    skip4<conp<3, 1, 1, 0, tag4<mish<bn_con<conp<fmaps4, 3, 1, 1, mish<bn_con<conp<fmaps4, 3, 1, 1, upsample<2, // 32...
    skip3<conp<3, 1, 1, 0, tag3<mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1, upsample<2, // 42...
    skip2<conp<3, 1, 1, 0, tag2<mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1, upsample<2, // 52...
    skip1<conp<3, 1, 1, 0, tag1<mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,  // 62...
    reshape_to_min_size<
    mish<bn_fc<fc<fc_k,
    input<noise_t>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using generator_type_0 =
loss_multiclass_log_per_pixel<
    sig<conp<3, 1, 1, 0,
    mish<bn_con<conp<fmaps7, 3, 1, 1,mish<bn_con<conp<fmaps7, 3, 1, 1,
    reshape_to_min_size<
    mish<bn_fc<fc<fc_k,
    input<noise_t>
    >>>>>>>>>>>>>;

using generator_type_1 =
loss_multiclass_log_per_pixel<
    add_prev2<multiply< // fade-in
    sig<conp<3, 1, 1, 0,
    mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1, upsample<2,
    skip1<
    tag2<multiply<
    upsample<2,
    sig<conp<3, 1, 1, 0,
    tag1<
    mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    reshape_to_min_size<
    mish<bn_fc<fc<fc_k,
    input<noise_t>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using generator_type_2 =
loss_multiclass_log_per_pixel<
    add_prev2<multiply< // fade-in
    sig<conp<3, 1, 1, 0,
    mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1, upsample<2,
    skip1<
    tag2<multiply<
    upsample<2,
    sig<conp<3, 1, 1, 0,
    tag1<
    mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    reshape_to_min_size<
    mish<bn_fc<fc<fc_k,
    input<noise_t>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using generator_type_3 =
loss_multiclass_log_per_pixel<
    add_prev2<multiply< // fade-in
    sig<conp<3, 1, 1, 0,
    mish<bn_con<conp<fmaps4, 3, 1, 1, mish<bn_con<conp<fmaps4, 3, 1, 1, upsample<2,
    skip1<
    tag2<multiply<
    upsample<2,
    sig<conp<3, 1, 1, 0,
    tag1<
    mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    reshape_to_min_size<
    mish<bn_fc<fc<fc_k,
    input<noise_t>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using generator_type_4 =
loss_multiclass_log_per_pixel<
    add_prev2<multiply< // fade-in
    sig<conp<3, 1, 1, 0,
    mish<bn_con<conp<fmaps3, 3, 1, 1, mish<bn_con<conp<fmaps3, 3, 1, 1, upsample<2,
    skip1<
    tag2<multiply<
    upsample<2,
    sig<conp<3, 1, 1, 0,
    tag1<
    mish<bn_con<conp<fmaps4, 3, 1, 1, mish<bn_con<conp<fmaps4, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    reshape_to_min_size<
    mish<bn_fc<fc<fc_k,
    input<noise_t>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using generator_type_5 =
loss_multiclass_log_per_pixel<
    add_prev2<multiply< // fade-in
    sig<conp<3, 1, 1, 0,
    mish<bn_con<conp<fmaps2, 3, 1, 1, mish<bn_con<conp<fmaps2, 3, 1, 1, upsample<2,
    skip1<
    tag2<multiply<
    upsample<2,
    sig<conp<3, 1, 1, 0,
    tag1<
    mish<bn_con<conp<fmaps3, 3, 1, 1, mish<bn_con<conp<fmaps3, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps4, 3, 1, 1, mish<bn_con<conp<fmaps4, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    reshape_to_min_size<
    mish<bn_fc<fc<fc_k,
    input<noise_t>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using generator_type_6 =
loss_multiclass_log_per_pixel<
    add_prev2<multiply< // fade-in
    sig<conp<3, 1, 1, 0,
    mish<bn_con<conp<64, 3, 1, 1, mish<bn_con<conp<64, 3, 1, 1, upsample<2,
    skip1<
    tag2<multiply<
    upsample<2,
    sig<conp<3, 1, 1, 0,
    tag1<
    mish<bn_con<conp<fmaps2, 3, 1, 1, mish<bn_con<conp<fmaps2, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps3, 3, 1, 1, mish<bn_con<conp<fmaps3, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps4, 3, 1, 1, mish<bn_con<conp<fmaps4, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1, upsample<2,
    mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    reshape_to_min_size<
    mish<bn_fc<fc<fc_k,
    input<noise_t>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

// Now, let's proceed to define the discriminator, whose role will be to decide whether an
// image is fake or not.
using discriminator_type =
    loss_binary_log<
    fc<1,mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps4, 3, 1, 1, mish<bn_con<conp<fmaps4, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps3, 3, 1, 1, mish<bn_con<conp<fmaps3, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps2, 3, 1, 1, mish<bn_con<conp<fmaps2, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps1, 3, 1, 1, mish<bn_con<conp<fmaps1, 3, 1, 1,

    // placeholders for faded-in layers - TODO: these need not be serialized
    skip1<
    conp<fmaps1, 3, 1, 1, conp<fmaps1, 3, 1, 1, skip1<
    conp<fmaps2, 3, 1, 1, conp<fmaps2, 3, 1, 1, skip1<
    conp<fmaps3, 3, 1, 1, conp<fmaps3, 3, 1, 1, skip1<
    conp<fmaps4, 3, 1, 1, conp<fmaps4, 3, 1, 1, skip1<
    conp<fmaps5, 3, 1, 1, conp<fmaps5, 3, 1, 1, skip1<
    conp<fmaps6, 3, 1, 1, conp<fmaps6, 3, 1, 1, skip1<
    conp<fmaps7, 3, 1, 1, conp<fmaps7, 3, 1, 1, tag1<
    input<matrix<rgb_pixel>>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using discriminator_type_0 =
loss_binary_log<
    fc<1,mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    input<matrix<rgb_pixel>>
    >>>>>>>>;

using discriminator_type_1 =
loss_binary_log<
    fc<1,
    add_prev2<multiply< // fade-in
    mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1,
    skip1<tag2<multiply<
    mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    avg2<
    tag1<
    input<matrix<rgb_pixel>>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using discriminator_type_2 =
loss_binary_log<
    fc<1,
    avg2<mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    add_prev2<multiply< // fade-in
    mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1,
    skip1<tag2<multiply<
    mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1,
    avg2<
    tag1<
    input<matrix<rgb_pixel>>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using discriminator_type_3 =
loss_binary_log<
    fc<1,
    avg2<mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1,
    add_prev2<multiply< // fade-in
    mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps4, 3, 1, 1, mish<bn_con<conp<fmaps4, 3, 1, 1,
    skip1<tag2<multiply<
    mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1,
    avg2<
    tag1<
    input<matrix<rgb_pixel>>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using discriminator_type_4 =
loss_binary_log<
    fc<1,
    avg2<mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1,
    add_prev2<multiply< // fade-in
    mish<bn_con<conp<fmaps4, 3, 1, 1, mish<bn_con<conp<fmaps4, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps3, 3, 1, 1, mish<bn_con<conp<fmaps3, 3, 1, 1,
    skip1<tag2<multiply<
    mish<bn_con<conp<fmaps4, 3, 1, 1, mish<bn_con<conp<fmaps4, 3, 1, 1,
    avg2<
    tag1<
    input<matrix<rgb_pixel>>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using discriminator_type_5 =
loss_binary_log<
    fc<1,
    avg2<mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps4, 3, 1, 1, mish<bn_con<conp<fmaps4, 3, 1, 1,
    add_prev2<multiply< // fade-in
    mish<bn_con<conp<fmaps3, 3, 1, 1, mish<bn_con<conp<fmaps3, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps2, 3, 1, 1, mish<bn_con<conp<fmaps2, 3, 1, 1,
    skip1<tag2<multiply<
    mish<bn_con<conp<fmaps3, 3, 1, 1, mish<bn_con<conp<fmaps3, 3, 1, 1,
    avg2<
    tag1<
    input<matrix<rgb_pixel>>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using discriminator_type_6 =
loss_binary_log<
    fc<1,
    avg2<mish<bn_con<conp<fmaps7, 3, 1, 1, mish<bn_con<conp<fmaps7, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps6, 3, 1, 1, mish<bn_con<conp<fmaps6, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps5, 3, 1, 1, mish<bn_con<conp<fmaps5, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps4, 3, 1, 1, mish<bn_con<conp<fmaps4, 3, 1, 1, 
    avg2<mish<bn_con<conp<fmaps3, 3, 1, 1, mish<bn_con<conp<fmaps3, 3, 1, 1,
    add_prev2<multiply< // fade-in
    mish<bn_con<conp<fmaps2, 3, 1, 1, mish<bn_con<conp<fmaps2, 3, 1, 1,
    avg2<mish<bn_con<conp<fmaps1, 3, 1, 1, mish<bn_con<conp<fmaps1, 3, 1, 1,
    skip1<tag2<multiply<
    mish<bn_con<conp<fmaps2, 3, 1, 1, mish<bn_con<conp<fmaps2, 3, 1, 1,
    avg2<
    tag1<
    input<matrix<rgb_pixel>>
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

// Some helper functions to generate and get the images from the generator
std::vector<matrix<rgb_pixel>> get_generated_images(const tensor& out)
{
    std::vector<matrix<rgb_pixel>> images;
    for (long long n = 0; n < out.num_samples(); ++n)
    {
        matrix<rgb_pixel> image(out.nr(), out.nc());
        for (size_t k = 0; k < 3; ++k)
        {
            matrix<float> output = image_plane(out, n, k);
            for (long y = 0; y < output.nr(); ++y)
            {
                for (long x = 0; x < output.nc(); ++x)
                {
                    const auto value = 255 * std::max(0.f, std::min(1.f, output(y, x)));
                    switch (k) {
                    case 0: image(y, x).red   = value; break;
                    case 1: image(y, x).green = value; break;
                    case 2: image(y, x).blue  = value; break;
                    default: throw std::runtime_error("Unexpected k: " + std::to_string(k));
                    }
                }
            }
        }
        images.push_back(std::move(image));
    }
    return images;
}

template <typename generator_type>
matrix<rgb_pixel> generate_image(generator_type& net, const noise_t& noise)
{
    resizable_tensor noise_tensor;
    const std::vector<noise_t> noises = { noise };
    net.to_tensor(noises.begin(), noises.end(), noise_tensor);
    const auto& output = net.forward(noise_tensor);
    const auto images = get_generated_images(output);
    DLIB_CASSERT(images.size() == 1);
    return images.front();
}

struct training_state {
    generator_type full_generator;
    discriminator_type full_discriminator;
    size_t iteration = 0;
    const size_t approximate_samples_per_progression = 1e6;
    long long max_nr = 0;
    long long max_nc = 0;
    long long current_nr = 0;
    long long current_nc = 0;
    std::mutex cout_mutex;
    std::mutex rnd_mutex;
    dlib::rand rnd;
    dlib::pipe<matrix<rgb_pixel>> training_images;
    
    training_state()
        : training_images(200)
    {}
};

template <size_t PROGRESSION, size_t LAYER, typename GENERATOR>
struct swap_g {
    static void swap(GENERATOR& generator, generator_type& full_generator)
    {
        DLIB_CASSERT(layer<21 + 7 * (PROGRESSION - LAYER)>(generator).layer_details().num_filters() == layer<77 - 10 * LAYER>(full_generator).layer_details().num_filters());
        DLIB_CASSERT(layer<24 + 7 * (PROGRESSION - LAYER)>(generator).layer_details().num_filters() == layer<80 - 10 * LAYER>(full_generator).layer_details().num_filters());

        std::swap(layer<19 + 7 * (PROGRESSION - LAYER)>(generator).layer_details(), layer<75 - 10 * LAYER>(full_generator).layer_details());
        std::swap(layer<20 + 7 * (PROGRESSION - LAYER)>(generator).layer_details(), layer<76 - 10 * LAYER>(full_generator).layer_details());
        std::swap(layer<21 + 7 * (PROGRESSION - LAYER)>(generator).layer_details(), layer<77 - 10 * LAYER>(full_generator).layer_details());

        std::swap(layer<22 + 7 * (PROGRESSION - LAYER)>(generator).layer_details(), layer<78 - 10 * LAYER>(full_generator).layer_details());
        std::swap(layer<23 + 7 * (PROGRESSION - LAYER)>(generator).layer_details(), layer<79 - 10 * LAYER>(full_generator).layer_details());
        std::swap(layer<24 + 7 * (PROGRESSION - LAYER)>(generator).layer_details(), layer<80 - 10 * LAYER>(full_generator).layer_details());

        swap_g<PROGRESSION, LAYER - 1, GENERATOR>::swap(generator, full_generator);
    }
};

template <size_t PROGRESSION, typename GENERATOR>
struct swap_g<PROGRESSION, 0, GENERATOR> {
    static void swap(GENERATOR& generator, generator_type& full_generator)
    {
        ;
    }
};

template <size_t PROGRESSION, typename GENERATOR>
void swap_gen(GENERATOR& generator, generator_type& full_generator)
{
    DLIB_CASSERT(layer<7>(generator).layer_details().num_filters() == layer<70 - 10 * PROGRESSION>(full_generator).layer_details().num_filters());

    //std::swap(layer<5>(generator).layer_details(), layer<68 - 10 * PROGRESSION>(full_generator).layer_details());
    //std::swap(layer<6>(generator).layer_details(), layer<69 - 10 * PROGRESSION>(full_generator).layer_details());
    std::swap(layer<7>(generator).layer_details(), layer<70 - 10 * PROGRESSION>(full_generator).layer_details());

    swap_g<PROGRESSION, PROGRESSION, GENERATOR>::swap(generator, full_generator);

    DLIB_CASSERT(layer<20 + 1 + 7 * PROGRESSION>(generator).layer_details().get_num_outputs() == layer<74>(full_generator).layer_details().get_num_outputs());

    std::swap(layer<18 + 1 + 7 * PROGRESSION>(generator).layer_details(), layer<72>(full_generator).layer_details());
    std::swap(layer<19 + 1 + 7 * PROGRESSION>(generator).layer_details(), layer<73>(full_generator).layer_details());
    std::swap(layer<20 + 1 + 7 * PROGRESSION>(generator).layer_details(), layer<74>(full_generator).layer_details());
}

template <size_t PROGRESSION, size_t LAYER, typename DISCRIMINATOR>
struct swap_d {
    static void swap(DISCRIMINATOR& discriminator, discriminator_type& full_discriminator)
    {
        swap_d<PROGRESSION, LAYER - 1, DISCRIMINATOR>::swap(discriminator, full_discriminator);

        DLIB_CASSERT(layer<-1 + 7 * LAYER - (LAYER < PROGRESSION ? 1 : 0)>(discriminator).layer_details().num_filters() == layer<-3 + 7 * LAYER>(full_discriminator).layer_details().num_filters());
        DLIB_CASSERT(layer< 2 + 7 * LAYER - (LAYER < PROGRESSION ? 1 : 0)>(discriminator).layer_details().num_filters() == layer< 0 + 7 * LAYER>(full_discriminator).layer_details().num_filters());

        std::swap(layer<-3 + 7 * LAYER - (LAYER < PROGRESSION ? 1 : 0)>(discriminator).layer_details(), layer<-5 + 7 * LAYER>(full_discriminator).layer_details()); // mish
        std::swap(layer<-2 + 7 * LAYER - (LAYER < PROGRESSION ? 1 : 0)>(discriminator).layer_details(), layer<-4 + 7 * LAYER>(full_discriminator).layer_details()); // 
        std::swap(layer<-1 + 7 * LAYER - (LAYER < PROGRESSION ? 1 : 0)>(discriminator).layer_details(), layer<-3 + 7 * LAYER>(full_discriminator).layer_details()); // 

        std::swap(layer<0 + 7 * LAYER - (LAYER < PROGRESSION ? 1 : 0)>(discriminator).layer_details(), layer<-2 + 7 * LAYER>(full_discriminator).layer_details()); // mish
        std::swap(layer<1 + 7 * LAYER - (LAYER < PROGRESSION ? 1 : 0)>(discriminator).layer_details(), layer<-1 + 7 * LAYER>(full_discriminator).layer_details()); // 
        std::swap(layer<2 + 7 * LAYER - (LAYER < PROGRESSION ? 1 : 0)>(discriminator).layer_details(), layer< 0 + 7 * LAYER>(full_discriminator).layer_details()); // 
    }
};

template <size_t PROGRESSION, typename DISCRIMINATOR>
struct swap_d<PROGRESSION, 0, DISCRIMINATOR> {
    static void swap(DISCRIMINATOR& discriminator, discriminator_type& full_discriminator)
    {
        ;
    }
};

template <size_t PROGRESSION, typename DISCRIMINATOR>
void swap_dis(DISCRIMINATOR& discriminator, discriminator_type& full_discriminator)
{
    swap_d<PROGRESSION, PROGRESSION, DISCRIMINATOR>::swap(discriminator, full_discriminator);
}

template <size_t PROGRESSION, typename generator_type, typename discriminator_type>
void train_progression_round(training_state& training_state)
{
    generator_type generator;
    discriminator_type discriminator;

    auto& full_generator = training_state.full_generator;
    auto& full_discriminator = training_state.full_discriminator;

    const auto swap_layers = [&](int step) {

        if constexpr (PROGRESSION == 0)
        {
            //std::swap(layer<1>(generator).layer_details(), layer<44>(full_generator).layer_details()); // sig
            std::swap(layer<2>(generator).layer_details(), layer<63>(full_generator).layer_details()); // contp

            DLIB_CASSERT(layer<5>(generator).layer_details().num_filters() == layer<67>(full_generator).layer_details().num_filters());
            DLIB_CASSERT(layer<8>(generator).layer_details().num_filters() == layer<70>(full_generator).layer_details().num_filters());

            DLIB_CASSERT(layer<12>(generator).layer_details().get_num_outputs() == layer<74>(full_generator).layer_details().get_num_outputs());

            std::swap(layer<3>(generator).layer_details(), layer<65>(full_generator).layer_details()); // mish
            std::swap(layer<4>(generator).layer_details(), layer<66>(full_generator).layer_details()); // bn_con
            std::swap(layer<5>(generator).layer_details(), layer<67>(full_generator).layer_details()); // conp

            std::swap(layer<6>(generator).layer_details(), layer<68>(full_generator).layer_details()); // mish
            std::swap(layer<7>(generator).layer_details(), layer<69>(full_generator).layer_details()); // bn_con
            std::swap(layer<8>(generator).layer_details(), layer<70>(full_generator).layer_details()); // conp

            std::swap(layer<10>(generator).layer_details(), layer<72>(full_generator).layer_details()); // mish
            std::swap(layer<11>(generator).layer_details(), layer<73>(full_generator).layer_details()); // bn_fc
            std::swap(layer<12>(generator).layer_details(), layer<74>(full_generator).layer_details()); // fc
        }
        else {
            if (step == 0) {
                std::swap(layer<16>(generator).layer_details(), layer<1>(full_generator).layer_details()); // sig
                std::swap(layer<17>(generator).layer_details(), layer<73 - 10 * PROGRESSION>(full_generator).layer_details()); // contp
            }
            else {
                std::swap(layer<3>(generator).layer_details(), layer<1>(full_generator).layer_details()); // sig
                std::swap(layer<4>(generator).layer_details(), layer<63 - 10 * PROGRESSION>(full_generator).layer_details()); // contp
            }

            swap_gen<PROGRESSION>(generator, full_generator);
        }

        DLIB_CASSERT(layer<1>(discriminator).layer_details().get_num_outputs() == layer<1>(full_discriminator).layer_details().get_num_outputs());
        std::swap(layer<1>(discriminator).layer_details(), layer<1>(full_discriminator).layer_details()); // fc

        if constexpr (PROGRESSION == 0) {
            DLIB_CASSERT(layer<4>(discriminator).layer_details().num_filters() == layer<69>(full_discriminator).layer_details().num_filters());
            DLIB_CASSERT(layer<7>(discriminator).layer_details().num_filters() == layer<70>(full_discriminator).layer_details().num_filters());

            //std::swap(layer<2>(discriminator).layer_details(), layer<2>(full_discriminator).layer_details()); // mish
            //std::swap(layer<3>(discriminator).layer_details(), layer<3>(full_discriminator).layer_details()); // bn_con
            std::swap(layer<4>(discriminator).layer_details(), layer<69>(full_discriminator).layer_details()); // conp

            //std::swap(layer<5>(discriminator).layer_details(), layer<2>(full_discriminator).layer_details()); // mish
            //std::swap(layer<6>(discriminator).layer_details(), layer<3>(full_discriminator).layer_details()); // bn_con
            std::swap(layer<7>(discriminator).layer_details(), layer<70>(full_discriminator).layer_details()); // conp
        }
        else {
            DLIB_CASSERT(layer<5 + (PROGRESSION == 1 ? 1 : 0)>(discriminator).layer_details().num_filters() == layer<4>(full_discriminator).layer_details().num_filters());
            DLIB_CASSERT(layer<12 + (PROGRESSION <= 2 ? 1 : 0)>(discriminator).layer_details().num_filters() == layer<11>(full_discriminator).layer_details().num_filters());

            DLIB_CASSERT((layer<5 + (PROGRESSION == 1 ? 1 : 0)>(discriminator).layer_details().get_layer_params().size() == 0) == (step == 0));

            //std::swap(layer<10 + 4 * PROGRESSION>(discriminator).layer_details(), target_layer_0.layer_details());
            //std::swap(layer<11 + 4 * PROGRESSION>(discriminator).layer_details(), target_layer_1.layer_details());

            DLIB_CASSERT(layer<15 + 7 * PROGRESSION>(discriminator).layer_details().num_filters() == layer<72 - 3 * PROGRESSION>(full_discriminator).layer_details().num_filters());
            DLIB_CASSERT(layer<18 + 7 * PROGRESSION>(discriminator).layer_details().num_filters() == layer<73 - 3 * PROGRESSION>(full_discriminator).layer_details().num_filters());

            std::swap(layer<15 + 7 * PROGRESSION>(discriminator).layer_details(), layer<72 - 3 * PROGRESSION>(full_discriminator).layer_details());
            std::swap(layer<18 + 7 * PROGRESSION>(discriminator).layer_details(), layer<73 - 3 * PROGRESSION>(full_discriminator).layer_details());

            //std::swap(layer<0 + 4 * PROGRESSION>(discriminator).layer_details(), target_layer_0.layer_details());
            //std::swap(layer<1 + 4 * PROGRESSION>(discriminator).layer_details(), target_layer_1.layer_details());
            //std::swap(layer<2 + 4 * PROGRESSION>(discriminator).layer_details(), layer<0 + 4 * PROGRESSION>(full_discriminator).layer_details());

            //std::swap(layer<4 + 7 * PROGRESSION>(discriminator).layer_details(), layer<2 + 4 * PROGRESSION>(full_discriminator).layer_details()); // mish
            //std::swap(layer<5 + 7 * PROGRESSION>(discriminator).layer_details(), layer<3 + 4 * PROGRESSION>(full_discriminator).layer_details()); // bn_con
            std::swap(layer<6 + 7 * PROGRESSION>(discriminator).layer_details(), layer<69 - 3 * PROGRESSION>(full_discriminator).layer_details()); // conp

            //std::swap(layer<7 + 7 * PROGRESSION>(discriminator).layer_details(), layer<2 + 4 * PROGRESSION>(full_discriminator).layer_details()); // mish
            //std::swap(layer<8 + 7 * PROGRESSION>(discriminator).layer_details(), layer<3 + 4 * PROGRESSION>(full_discriminator).layer_details()); // bn_con
            std::swap(layer<9 + 7 * PROGRESSION>(discriminator).layer_details(), layer<70 - 3 * PROGRESSION>(full_discriminator).layer_details()); // conp

            swap_dis<PROGRESSION>(discriminator, full_discriminator);

            DLIB_CASSERT((layer<5 + (PROGRESSION == 1 ? 1 : 0)>(discriminator).layer_details().get_layer_params().size() > 0) == (step == 0));
        }
    };

    swap_layers(0);

    const auto generated_image = generate_image(generator, make_noise(training_state.rnd));

    cout << "Current image size: " << training_state.current_nr << " x " << training_state.current_nc << endl << endl;

    cout << "current generator" << endl;
    cout << generator << endl;

    discriminator(generated_image);

    cout << "current discriminator" << endl;
    cout << discriminator << endl;

    DLIB_CASSERT(generated_image.nr() == training_state.current_nr);
    DLIB_CASSERT(generated_image.nc() == training_state.current_nc);

    dlib::image_window training_samples_window, generated_samples_window;

    // The solvers for the generator and discriminator networks.
    std::vector<adam> g_solvers(generator.num_computational_layers, adam(0, 0.5, 0.999));
    std::vector<adam> d_solvers(discriminator.num_computational_layers, adam(0, 0.5, 0.999));

    const double g_learning_rate = 2.0e-4;
    const double d_learning_rate = 1.0e-4;

    const auto get_minibatch_size = []() {
        switch (PROGRESSION) {
        case 0: return 25*25;
        case 1: return 15*15;
        case 2: return 7*7;
        case 3: return 5*5;
        case 4: return 4*4;
        case 5: return 3*3;
        case 6: return 2*2;
        default: throw std::runtime_error("Unsupported progression round: " + std::to_string(PROGRESSION));
        }
    };

    const auto minibatch_size = get_minibatch_size();

    const auto max_iter = std::max(2ull, static_cast<size_t>(std::round(
        training_state.approximate_samples_per_progression / static_cast<double>(minibatch_size)
    )));

    const auto bn_window = std::max(1ull, std::min(static_cast<size_t>(max_iter * 0.1), 100ull));
    const float warmup_iter = PROGRESSION > 0 ? bn_window : 0.f;

    set_all_bn_running_stats_window_sizes(generator, bn_window);
    set_all_bn_running_stats_window_sizes(discriminator, bn_window);

    const std::vector<float> real_labels(minibatch_size, 1);
    const std::vector<float> fake_labels(minibatch_size, -1);

    resizable_tensor real_samples_tensor, fake_samples_tensor, noises_tensor;

    running_stats<double> g_loss, d_loss;
    while (training_state.iteration < max_iter)
    {
        const auto alpha = std::clamp(
            (training_state.iteration - warmup_iter) / (max_iter / 2.f - warmup_iter),
            0.f,
            1.f
        );

        if constexpr (PROGRESSION > 0)
        {
            // Fade-in
            const auto alpha = std::clamp(
                (training_state.iteration - warmup_iter) / (max_iter / 2.f - warmup_iter),
                0.f,
                1.f
            );

            // Generator
            layer<2>(generator).layer_details().set_multiply_value(alpha);
            layer<14>(generator).layer_details().set_multiply_value(1.f - alpha);

            // Discriminator
            layer<-4 + 7 * PROGRESSION>(discriminator).layer_details().set_multiply_value(alpha);
            layer<12 + 7 * PROGRESSION>(discriminator).layer_details().set_multiply_value(1.f - alpha);
        }

        // Train the discriminator with real images
        std::vector<matrix<rgb_pixel>> real_samples;
        while (real_samples.size() < minibatch_size)
        {
            dlib::matrix<dlib::rgb_pixel> img;
            if (training_state.training_images.dequeue(img)) {
                real_samples.push_back(std::move(img));
            }
        }

        // Visualize the training samples
        training_samples_window.set_image(tile_images(real_samples));
        training_samples_window.set_title("Training images, step#: " + to_string(training_state.iteration));

        // The following lines are equivalent to calling train_one_step(real_samples, real_labels)
        discriminator.to_tensor(real_samples.begin(), real_samples.end(), real_samples_tensor);
        d_loss.add(discriminator.compute_loss(real_samples_tensor, real_labels.begin()));
        discriminator.back_propagate_error(real_samples_tensor);
        discriminator.update_parameters(d_solvers, training_state.iteration < warmup_iter ? 0.0 : d_learning_rate);

        // Train the discriminator with fake images
        // 1. Generate some random noise
        std::vector<noise_t> noises;
        while (noises.size() < minibatch_size)
        {
            noises.push_back(make_noise(training_state.rnd));
        }
        // 2. Convert noises into a tensor 
        generator.to_tensor(noises.begin(), noises.end(), noises_tensor);
        // 3. Forward the noise through the network and convert the outputs into images
        const auto fake_samples = get_generated_images(generator.forward(noises_tensor));
        // 4. Finally train the discriminator
        discriminator.to_tensor(fake_samples.begin(), fake_samples.end(), fake_samples_tensor);
        d_loss.add(discriminator.compute_loss(fake_samples_tensor, fake_labels.begin()));
        discriminator.back_propagate_error(fake_samples_tensor);
        discriminator.update_parameters(d_solvers, training_state.iteration < warmup_iter ? 0.0 : d_learning_rate);

        // Visualize the generated samples
        generated_samples_window.set_image(tile_images(fake_samples));
        generated_samples_window.set_title("Generated images, step#: " + to_string(training_state.iteration));

        // Train the generator
        // 1. Forward the fake samples and compute the loss with real labels
        g_loss.add(discriminator.compute_loss(fake_samples_tensor, real_labels.begin()));
        // 2. Back propagate the error to fill the final data gradient
        discriminator.back_propagate_error(fake_samples_tensor);
        // 3. Get the gradient that will tell the generator how to update itself
        const tensor& d_grad = discriminator.get_final_data_gradient();
        generator.back_propagate_error(noises_tensor, d_grad);
        generator.update_parameters(g_solvers, training_state.iteration < warmup_iter ? 0.0 : g_learning_rate);

        ++training_state.iteration;

        // Periodically save the results
        if (training_state.iteration % 100 == 0 || training_state.iteration <= 10 || (training_state.iteration >= 90 && training_state.iteration <= 110) || (training_state.iteration <= 150 && training_state.iteration % 10 == 0) || training_state.iteration == max_iter)
        {
            //serialize_to_sync_file();

            {
                lock_guard<std::mutex> cout_lock(training_state.cout_mutex);
                cout <<
                    "p: " << PROGRESSION <<
                    "\ti: " << training_state.iteration << " / " << max_iter <<
                    "\td: " << d_loss.mean() * 2 <<
                    "\tg: " << g_loss.mean();

                if (PROGRESSION > 0) {
                    cout << "\talpha: " << alpha;
                }

                cout << endl;
            }

            d_loss.clear();
            g_loss.clear();
        }

        if (training_state.iteration % 100 == 0 || training_state.iteration <= 10 || (training_state.iteration >= 90 && training_state.iteration <= 110) || (training_state.iteration <= 150 && training_state.iteration % 10 == 0) || training_state.iteration == max_iter)
        {
            const std::string base_filename = "p" + std::to_string(PROGRESSION) + "_i" + to_string(training_state.iteration);

            save_png(tile_images(real_samples), "training_images/" + base_filename + ".png");
            save_png(tile_images(fake_samples), "generated_images/" + base_filename + ".png");

            if (training_state.iteration == max_iter) {

                // GENERATE AGAIN - THIS CAN BE REMOVED LATER ON
                const auto fake_samples = get_generated_images(generator.forward(noises_tensor));

                save_png(tile_images(fake_samples), "generated_images/" + base_filename + "_.png");
            }
        }
    }

    generator.clean();
    discriminator.clean();

    swap_layers(1);
}

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        cout << "This example needs image data to run!" << endl;
        cout << "Give a folder containing images as input to this program." << endl;
        return EXIT_FAILURE;
    }

    training_state training_state;

    auto& full_generator = training_state.full_generator;
    auto& full_discriminator = training_state.full_discriminator;

    // Remove the bias learning from the networks
    visit_layers(full_generator, visitor_no_bias());
    visit_layers(full_discriminator, visitor_no_bias());

    // Forward random noise so that we see the tensor size at each layer
    const auto full_size_generated_image = generate_image(full_generator, make_noise(training_state.rnd));

    cout << "full generator" << endl;
    cout << full_generator << endl;

    full_discriminator(full_size_generated_image);

    cout << "full discriminator" << endl;
    cout << full_discriminator << endl;

    training_state.max_nr = full_size_generated_image.nr();
    training_state.max_nc = full_size_generated_image.nc();

    cout << "EVENTUAL IMAGE SIZE: " << training_state.max_nr << " x " << training_state.max_nc << endl << endl;

    // As said already, start from generating very small images
    training_state.current_nr = 4;
    training_state.current_nc = 4;

    cout << "Finding images..." << endl;

    dlib::pipe<file> found_files(20000);

    class program_closing : public std::exception {};

    std::thread file_finder([&]() {
        size_t files_found_counter = 0;
        chrono::steady_clock::time_point time_last_logged;
        chrono::milliseconds logging_interval(10);

        const auto match = dlib::match_endings(".jpg .jpeg .png");

        for (const auto& i : std::filesystem::recursive_directory_iterator(argv[1])) {
            if (!found_files.is_enabled()) {
                break;
            }

            if (i.is_regular_file()) {
                const auto filename = i.path().string();
                if (match(filename)) {
                    found_files.enqueue(file(filename));

                    ++files_found_counter;

                    const auto now = chrono::steady_clock::now();
                    if (now - time_last_logged >= logging_interval)
                    {
                        time_last_logged = now;
                        logging_interval *= 2;

                        lock_guard<std::mutex> cout_lock(training_state.cout_mutex);
                        cout << "Found " << files_found_counter
                            << " file" << (files_found_counter > 1 ? "s" : "")
                            << " so far..." << endl;
                    }
                }
            }
        }

        if (found_files.is_enabled()) {
            lock_guard<std::mutex> cout_lock(training_state.cout_mutex);
            cout << "Found " << files_found_counter << " files in total" << endl;
        }
    });

    const auto get_random_64bit_number = [&]() {
        lock_guard<std::mutex> lock(training_state.rnd_mutex);
        return training_state.rnd.get_random_64bit_number();
    };

    deque<file> files;
    std::mutex files_mutex;

    const auto get_random_file = [&]() {
        const auto random_number = get_random_64bit_number();

        lock_guard<std::mutex> files_lock(files_mutex);

        file new_file;

        const auto get_new_file = [&]() {
            if (files.empty())
                // We need a new file
                return found_files.dequeue(new_file);

            // We don't necessarily need a new file, but we can try
            return found_files.dequeue_or_timeout(new_file, 0);
        };

        // Read from queue any and all files that may possibly have
        // been found since we were here last time
        while (get_new_file())
            files.push_back(new_file);

        if (files.empty()) // Sanity check
        {
            assert(!found_files.is_enabled());
            throw program_closing();
        }

        return files[random_number % files.size()];
    };

    // Resume training from last sync file
    size_t progression_round = 0;

    const auto sync_filename = "progan_sync.dat";

    // To help keep serialization and deserialization synchronized,
    // define these functions right next to each other
    const auto serialize_to_sync_file = [&]() {
        serialize(sync_filename)
            << full_generator << full_discriminator
            << training_state.current_nr << training_state.current_nc
            << progression_round << training_state.iteration;
    };
    const auto deserialize_from_sync_file = [&]() {
        deserialize(sync_filename)
            >> full_generator >> full_discriminator
            >> training_state.current_nr >> training_state.current_nc
            >> progression_round >> training_state.iteration;
    };

    if (file_exists(sync_filename))
        deserialize_from_sync_file();

    bool has_error = false;

    // Progressively increase the image size
    while (training_state.current_nr <= training_state.max_nr && training_state.current_nc <= training_state.max_nc && !has_error) {

        // Start a bunch of threads that read images from disk and resize them to the currently
        // desired resolution

        auto keep_loading_images = [&](time_t seed)
        {
            dlib::rand rnd(time(0) + seed);

            const auto max_nr = training_state.max_nr;
            const auto max_nc = training_state.max_nc;

            const auto current_nr = training_state.current_nr;
            const auto current_nc = training_state.current_nc;

            try {
                while (training_state.training_images.is_enabled())
                {
                    const auto filename = get_random_file();
                    matrix<rgb_pixel> img;
                    try {
                        load_image(img, filename);

                        // Accept only images that are large enough to begin with
                        // (alternatively, small images could well be upsaled)
                        if (img.nr() >= max_nr && img.nc() >= max_nc)
                        {
                            // Find a suitable crop
                            const auto left = rnd.get_integer_in_range(0, img.nc() - max_nc);
                            const auto top = rnd.get_integer_in_range(0, img.nr() - max_nr);
                            const rectangle rect(left, top, left + max_nc - 1, top + max_nr - 1);
                            const auto crop = sub_image(img, rect);

                            // Resize to the currently desired size
                            matrix<rgb_pixel> resized_crop(current_nr, current_nc);
                            resize_image(crop, resized_crop);
                            training_state.training_images.enqueue(resized_crop);
                        }
                    }
                    catch (std::exception& e) {
                        cerr << "File " << filename << ": " << e.what() << endl;
                    }
                }
            }
            catch (program_closing&) {
                ;
            }
        };

        std::deque<std::thread> data_loaders;
        for (int i = 0; i < 4; ++i) {
            data_loaders.emplace_back(
                [keep_loading_images, i]() {
                    keep_loading_images(i);
                }
            );
        }

        try {
            switch (progression_round) {
            case 0: train_progression_round<0, generator_type_0, discriminator_type_0>(training_state); break;
            case 1: train_progression_round<1, generator_type_1, discriminator_type_1>(training_state); break;
            case 2: train_progression_round<2, generator_type_2, discriminator_type_2>(training_state); break;
            case 3: train_progression_round<3, generator_type_3, discriminator_type_3>(training_state); break;
            case 4: train_progression_round<4, generator_type_4, discriminator_type_4>(training_state); break;
            case 5: train_progression_round<5, generator_type_5, discriminator_type_5>(training_state); break;
            case 6: train_progression_round<6, generator_type_6, discriminator_type_6>(training_state); break;
            default: throw std::runtime_error("Unsupported progression round: " + std::to_string(progression_round));
            }

            training_state.iteration = 0;
            training_state.current_nr *= 2;
            training_state.current_nc *= 2;
            ++progression_round;
        }
        catch (exception& e) {
            cout << e.what() << endl;
            has_error = true;
        }

        training_state.training_images.disable();
        for (auto& data_loader : data_loaders) {
            data_loader.join();
        }

        // Clear the queue
        training_state.training_images.enable();
        matrix<rgb_pixel> dummy_image;
        while (training_state.training_images.dequeue_or_timeout(dummy_image, 0))
            ;
    }

    if (!has_error)
    {
        // Once the training has finished, we don't need the discriminator any more. We just keep the
        // generator.
        full_generator.clean();
        serialize("progan.dnn") << full_generator;

        // To test the generator, we just forward some random noise through it and visualize the
        // output.
        dlib::image_window win;
        while (!win.is_closed())
        {
            win.set_image(generate_image(full_generator, make_noise(training_state.rnd)));
            cout << "Hit enter to generate a new image";
            cin.get();
        }
    }

    found_files.disable();
    file_finder.join();

    return has_error
        ? EXIT_FAILURE
        : EXIT_SUCCESS;
}
catch(exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
