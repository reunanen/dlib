// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    Semantic segmentation using the PASCAL VOC2012 dataset.

    In segmentation, the task is to assign each pixel of an input image
    a label - for example, 'dog'.  Then, the idea is that neighboring
    pixels having the same label can be connected together to form a
    larger region, representing a complete (or partially occluded) dog.
    So technically, segmentation can be viewed as classification of
    individual pixels (using the relevant context in the input images),
    however the goal usually is to identify meaningful regions that
    represent complete entities of interest (such as dogs).

    Instructions how to run the example:
    1. Download the PASCAL VOC2012 data, and untar it somewhere.
       http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    2. Build the dnn_semantic_segmentation_train_ex example program.
    3. Run:
       ./dnn_semantic_segmentation_train_ex /path/to/VOC2012
    4. Wait while the network is being trained.
    5. Build the dnn_semantic_segmentation_ex example program.
    6. Run:
       ./dnn_semantic_segmentation_ex /path/to/VOC2012-or-other-images

    An alternative to steps 2-4 above is to download a pre-trained network
    from here: http://dlib.net/files/semantic_segmentation_voc2012net_v2.dnn

    It would be a good idea to become familiar with dlib's DNN tooling before reading this
    example.  So you should read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp
    before reading this example program.
*/

#ifndef DLIB_DNn_SEMANTIC_SEGMENTATION_EX_H_
#define DLIB_DNn_SEMANTIC_SEGMENTATION_EX_H_

#include <dlib/dnn.h>

// ----------------------------------------------------------------------------------------

static const char* semantic_segmentation_net_filename = "semantic_segmentation_voc2012net_v2.dnn";

// ----------------------------------------------------------------------------------------

inline bool operator == (const dlib::rgb_pixel& a, const dlib::rgb_pixel& b)
{
    return a.red == b.red && a.green == b.green && a.blue == b.blue;
}

// ----------------------------------------------------------------------------------------

// The PASCAL VOC2012 dataset contains 20 ground-truth classes + background.  Each class
// is represented using an RGB color value.  We associate each class also to an index in the
// range [0, 20], used internally by the network.

struct Voc2012class {
    Voc2012class(uint16_t index, const dlib::rgb_pixel& rgb_label, const std::string& classlabel)
        : index(index), rgb_label(rgb_label), classlabel(classlabel)
    {}

    // The index of the class. In the PASCAL VOC 2012 dataset, indexes from 0 to 20 are valid.
    const uint16_t index = 0;

    // The corresponding RGB representation of the class.
    const dlib::rgb_pixel rgb_label;

    // The label of the class in plain text.
    const std::string classlabel;
};

namespace {
    constexpr int class_count = 21; // background + 20 classes

    const std::vector<Voc2012class> classes = {
        Voc2012class(0, dlib::rgb_pixel(0, 0, 0), ""), // background

        // The cream-colored `void' label is used in border regions and to mask difficult objects
        // (see http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html)
        Voc2012class(dlib::loss_multiclass_log_per_pixel_::label_to_ignore,
            dlib::rgb_pixel(224, 224, 192), "border"),

        Voc2012class(1,  dlib::rgb_pixel(128,   0,   0), "aeroplane"),
        Voc2012class(2,  dlib::rgb_pixel(  0, 128,   0), "bicycle"),
        Voc2012class(3,  dlib::rgb_pixel(128, 128,   0), "bird"),
        Voc2012class(4,  dlib::rgb_pixel(  0,   0, 128), "boat"),
        Voc2012class(5,  dlib::rgb_pixel(128,   0, 128), "bottle"),
        Voc2012class(6,  dlib::rgb_pixel(  0, 128, 128), "bus"),
        Voc2012class(7,  dlib::rgb_pixel(128, 128, 128), "car"),
        Voc2012class(8,  dlib::rgb_pixel( 64,   0,   0), "cat"),
        Voc2012class(9,  dlib::rgb_pixel(192,   0,   0), "chair"),
        Voc2012class(10, dlib::rgb_pixel( 64, 128,   0), "cow"),
        Voc2012class(11, dlib::rgb_pixel(192, 128,   0), "diningtable"),
        Voc2012class(12, dlib::rgb_pixel( 64,   0, 128), "dog"),
        Voc2012class(13, dlib::rgb_pixel(192,   0, 128), "horse"),
        Voc2012class(14, dlib::rgb_pixel( 64, 128, 128), "motorbike"),
        Voc2012class(15, dlib::rgb_pixel(192, 128, 128), "person"),
        Voc2012class(16, dlib::rgb_pixel(  0,  64,   0), "pottedplant"),
        Voc2012class(17, dlib::rgb_pixel(128,  64,   0), "sheep"),
        Voc2012class(18, dlib::rgb_pixel(  0, 192,   0), "sofa"),
        Voc2012class(19, dlib::rgb_pixel(128, 192,   0), "train"),
        Voc2012class(20, dlib::rgb_pixel(  0,  64, 128), "tvmonitor"),
    };
}

template <typename Predicate>
const Voc2012class& find_voc2012_class(Predicate predicate)
{
    const auto i = std::find_if(classes.begin(), classes.end(), predicate);

    if (i != classes.end())
    {
        return *i;
    }
    else
    {
        throw std::runtime_error("Unable to find a matching VOC2012 class");
    }
}

// ----------------------------------------------------------------------------------------

// Introduce the building blocks used to define the segmentation network.
// The network first does downsampling, and then upsampling. In addition, U-Net style
// skip connections are used, so that not every simple detail needs to reprented on
// the low levels. (See Ronneberger et al. (2015), U-Net: Convolutional Networks for
// Biomedical Image Segmentation, https://arxiv.org/pdf/1505.04597.pdf)

template <int num_filters, template <typename> class BN, typename SUBNET>
using con3 = dlib::relu<BN<dlib::con<num_filters,3,3,1,1,SUBNET>>>;

template <int num_filters, template <typename> class BN, typename SUBNET>
using level = con3<num_filters,BN,con3<num_filters,BN,SUBNET>>;

template <typename SUBNET>
using down = dlib::max_pool<3,3,2,2,SUBNET>;

template <int num_filters, typename SUBNET>
using up = dlib::cont<num_filters,3,3,2,2,SUBNET>;

// ----------------------------------------------------------------------------------------

#if 0
template <int num_filters, typename SUBNET>
using adown = down<num_filters,dlib::affine,SUBNET>;

template <int num_filters, typename SUBNET>
using bdown = down<num_filters,dlib::bn_con,SUBNET>;

template <int num_filters, typename SUBNET>
using aup = up<num_filters,dlib::affine,SUBNET>;

template <int num_filters, typename SUBNET>
using bup = up<num_filters,dlib::bn_con,SUBNET>;
#endif

// ----------------------------------------------------------------------------------------

template <
    template<typename> class TAG1,
    template<typename> class TAG2,
    typename SUBNET
>
using resize_and_concat = dlib::add_layer<
                          dlib::concat_<TAG1,TAG2>,
                          TAG2<dlib::resize_to_prev<TAG1,SUBNET>>>;

template <typename SUBNET> using utag0 = dlib::add_tag_layer<2100+0,SUBNET>;
template <typename SUBNET> using utag1 = dlib::add_tag_layer<2100+1,SUBNET>;
template <typename SUBNET> using utag2 = dlib::add_tag_layer<2100+2,SUBNET>;
template <typename SUBNET> using utag3 = dlib::add_tag_layer<2100+3,SUBNET>;
template <typename SUBNET> using utag4 = dlib::add_tag_layer<2100+4,SUBNET>;
template <typename SUBNET> using utag5 = dlib::add_tag_layer<2100+5,SUBNET>;
template <typename SUBNET> using utag6 = dlib::add_tag_layer<2100+6, SUBNET>;

template <typename SUBNET> using utag0_ = dlib::add_tag_layer<2110+0,SUBNET>;
template <typename SUBNET> using utag1_ = dlib::add_tag_layer<2110+1,SUBNET>;
template <typename SUBNET> using utag2_ = dlib::add_tag_layer<2110+2,SUBNET>;
template <typename SUBNET> using utag3_ = dlib::add_tag_layer<2110+3,SUBNET>;
template <typename SUBNET> using utag4_ = dlib::add_tag_layer<2110+4,SUBNET>;
template <typename SUBNET> using utag5_ = dlib::add_tag_layer<2110+5,SUBNET>;
template <typename SUBNET> using utag6_ = dlib::add_tag_layer<2110+6,SUBNET>;

template <typename SUBNET> using concat_utag0 = resize_and_concat<utag0,utag0_,SUBNET>;
template <typename SUBNET> using concat_utag1 = resize_and_concat<utag1,utag1_,SUBNET>;
template <typename SUBNET> using concat_utag2 = resize_and_concat<utag2,utag2_,SUBNET>;
template <typename SUBNET> using concat_utag3 = resize_and_concat<utag3,utag3_,SUBNET>;
template <typename SUBNET> using concat_utag4 = resize_and_concat<utag4,utag4_,SUBNET>;
template <typename SUBNET> using concat_utag5 = resize_and_concat<utag5,utag5_,SUBNET>;
template <typename SUBNET> using concat_utag6 = resize_and_concat<utag6,utag6_,SUBNET>;

// ----------------------------------------------------------------------------------------

#if 1
template <template <typename> class BN>
using net_type = dlib::loss_multiclass_log_per_pixel<
                              dlib::con<class_count,1,1,1,1,
                              level<64,BN,concat_utag0<up<64,
                              level<96,BN,concat_utag1<up<96,
                              level<128,BN,concat_utag2<up<128,
                              level<256,BN,concat_utag3<up<256,
                              level<384,BN,concat_utag4<up<384,
                              level<512,BN,concat_utag5<up<512,
                              level<768,BN,concat_utag6<up<768,
                              level<1024,BN,
                              down<utag6<level<768,BN,
                              down<utag5<level<512,BN,
                              down<utag4<level<384,BN,
                              down<utag3<level<256,BN,
                              down<utag2<level<128,BN,
                              down<utag1<level<96,BN,
                              down<utag0<level<64,BN,
                              dlib::input<dlib::matrix<dlib::rgb_pixel>>
                              >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using bnet_type = net_type<dlib::bn_con>; // training network type
using anet_type = net_type<dlib::affine>; // inference network type
#endif

#if 0
using bnet_type = dlib::loss_multiclass_log_per_pixel<
                              dlib::cont<class_count,1,1,1,1,
                              dlib::relu<dlib::bn_con<dlib::con<32,3,3,1,1,
                              concat_utag0<bup<64,7,
                              concat_utag1<bup<64,3,
                              concat_utag2<bup<128,3,
                              concat_utag3<bup<256,3,
                              concat_utag4<bup<512,3,
                              bdown<512,3,utag4<
                              bdown<256,3,utag3<
                              bdown<128,3,utag2<
                              bdown<64,3,utag1<
                              bdown<64,7,utag0<
                              dlib::relu<dlib::bn_con<dlib::con<16,3,3,1,1,
                              dlib::input<dlib::matrix<dlib::rgb_pixel>>
                              >>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using anet_type = dlib::loss_multiclass_log_per_pixel<
                              dlib::cont<class_count,1,1,1,1,
                              dlib::relu<dlib::affine<dlib::con<32,3,3,1,1,
                              concat_utag0<aup<64,7,
                              concat_utag1<aup<64,3,
                              concat_utag2<aup<128,3,
                              concat_utag3<aup<256,3,
                              concat_utag4<aup<512,3,
                              adown<512,3,utag4<
                              adown<256,3,utag3<
                              adown<128,3,utag2<
                              adown<64,3,utag1<
                              adown<64,7,utag0<
                              dlib::relu<dlib::affine<dlib::con<16,3,3,1,1,
                              dlib::input<dlib::matrix<dlib::rgb_pixel>>
                              >>>>>>>>>>>>>>>>>>>>>>>>>>>>;
#endif

// ----------------------------------------------------------------------------------------

#endif // DLIB_DNn_SEMANTIC_SEGMENTATION_EX_H_
