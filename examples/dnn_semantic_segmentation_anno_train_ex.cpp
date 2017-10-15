// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to train a semantic segmentation net using the data generated
    using the anno program (see https://github.com/reunanen/anno).

    Instructions:
    1. Use anno to label some data.
    2. Build the dnn_semantic_segmentation_anno_train_ex example program.
    3. Run:
       ./dnn_semantic_segmentation_anno_train_ex /path/to/anno/data
    4. Wait while the network is being trained.
    5. Build the dnn_semantic_segmentation_anno_ex example program.
    6. Run:
       ./dnn_semantic_segmentation_anno_ex /path/to/anno/data
*/

#include "dnn_semantic_segmentation_anno_ex.h"

#include <iostream>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>
#include <iterator>
#include <thread>

using namespace std;
using namespace dlib;

typedef std::pair<matrix<rgb_pixel>, matrix<uint16_t>> training_sample;

// ----------------------------------------------------------------------------------------

rectangle make_random_cropping_rect(
    int dim,
    const matrix<rgb_pixel>& img,
    dlib::rand& rnd
)
{
    DLIB_CASSERT(img.nc() > dim && img.nr() > dim);
    rectangle rect(dim, dim);
    // randomly shift the box around
    point offset(rnd.get_random_32bit_number()%(img.nc()-rect.width()),
                 rnd.get_random_32bit_number()%(img.nr()-rect.height()));
    return move_rect(rect, offset);
}

rectangle make_cropping_rect_around_defect(
    int dim,
    const matrix<rgb_pixel>& img,
    point defect_point,
    dlib::rand& rnd
)
{
    rectangle rect(dim, dim);

    long min_x = std::max(defect_point.x() - dim / 2, 0L);
    long max_x = std::min(min_x + dim, img.nc());
    long min_y = std::max(defect_point.y() - dim / 2, 0L);
    long max_y = std::min(min_y + dim, img.nr());

    DLIB_CASSERT(max_x > min_x);
    DLIB_CASSERT(max_y > min_y);

    point offset(min_x + rnd.get_random_32bit_number() % (max_x - min_x),
                 min_y + rnd.get_random_32bit_number() % (max_y - min_y));

    return move_rect(rect, offset);
}

// ----------------------------------------------------------------------------------------

void randomly_crop_image (
    const matrix<rgb_pixel>& input_image,
    const matrix<uint16_t>& label_image,
    training_sample& crop,
    dlib::rand& rnd
)
{
    const int dim = 227;

    const bool crop_around_defect = rnd.get_random_32bit_number() % 2 == 0;
    
    rectangle rect;

    if (crop_around_defect) {
        std::vector<point> nonzero;
        for (long r = 0, nr = label_image.nr(); r < nr; ++r) {
            for (long c = 0, nc = label_image.nc(); c < nc; ++c) {
                const auto label = label_image(r, c);
                if (label > 0 && label != loss_multiclass_log_per_pixel_::label_to_ignore) {
                    nonzero.push_back(point(c, r));
                }
            }
        }
        if (!nonzero.empty()) {
            const point& defect_point = nonzero[rnd.get_random_64bit_number() % nonzero.size()];
            rect = make_cropping_rect_around_defect(dim, input_image, defect_point, rnd);
        }
        else {
            rect = make_random_cropping_rect(dim, input_image, rnd);
        }
    }
    else {
        rect = make_random_cropping_rect(dim, input_image, rnd);
    }

    const chip_details chip_details(rect, chip_dims(dim, dim));

    // Crop the input image.
    extract_image_chip(input_image, chip_details, crop.first, interpolate_bilinear());

    // Crop the labels correspondingly. However, note that here bilinear
    // interpolation would make absolutely no sense.
    extract_image_chip(label_image, chip_details, crop.second, interpolate_nearest_neighbor());

    // Also randomly flip the input image and the labels.
    if (rnd.get_random_double() > 0.5) {
        crop.first = flipud(crop.first);
        crop.second = flipud(crop.second);
    }

    // And then randomly adjust the colors.
    //apply_random_color_offset(crop.first, rnd);
}

// ----------------------------------------------------------------------------------------

struct image_info
{
    string image_filename;
    string label_filename;
};

std::vector<image_info> get_anno_data_listing(
    const std::string& anno_data_folder
)
{
    const std::vector<file> files = get_files_in_directory_tree(anno_data_folder,
        [](const file& name) {
            if (match_ending("_mask.png")(name)) {
                return false;
            }
            if (match_ending("_result.png")(name)) {
                return false;
            }
            return match_ending(".jpg")(name)
                || match_ending(".png")(name);
        });

    std::vector<image_info> results;

    for (const file& name : files) {
        image_info image_info;
        image_info.image_filename = name;
        image_info.label_filename = name.full_name() + "_mask.png";
        std::ifstream label_file(image_info.label_filename, std::ios::binary);
        if (label_file) {
            results.push_back(image_info);
            std::cout << "Added file " << image_info.image_filename << std::endl;
        }
        else {
            std::cout << "Warning: unable to open " << image_info.label_filename << std::endl;
        }
    }

    return results;
}

// ----------------------------------------------------------------------------------------

const AnnoClass& find_anno_class(const dlib::rgb_pixel& rgb_label)
{
    return find_anno_class(
        [&rgb_label](const AnnoClass& anno_class) {
            return rgb_label == anno_class.rgb_label;
        }
    );
}

inline uint16_t rgb_label_to_index_label(const dlib::rgb_pixel& rgb_label)
{
    return find_anno_class(rgb_label).index;
}

void rgb_label_image_to_index_label_image(const dlib::matrix<dlib::rgb_pixel>& rgb_label_image, dlib::matrix<uint16_t>& index_label_image)
{
    const long nr = rgb_label_image.nr();
    const long nc = rgb_label_image.nc();

    index_label_image.set_size(nr, nc);

    for (long r = 0; r < nr; ++r) {
        for (long c = 0; c < nc; ++c) {
            index_label_image(r, c) = rgb_label_to_index_label(rgb_label_image(r, c));
        }
    }
}

// ----------------------------------------------------------------------------------------

double calculate_accuracy(anet_type& anet, const std::vector<image_info>& dataset)
{
    int num_right = 0;
    int num_wrong = 0;

    matrix<rgb_pixel> input_image;
    matrix<rgb_pixel> rgb_label_image;
    matrix<uint16_t> index_label_image;

    for (const auto& image_info : dataset) {
        load_image(input_image, image_info.image_filename);
        load_image(rgb_label_image, image_info.label_filename);

        matrix<uint16_t> net_output = anet(input_image);

        rgb_label_image_to_index_label_image(rgb_label_image, index_label_image);

        const long nr = index_label_image.nr();
        const long nc = index_label_image.nc();

        for (long r = 0; r < nr; ++r) {
            for (long c = 0; c < nc; ++c) {
                const uint16_t truth = index_label_image(r, c);
                if (truth != dlib::loss_multiclass_log_per_pixel_::label_to_ignore) {
                    const uint16_t prediction = net_output(r, c);
                    if (prediction == truth) {
                        ++num_right;
                    }
                    else {
                        ++num_wrong;
                    }
                }
            }
        }
    }

    return num_right / static_cast<double>(num_right + num_wrong);
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        cout << "To run this program you need data annotated using the anno program." << endl;
        cout << endl;
        cout << "You call this program like this: " << endl;
        cout << "./dnn_semantic_segmentation_anno_train_ex /path/to/anno/data" << endl;
        return 1;
    }

    cout << "\nSCANNING ANNO DATASET\n" << endl;

    const auto listing = get_anno_data_listing(argv[1]);
    cout << "images in dataset: " << listing.size() << endl;
    if (listing.size() == 0)
    {
        cout << "Didn't find an anno dataset. " << endl;
        return 1;
    }

    const double initial_learning_rate = 0.1;
    const double weight_decay = 0.0001;
    const double momentum = 0.9;

    net_type net;
    dnn_trainer<net_type> trainer(net,sgd(weight_decay, momentum));
    trainer.be_verbose();
    trainer.set_learning_rate(initial_learning_rate);
    trainer.set_synchronization_file("anno_trainer_state_file.dat", std::chrono::minutes(10));
    // This threshold is probably excessively large.
    trainer.set_iterations_without_progress_threshold(20000);
    // Since the progress threshold is so large might as well set the batch normalization
    // stats window to something big too.
    set_all_bn_running_stats_window_sizes(net, 1000);

    std::vector<matrix<rgb_pixel>> samples;
    std::vector<matrix<uint16_t>> labels;

    std::vector<std::future<training_sample>> full_image_futures;
    full_image_futures.reserve(listing.size());

    const auto read_training_sample = [](const image_info& image_info)
    {
        matrix<rgb_pixel> input_image;
        matrix<rgb_pixel> rgb_label_image;
        matrix<uint16_t> index_label_image;

        load_image(input_image, image_info.image_filename);
        load_image(rgb_label_image, image_info.label_filename);
        rgb_label_image_to_index_label_image(rgb_label_image, index_label_image);
        return std::make_pair(input_image, index_label_image);
    };

    for (const image_info& image_info : listing) {
        full_image_futures.push_back(std::async(read_training_sample, image_info));
    }

    std::vector<training_sample> full_images(listing.size());

    for (size_t i = 0, end = full_image_futures.size(); i < end; ++i) {
        std::cout << "\rReading image " << (i + 1) << " of " << end << "...";
        full_images[i] = full_image_futures[i].get();
    }

    cout << endl << "Now training..." << endl;

    // Start a bunch of threads that read images from disk and pull out random crops.  It's
    // important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
    // thread for this kind of data preparation helps us do that.  Each thread puts the
    // crops into the data queue.
    dlib::pipe<training_sample> data(100);
    auto f = [&data, &full_images](time_t seed)
    {
        dlib::rand rnd(time(0)+seed);
        matrix<rgb_pixel> input_image;
        matrix<rgb_pixel> rgb_label_image;
        matrix<uint16_t> index_label_image;
        training_sample temp;
        while(data.is_enabled())
        {
            const size_t index = rnd.get_random_32bit_number() % full_images.size();
            const training_sample& training_sample = full_images[index];
            randomly_crop_image(training_sample.first, training_sample.second, temp, rnd);
            data.enqueue(temp);
        }
    };
    std::thread data_loader1([f](){ f(1); });
    std::thread data_loader2([f](){ f(2); });
    std::thread data_loader3([f](){ f(3); });
    std::thread data_loader4([f](){ f(4); });

    // The main training loop.  Keep making mini-batches and giving them to the trainer.
    // We will run until the learning rate has dropped by a factor of 1e-4.
    while(trainer.get_learning_rate() >= 1e-4)
    {
        samples.clear();
        labels.clear();

        // make a 30-image mini-batch
        training_sample temp;
        while(samples.size() < 30)
        {
            data.dequeue(temp);

            samples.push_back(std::move(temp.first));
            labels.push_back(std::move(temp.second));
        }

        trainer.train_one_step(samples, labels);
    }

    // Training done, tell threads to stop and make sure to wait for them to finish before
    // moving on.
    data.disable();
    data_loader1.join();
    data_loader2.join();
    data_loader3.join();
    data_loader4.join();

    // also wait for threaded processing to stop in the trainer.
    trainer.get_net();

    net.clean();
    cout << "saving network" << endl;
    serialize("annonet.dnn") << net;



#if 0
    anet_type anet = net;

    cout << "Testing the network..." << endl;

    cout << "train accuracy  :  " << calculate_accuracy(anet, get_anno_data_listing(argv[1])) << endl;
#endif
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

