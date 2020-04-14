// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  I'm assuming you have already read dnn_introduction_ex.cpp, and
    dnn_dcgan_train_ex.cpp.  In this example program we are going to show how one can use
    a GAN-like setting for semi-supervised learning, i.e. train a regular classifier but
    take additional benefit from unlabeled data.

    In this example, we are going to learn to classify digits from the MNIST dataset,
    using only a small part of the labels.
*/

#include <algorithm>
#include <iostream>

#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/matrix.h>

using namespace std;
using namespace dlib;

// --- Begin: code copied from dnn_dcgan_train_ex.cpp

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
const size_t noise_size = 100;
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

// A transposed convolution to with custom padding
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using contp = add_layer<cont_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// --- End: code copied from dnn_dcgan_train_ex.cpp

// Even the generator looks the same as in dnn_dcgan_train_ex.cpp.
using generator_type =
    loss_binary_log_per_pixel<
    sig<contp<1, 4, 2, 1,
    relu<bn_con<contp<64, 4, 2, 1,
    relu<bn_con<contp<128, 3, 2, 1,
    relu<bn_con<contp<256, 4, 1, 0,
    input<noise_t>
    >>>>>>>>>>>>;

// The discriminator, however, tries to do two things at the same time:
// 1) Classify the image to any of the digit classes (0-9)
// 2) Decide if the image is "real" or "fake"
//    ("real": from the training set; "fake": dreamed of by the generator)
using discriminator_type =
    loss_multimulticlass_log<
    conp<12, 3, 1, 0, // 12 = 10 for the digit classification task + 2 for the real/fake task
    leaky_relu<bn_con<conp<256, 4, 2, 1,
    leaky_relu<bn_con<conp<128, 4, 2, 1,
    leaky_relu<conp<64, 4, 2, 1,
    input<matrix<unsigned char>>
    >>>>>>>>>>;

// --- Begin: more code copied from dnn_dcgan_train_ex.cpp

// Some helper functions to generate and get the images from the generator
matrix<unsigned char> generate_image(generator_type& net, const noise_t& noise)
{
    const matrix<float> output = net(noise);
    matrix<unsigned char> image;
    assign_image(image, 255 * output);
    return image;
}

std::vector<matrix<unsigned char>> get_generated_images(const tensor& out)
{
    std::vector<matrix<unsigned char>> images;
    for (long n = 0; n < out.num_samples(); ++n)
    {
        matrix<float> output = image_plane(out, n);
        matrix<unsigned char> image;
        assign_image(image, 255 * output);
        images.push_back(std::move(image));
    }
    return images;
}

// --- End: more code copied from dnn_dcgan_train_ex.cpp

constexpr auto unknown_label = std::numeric_limits<unsigned long>::max();

// Intentionally forget most of the training labels (to simulate semi-supervised learning).
// Returns the number of labels not forgotten.
size_t decimate_training_labels(std::vector<unsigned long>& training_labels, dlib::rand& rnd)
{
    // Keep only about 0.1% of the labels
    const double prob_keep_training_sample_label = 0.001;

    size_t keep_count = 0;

    for (auto& training_label : training_labels)
    {
        const bool keep_label = rnd.get_double_in_range(0.0, 1.0) < prob_keep_training_sample_label;
        if (keep_label)
            ++keep_count;
        else
            training_label = unknown_label;
    }

    return keep_count;
}

int main(int argc, char** argv) try
{
    // This example is going to run on the MNIST dataset.
    if (argc != 2)
    {
        cout << "This example needs the MNIST dataset to run!" << endl;
        cout << "You can get MNIST from http://yann.lecun.com/exdb/mnist/" << endl;
        cout << "Download the 4 files that comprise the dataset, decompress them, and" << endl;
        cout << "put them in a folder.  Then give that folder as input to this program." << endl;
        return EXIT_FAILURE;
    }

    // MNIST is broken into two parts, a training set of 60000 images and a test set of 10000
    // images.  Each image is labeled so that we know what hand written digit is depicted.
    // These next statements load the dataset into memory.
    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long>         training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long>         testing_labels;
    load_mnist_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);

    // Fix the random generator seeds for network initialization and noise
    srand(1234);
    dlib::rand rnd(std::rand());

    // Actually use only part of the training data available
    const auto training_label_count = decimate_training_labels(training_labels, rnd);

    // Define the universe of possible labels that may be passed to the discriminator
    const std::map<string, std::vector<string>> possible_labels {
        { "Supervised", { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" } },
        { "Unsupervised", { "Real", "Fake" } }
    };

    // Instantiate both generator and discriminator
    generator_type generator;
    discriminator_type discriminator(
        possible_labels,
        leaky_relu_(0.2), leaky_relu_(0.2), leaky_relu_(0.2)
    );
    // Remove the bias learning from the networks
    visit_layers(generator, visitor_no_bias());
    visit_layers(discriminator, visitor_no_bias());

    cout << endl << "Using " << training_label_count << " training labels" << endl;

    // The solvers for the generator and discriminator networks.
    // Copied from dnn_dcgan_train_ex.cpp.
    std::vector<adam> g_solvers(generator.num_computational_layers, adam(0, 0.5, 0.999));
    std::vector<adam> d_solvers(discriminator.num_computational_layers, adam(0, 0.5, 0.999));
    double learning_rate = 2e-4;

    // Resume training from last sync file, if any.
    size_t iteration = 0;
    if (file_exists("semisupervised_sync"))
        deserialize("semisupervised_sync") >> generator >> discriminator >> iteration;

    const size_t minibatch_size = 64;

    // All generated images have the same target labels: unknown (empty) digit class, and fake image
    const std::vector<std::map<std::string, std::string>> fake_labels(
        minibatch_size, { { "Supervised", "" }, { "Unsupervised", "Fake" } }
    );

    dlib::image_window win;
    resizable_tensor real_samples_tensor, fake_samples_tensor, noises_tensor;
    running_stats<double> g_loss, d_loss;
    while (iteration < 50000)
    {
        // Train the discriminator with real images. This is the same as in dnn_dcgan_train_ex.cpp,
        // except now we also set the classes (where available).
        std::vector<matrix<unsigned char>> real_samples;
        std::vector<std::map<std::string, std::string>> real_labels;

        while (real_samples.size() < minibatch_size)
        {
            auto idx = rnd.get_random_32bit_number() % training_images.size();
            real_samples.push_back(training_images[idx]);

            // For real images, supply the digit class where available; also specify that the image is "real"
            const auto label = training_labels[idx] == unknown_label ? "" : std::to_string(training_labels[idx]);
            real_labels.push_back({ { "Supervised", label }, { "Unsupervised", "Real" } });
        }
        discriminator.to_tensor(real_samples.begin(), real_samples.end(), real_samples_tensor);
        d_loss.add(discriminator.compute_loss(real_samples_tensor, real_labels.begin()));
        discriminator.back_propagate_error(real_samples_tensor);
        discriminator.update_parameters(d_solvers, learning_rate);

        // Train the discriminator with fake images. With fake_labels already initialized, the code itself
        // does not differ from dnn_dcgan_train_ex.cpp.
        std::vector<noise_t> noises;
        while (noises.size() < minibatch_size)
        {
            noises.push_back(make_noise(rnd));
        }
        generator.to_tensor(noises.begin(), noises.end(), noises_tensor);
        const auto fake_samples = get_generated_images(generator.forward(noises_tensor));
        discriminator.to_tensor(fake_samples.begin(), fake_samples.end(), fake_samples_tensor);
        d_loss.add(discriminator.compute_loss(fake_samples_tensor, fake_labels.begin()));
        discriminator.back_propagate_error(fake_samples_tensor);
        discriminator.update_parameters(d_solvers, learning_rate);

        // Train the generator. This does not differ from dnn_dcgan_train_ex.cpp either.
        g_loss.add(discriminator.compute_loss(fake_samples_tensor, real_labels.begin()));
        discriminator.back_propagate_error(fake_samples_tensor);
        const tensor& d_grad = discriminator.get_final_data_gradient();
        generator.back_propagate_error(noises_tensor, d_grad);
        generator.update_parameters(g_solvers, learning_rate);

        // The generated images should very soon start looking like samples from the
        // MNIST dataset.
        if (++iteration % 1000 == 0)
        {
            serialize("semisupervised_sync") << generator << discriminator << iteration;
            std::cout <<
                "step#: " << iteration <<
                "\tdiscriminator loss: " << d_loss.mean() * 2 <<
                "\tgenerator loss: " << g_loss.mean() << '\n';
            win.set_image(tile_images(fake_samples));
            win.set_title("Semisupervised training step#: " + to_string(iteration));
            d_loss.clear();
            g_loss.clear();
        }
    }

    // Once the training has finished, we don't need the generator any more.  We just keep the
    // discriminator.  (Because here in this example our objective is to classify images, and
    // not generate new images as in dnn_dcgan_train_ex.cpp.)
    discriminator.clean();
    serialize("semisupervised_mnist.dnn") << discriminator;

    // (The discriminator could be further simplified by dropping the real/fake classifier.)

    // Now let's classify the test set.  Remember that we trained with only about 0.1% of the
    // labels!
    const auto predicted_labels = discriminator(testing_images);
    size_t num_right = 0;
    size_t num_wrong = 0;
    for (size_t i = 0; i < testing_images.size(); ++i)
    {
        const std::string predicted_label = predicted_labels[i].find("Supervised")->second;
        if (std::stoul(predicted_label) == testing_labels[i])
            ++num_right;
        else
            ++num_wrong;

    }
    cout << "testing num_right: " << num_right << endl;
    cout << "testing num_wrong: " << num_wrong << endl;
    cout << "testing accuracy:  " << num_right / (double)(num_right + num_wrong) << endl;

    return EXIT_SUCCESS;
}
catch(exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
