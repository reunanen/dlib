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

// This class is simply copied from dnn_dcgan_train_ex.cpp
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

const size_t noise_size = 100;
using noise_t = std::array<matrix<float, 1, 1>, noise_size>;

// This, too, is copied from dnn_dcgan_train_ex.cpp
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

// Even the generator looks the same as in dnn_dcgan_train_ex.cpp
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

// Also the following two functions are copied from dnn_dcgan_train_ex.cpp
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

constexpr auto unknown_label = std::numeric_limits<unsigned long>::max();

// Intentionally forget most of the training labels, to simulate semi-supervised learning
void decimate_training_labels(std::vector<unsigned long>& training_labels)
{
    // Keep only 10 samples per class
    const size_t samples_to_keep_per_class = 10;

    std::map<unsigned long, std::deque<size_t>> indexes_by_class;

    for (size_t i = 0; i < training_labels.size(); ++i)
        indexes_by_class[training_labels[i]].push_back(i);

    for (auto& i : indexes_by_class)
    {
        auto& indexes = i.second;
        std::random_shuffle(indexes.begin(), indexes.end());

        // Forget the label of all samples except the first N (after shuffling)
        for (size_t j = samples_to_keep_per_class; j < indexes.size(); ++j)
            training_labels[indexes[j]] = unknown_label;
    }
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
    decimate_training_labels(training_labels);

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

    std::deque<size_t> training_label_indexes;

    for (size_t i = 0; i < training_labels.size(); ++i)
        if (training_labels[i] != unknown_label)
            training_label_indexes.push_back(i);

    cout << endl << "Using " << training_label_indexes.size() << " training labels" << endl;

    // The solvers for the generator and discriminator networks.
    // Copied from dnn_dcgan_train_ex.cpp.
    std::vector<adam> g_solvers(generator.num_computational_layers, adam(0, 0.5, 0.999));
    std::vector<adam> d_solvers(discriminator.num_computational_layers, adam(0, 0.5, 0.999));

    const double g_learning_rate = 2e-4;
    const double d_learning_rate = 1e-4;

    // Resume training from last sync file, if any.
    size_t iteration = 0;
    if (file_exists("semisupervised_sync"))
        deserialize("semisupervised_sync") >> generator >> discriminator >> iteration;

    const size_t minibatch_size = 64;
    const size_t max_iter = 50000;

    // All generated images have the same target labels: unknown (empty) digit class, and fake image
    const std::vector<std::map<std::string, std::string>> fake_labels(
        minibatch_size, { { "Supervised", "" }, { "Unsupervised", "Fake" } }
    );

    dlib::image_window win;
    resizable_tensor real_samples_tensor, fake_samples_tensor, noises_tensor;
    running_stats<double> g_loss, d_loss;
    while (iteration < max_iter)
    {
        // Train the discriminator with real images.  This is the same as in dnn_dcgan_train_ex.cpp,
        // except in that now we also set the classes (where available).  In addition, we control the
        // proportion of labeled samples.
        std::vector<matrix<unsigned char>> real_samples;
        std::vector<std::map<std::string, std::string>> real_labels;

        const auto proportion_of_labeled_samples = iteration / static_cast<double>(max_iter);
        const auto w = 1.0 - proportion_of_labeled_samples;

        while (real_samples.size() < minibatch_size)
        {
            const bool require_labeled = false; // rnd.get_double_in_range(0, 1) < proportion_of_labeled_samples;
            const auto random_number = rnd.get_random_64bit_number();

            const auto index = require_labeled
                ? training_label_indexes[random_number % training_label_indexes.size()]
                : random_number % training_images.size();

            real_samples.push_back(training_images[index]);

            // For real images, supply the digit class where available; also specify that the image is "real"
            const auto label = training_labels[index] == unknown_label ? "" : std::to_string(training_labels[index]);
            real_labels.push_back({ { "Supervised", label }, { "Unsupervised", "Real" } });
        }
        discriminator.to_tensor(real_samples.begin(), real_samples.end(), real_samples_tensor);
        d_loss.add(discriminator.compute_loss(real_samples_tensor, real_labels.begin()));
        discriminator.back_propagate_error(real_samples_tensor);
        discriminator.update_parameters(d_solvers, w * d_learning_rate);

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
        discriminator.update_parameters(d_solvers, w * d_learning_rate);

        // Let's not use the supervised labels for training the generator.
        for (auto& real_label : real_labels)
            real_label["Supervised"] = "";

        // Train the generator. This does not differ from dnn_dcgan_train_ex.cpp either.
        g_loss.add(discriminator.compute_loss(fake_samples_tensor, real_labels.begin()));
        discriminator.back_propagate_error(fake_samples_tensor);
        const tensor& d_grad = discriminator.get_final_data_gradient();
        generator.back_propagate_error(noises_tensor, d_grad);
        generator.update_parameters(g_solvers, w * g_learning_rate);

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

    // Now let's classify the test set.  Remember that we trained with 100 labels only!
    // Generally an error of about 1% (100 samples) is expected. (See Table 1 of Salimans et
    // al., 2016, Improved Techniques for Training GANs, https://arxiv.org/pdf/1606.03498.pdf)
    size_t num_right = 0;
    size_t num_wrong = 0;
    for (size_t i = 0; i < testing_images.size(); ++i)
    {
        // Feed data one by one, because the batch normalization layer supports this kind of
        // a test mode (alternatively, a new net with affine layers could be instantiated).
        const auto& prediction = discriminator(testing_images[i]);
        const std::string& predicted_label = prediction.find("Supervised")->second;
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
