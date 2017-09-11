// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This example shows how to run a CNN based vehicle detector using dlib.  The
    example loads a pretrained model and uses it to find the rear ends of cars in
    an image.  We will also visualize some of the detector's processing steps by
    plotting various intermediate images on the screen.  Viewing these can help
    you understand how the detector works.
    
    The model used by this example was trained by the dnn_mmod_train_find_cars_ex.cpp 
    example.  Also, since this is a CNN, you really should use a GPU to get the
    best execution speed.  For instance, when run on a NVIDIA 1080ti, this
    detector runs at 39fps when run on the provided test image.  That's about an 
    order of magnitude faster than when run on the CPU.

    Users who are just learning about dlib's deep learning API should read
    the dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp examples to learn
    how the API works.  For an introduction to the object detection method you
    should read dnn_mmod_ex.cpp.

    You can also see some videos of this vehicle detector running on YouTube:
        https://www.youtube.com/watch?v=4B3bzmxMAZU
        https://www.youtube.com/watch?v=bP2SUo5vSlc
*/


#include <iostream>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>

using namespace std;
using namespace dlib;



// The rear view vehicle detector network
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<55,SUBNET>>>;
using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image>>>>>>;

// ----------------------------------------------------------------------------------------

int main() try
{
    net_type net;
    shape_predictor sp;
    // You can get this file from http://dlib.net/files/mmod_rear_end_vehicle_detector.dat.bz2
    // This network was produced by the dnn_mmod_train_find_cars_ex.cpp example program.
    // As you can see, it also includes a shape_predictor.  To see a generic example of how
    // to train those refer to train_shape_predictor_ex.cpp.
    deserialize("mmod_rear_end_vehicle_detector.dat") >> net >> sp;

    matrix<rgb_pixel> img;
    load_image(img, "../mmod_cars_test_image.jpg");

    image_window win;
    win.set_image(img);

    // Run the detector on the image and show us the output.
    for (auto&& d : net(img))
    {
        // We use a shape_predictor to refine the exact shape and location of the detection
        // box.  This shape_predictor is trained to simply output the 4 corner points of
        // the box.  So all we do is make a rectangle that tightly contains those 4 points
        // and that rectangle is our refined detection position.
        auto fd = sp(img,d);
        rectangle rect;
        for (unsigned long j = 0; j < fd.num_parts(); ++j)
            rect += fd.part(j);
        win.add_overlay(rect, rgb_pixel(255,0,0));
    }



    cout << "Hit enter to view the intermediate processing steps" << endl;
    cin.get();


    // Now let's look at how the detector works.  The high level processing steps look like:
    //   1. Create an image pyramid and pack the pyramid into one big image.  We call this
    //      image the "tiled pyramid".
    //   2. Run the tiled pyramid image through the CNN.  The CNN outputs a new image where
    //      bright pixels in the output image indicate the presence of cars.  
    //   3. Find pixels in the CNN's output image with a value > 0.  Those locations are your
    //      preliminary car detections.  
    //   4. Perform non-maximum suppression on the preliminary detections to produce the
    //      final output.
    //
    // We will be plotting the images from steps 1 and 2 so you can visualize what's
    // happening.  For the CNN's output image, we will use the jet colormap so that "bright"
    // outputs, i.e. pixels with big values, appear in red and "dim" outputs appear as a
    // cold blue color.  To do this we pick a range of CNN output values for the color
    // mapping.  The specific values don't matter.  They are just selected to give a nice
    // looking output image.
    const float lower = -2.5;
    const float upper = 0.0;
    cout << "jet color mapping range:  lower="<< lower << "  upper="<< upper << endl;



    // This CNN detector represents a sliding window detector with 3 sliding windows.  Each
    // of the 3 windows has a different aspect ratio, allowing it to find vehicles which
    // are either tall and skinny, squarish, or short and wide.  The aspect ratio of a
    // detection is determined by which channel in the output image triggers the detection.
    // Here we are just going to max pool the channels together to get one final image for
    // our display.  In this image, a pixel will be bright if any of the sliding window
    // detectors thinks there is a car at that location.
    cout << "Number of channels in final tensor image: " << net.subnet().get_output().k() << endl;
    matrix<float> network_output = image_plane(net.subnet().get_output(),0,0);
    for (long k = 1; k < net.subnet().get_output().k(); ++k)
        network_output = max_pointwise(network_output, image_plane(net.subnet().get_output(),0,k));
    // We will also upsample the CNN's output image.  The CNN we defined has an 8x
    // downsampling layer at the beginning. In the code below we are going to overlay this
    // CNN output image on top of the raw input image.  To make that look nice it helps to
    // upsample the CNN output image back to the same resolution as the input image, which
    // we do here.
    const double network_output_scale = img.nc()/(double)network_output.nc();
    resize_image(network_output_scale, network_output);


    // Display the network's output as a color image.   
    image_window win_output(jet(network_output, upper, lower), "Output tensor from the network");



    cout << "Hit enter to end program" << endl;
    cin.get();
}
catch(image_load_error& e)
{
    cout << e.what() << endl;
    cout << "The test image is located in the examples folder.  So you should run this program from a sub folder so that the relative path is correct." << endl;
}
catch(serialization_error& e)
{
    cout << e.what() << endl;
    cout << "The model file can be obtained from: http://dlib.net/files/mmod_rear_end_vehicle_detector.dat.bz2   Don't forget to unzip the file." << endl;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




