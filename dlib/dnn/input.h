// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_INPUT_H_
#define DLIB_DNn_INPUT_H_

#include "input_abstract.h"
#include "../matrix.h"
#include "../array2d.h"
#include "../pixel.h"
#include "../image_processing.h"
#include <sstream>
#include <array>
#include "../cuda/tensor_tools.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename T>
    class input
    {
        const static bool always_false = sizeof(T)!=sizeof(T); 
        static_assert(always_false, "Unsupported type given to input<>.  input<> only supports "
            "dlib::matrix and dlib::array2d objects."); 
    };

// ----------------------------------------------------------------------------------------

    template <size_t NR, size_t NC=NR, typename mem_manager = default_memory_manager>
    class input_rgb_image_sized;

    template <typename mem_manager = default_memory_manager>
    class input_rgb_image_pair;

    template <typename mem_manager = default_memory_manager>
    class input_rgb_image
    {
    public:
        typedef matrix<rgb_pixel,0,0,mem_manager> input_type;

        input_rgb_image (
        ) : 
            avg_red(122.782f),
            avg_green(117.001f),
            avg_blue(104.298f)
        {
        }

        input_rgb_image (
            float avg_red_,
            float avg_green_,
            float avg_blue_
        ) : avg_red(avg_red_), avg_green(avg_green_), avg_blue(avg_blue_) 
        {}

        template <size_t NR, size_t NC>
        inline input_rgb_image (
            const input_rgb_image_sized<NR,NC>& item
        );

        inline input_rgb_image (
            const input_rgb_image_pair& item
        );

        float get_avg_red()   const { return avg_red; }
        float get_avg_green() const { return avg_green; }
        float get_avg_blue()  const { return avg_blue; }

        bool image_contained_point ( const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space ( const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space ( const tensor& /*data*/, double /*scale*/, drectangle r ) const { return r; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            const auto nr = ibegin->nr();
            const auto nc = ibegin->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->nr()==nr && i->nc()==nc,
                    "\t input_rgb_image::to_tensor()"
                    << "\n\t All matrices given to to_tensor() must have the same dimensions."
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t i->nr(): " << i->nr()
                    << "\n\t i->nc(): " << i->nc()
                );
            }


            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(ibegin,iend), 3, nr, nc);


            const size_t offset = nr*nc;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        rgb_pixel temp = (*i)(r,c);
                        auto p = ptr++;
                        *p = (temp.red-avg_red)/256.0; 
                        p += offset;
                        *p = (temp.green-avg_green)/256.0; 
                        p += offset;
                        *p = (temp.blue-avg_blue)/256.0; 
                        p += offset;
                    }
                }
                ptr += offset*(data.k()-1);
            }

        }

        friend void serialize(const input_rgb_image& item, std::ostream& out)
        {
            serialize("input_rgb_image", out);
            serialize(item.avg_red, out);
            serialize(item.avg_green, out);
            serialize(item.avg_blue, out);
        }

        friend void deserialize(input_rgb_image& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input_rgb_image" && version != "input_rgb_image_sized" && version != "input_rgb_image_pair")
                throw serialization_error("Unexpected version found while deserializing dlib::input_rgb_image.");
            deserialize(item.avg_red, in);
            deserialize(item.avg_green, in);
            deserialize(item.avg_blue, in);

            // read and discard the sizes if this was really a sized input layer.
            if (version == "input_rgb_image_sized")
            {
                size_t nr, nc;
                deserialize(nr, in);
                deserialize(nc, in);
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const input_rgb_image& item)
        {
            out << "input_rgb_image("<<item.avg_red<<","<<item.avg_green<<","<<item.avg_blue<<")";
            return out;
        }

        friend void to_xml(const input_rgb_image& item, std::ostream& out)
        {
            out << "<input_rgb_image r='"<<item.avg_red<<"' g='"<<item.avg_green<<"' b='"<<item.avg_blue<<"'/>\n";
        }

    private:
        float avg_red;
        float avg_green;
        float avg_blue;
    };

// ----------------------------------------------------------------------------------------

    template <size_t NR, size_t NC, typename mem_manager>
    class input_rgb_image_sized
    {
    public:
        static_assert(NR != 0 && NC != 0, "The input image can't be empty.");

        typedef matrix<rgb_pixel,0,0,mem_manager> input_type;

        input_rgb_image_sized (
        ) : 
            avg_red(122.782), 
            avg_green(117.001),
            avg_blue(104.298) 
        {
        }

        input_rgb_image_sized (
            const input_rgb_image& item
        ) : avg_red(item.get_avg_red()),
            avg_green(item.get_avg_green()),
            avg_blue(item.get_avg_blue())
        {}

        input_rgb_image_sized (
            float avg_red_,
            float avg_green_,
            float avg_blue_
        ) : avg_red(avg_red_), avg_green(avg_green_), avg_blue(avg_blue_) 
        {}

        float get_avg_red()   const { return avg_red; }
        float get_avg_green() const { return avg_green; }
        float get_avg_blue()  const { return avg_blue; }

        bool image_contained_point ( const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space ( const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space ( const tensor& /*data*/, double /*scale*/, drectangle r ) const { return r; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            // make sure all input images have the correct size
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->nr()==NR && i->nc()==NC,
                    "\t input_rgb_image_sized::to_tensor()"
                    << "\n\t All input images must have "<<NR<<" rows and "<<NC<< " columns, but we got one with "<<i->nr()<<" rows and "<<i->nc()<<" columns."
                );
            }


            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(ibegin,iend), 3, NR, NC);


            const size_t offset = NR*NC;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (size_t r = 0; r < NR; ++r)
                {
                    for (size_t c = 0; c < NC; ++c)
                    {
                        rgb_pixel temp = (*i)(r,c);
                        auto p = ptr++;
                        *p = (temp.red-avg_red)/256.0; 
                        p += offset;
                        *p = (temp.green-avg_green)/256.0; 
                        p += offset;
                        *p = (temp.blue-avg_blue)/256.0; 
                        p += offset;
                    }
                }
                ptr += offset*(data.k()-1);
            }

        }

        friend void serialize(const input_rgb_image_sized& item, std::ostream& out)
        {
            serialize("input_rgb_image_sized", out);
            serialize(item.avg_red, out);
            serialize(item.avg_green, out);
            serialize(item.avg_blue, out);
            serialize(NR, out);
            serialize(NC, out);
        }

        friend void deserialize(input_rgb_image_sized& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input_rgb_image_sized")
                throw serialization_error("Unexpected version found while deserializing dlib::input_rgb_image_sized.");
            deserialize(item.avg_red, in);
            deserialize(item.avg_green, in);
            deserialize(item.avg_blue, in);
            size_t nr, nc;
            deserialize(nr, in);
            deserialize(nc, in);
            if (nr != NR || nc != NC)
            {
                std::ostringstream sout;
                sout << "Wrong image dimensions found while deserializing dlib::input_rgb_image_sized.\n";
                sout << "Expected "<<NR<<" rows and "<<NC<< " columns, but found "<<nr<<" rows and "<<nc<<" columns.";
                throw serialization_error(sout.str());
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const input_rgb_image_sized& item)
        {
            out << "input_rgb_image_sized("<<item.avg_red<<","<<item.avg_green<<","<<item.avg_blue<<") nr="<<NR<<" nc="<<NC;
            return out;
        }

        friend void to_xml(const input_rgb_image_sized& item, std::ostream& out)
        {
            out << "<input_rgb_image_sized r='"<<item.avg_red<<"' g='"<<item.avg_green<<"' b='"<<item.avg_blue<<"' nr='"<<NR<<"' nc='"<<NC<<"'/>\n";
        }

    private:
        float avg_red;
        float avg_green;
        float avg_blue;
    };

// ----------------------------------------------------------------------------------------
#if 0
    template <size_t NR, size_t NC, typename mem_manager>
    input_rgb_image<mem_manager>::
    input_rgb_image (
        const input_rgb_image_sized<NR,NC,mem_manager>& item
    ) : avg_red(item.get_avg_red()),
        avg_green(item.get_avg_green()),
        avg_blue(item.get_avg_blue())
    {}
#endif
// ----------------------------------------------------------------------------------------

    template <typename mem_manager = default_memory_manager>
    class input_grayscale_image
    {
    public:
        typedef matrix<uint8_t,0,0,mem_manager> input_type;

        input_grayscale_image(
        ) : average(127.5f)
        {}

        input_grayscale_image (
            float average_
        ) : average(average_)
        {}

        float get_average()   const { return average; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            const auto nr = ibegin->nr();
            const auto nc = ibegin->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->nr()==nr && i->nc()==nc,
                    "\t input_grayscale_image::to_tensor()"
                    << "\n\t All matrices given to to_tensor() must have the same dimensions."
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t i->nr(): " << i->nr()
                    << "\n\t i->nc(): " << i->nc()
                );
            }

            
            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(ibegin,iend), 1, nr, nc);


            const size_t offset = nr*nc;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        const uint8_t temp = (*i)(r,c);
                        *ptr++ = (temp-average)/256.0;
                    }
                }
            }

        }

        friend void serialize(const input_grayscale_image& item, std::ostream& out)
        {
            serialize("input_grayscale_image", out);
            serialize(item.average, out);
        }

        friend void deserialize(input_grayscale_image& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input_grayscale_image")
                throw serialization_error("Unexpected version found while deserializing dlib::input_grayscale_image.");
            deserialize(item.average, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const input_grayscale_image& item)
        {
            out << "input_grayscale_image("<<item.average<<")";
            return out;
        }

        friend void to_xml(const input_grayscale_image& item, std::ostream& out)
        {
            out << "<input_grayscale_image r='"<<item.average<<"'/>";
        }

    private:
        float average;
    };

// ----------------------------------------------------------------------------------------

    template <typename mem_manager>
    class input_rgb_image_pair
    {
    public:
        typedef std::pair<matrix<rgb_pixel>, matrix<rgb_pixel>> input_type;

        input_rgb_image_pair (
        ) :
            avg_red(122.782),
            avg_green(117.001),
            avg_blue(104.298)
        {
        }

        input_rgb_image_pair (
            float avg_red,
            float avg_green,
            float avg_blue
        ) : avg_red(avg_red), avg_green(avg_green), avg_blue(avg_blue)
        {}

        inline input_rgb_image_pair (
            const input_rgb_image& item
        ) :
            avg_red(item.get_avg_red()),
            avg_green(item.get_avg_green()),
            avg_blue(item.get_avg_blue())
        {}

        template <size_t NR, size_t NC>
        inline input_rgb_image_pair (
            const input_rgb_image_sized<NR, NC>& item
        ) :
            avg_red(item.get_avg_red()),
            avg_green(item.get_avg_green()),
            avg_blue(item.get_avg_blue())
        {}

        float get_avg_red()   const { return avg_red; }
        float get_avg_green() const { return avg_green; }
        float get_avg_blue()  const { return avg_blue; }

        bool image_contained_point ( const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space ( const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space ( const tensor& /*data*/, double /*scale*/, drectangle r ) const { return r; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin, iend) > 0);
            const auto nr = ibegin->first.nr();
            const auto nc = ibegin->first.nc();

            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->first.nr() == nr && i->first.nc()==nc &&
                             i->second.nr() == nr && i->second.nc() == nc,
                    "\t input_rgb_image_pair::to_tensor()"
                    << "\n\t All matrices given to to_tensor() must have the same dimensions."
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t i->first.nr(): " << i->first.nr()
                    << "\n\t i->first.nc(): " << i->first.nc()
                    << "\n\t i->second.nr(): " << i->second.nr()
                    << "\n\t i->second.nc(): " << i->second.nc()
                );
            }

            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(2 * std::distance(ibegin, iend), 3, nr, nc);

            const size_t offset = nr * nc;
            const size_t offset2 = data.size() / 2;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        rgb_pixel temp_first = i->first(r, c);
                        rgb_pixel temp_second = i->second(r, c);
                        auto p = ptr++;
                        *p = (temp_first.red - avg_red) / 256.0;
                        *(p + offset2) = (temp_second.red - avg_red) / 256.0;
                        p += offset;
                        *p = (temp_first.green - avg_green) / 256.0;
                        *(p + offset2) = (temp_second.green - avg_green) / 256.0;
                        p += offset;
                        *p = (temp_first.blue - avg_blue) / 256.0;
                        *(p + offset2) = (temp_second.blue - avg_blue) / 256.0;
                        p += offset;
                    }
                }
                ptr += offset * (data.k() - 1);
            }
        }

        friend void serialize(const input_rgb_image_pair& item, std::ostream& out)
        {
            serialize("input_rgb_image_pair", out);
            serialize(item.avg_red, out);
            serialize(item.avg_green, out);
            serialize(item.avg_blue, out);
        }

        friend void deserialize(input_rgb_image_pair& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input_rgb_image_pair" && version != "input_rgb_image" && version != "input_rgb_image_sized")
                throw serialization_error("Unexpected version found while deserializing dlib::input_rgb_image_pair.");

            deserialize(item.avg_red, in);
            deserialize(item.avg_green, in);
            deserialize(item.avg_blue, in);
            // read and discard the sizes if this was really a sized input layer.
            if (version == "input_rgb_image_sized")
            {
                size_t nr, nc;
                deserialize(nr, in);
                deserialize(nc, in);
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const input_rgb_image_pair& item)
        {
            out << "input_rgb_image_pair("<< item.avg_red<<","<<item.avg_green<<","<<item.avg_blue << ")";
            return out;
        }

        friend void to_xml(const input_rgb_image_pair& item, std::ostream& out)
        {
            out << "<input_rgb_image_pair r='"<<item.avg_red<<"' g='"<<item.avg_green<<"' b='"<<item.avg_blue<<"'/>\n";
        }

    private:
        float avg_red;
        float avg_green;
        float avg_blue;
    };

// ----------------------------------------------------------------------------------------

#if 0
    input_rgb_image::
    input_rgb_image (
        const input_rgb_image_pair& item
    ) : avg_red(item.get_avg_red()),
        avg_green(item.get_avg_green()),
        avg_blue(item.get_avg_blue())
    {}
#endif

// ----------------------------------------------------------------------------------------

    template <typename T, long NR, long NC, typename MM, typename L>
    class input<matrix<T,NR,NC,MM,L>> 
    {
    public:
        typedef matrix<T,NR,NC,MM,L> input_type;

        input() {}

        template <typename mm>
        input(const input<array2d<T,mm>>&) {}

        bool image_contained_point ( const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space ( const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space ( const tensor& /*data*/, double /*scale*/, drectangle r ) const { return r; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            const auto nr = ibegin->nr();
            const auto nc = ibegin->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->nr()==nr && i->nc()==nc,
                    "\t input::to_tensor()"
                    << "\n\t All matrices given to to_tensor() must have the same dimensions."
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t i->nr(): " << i->nr()
                    << "\n\t i->nc(): " << i->nc()
                );
            }

            
            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(ibegin,iend), pixel_traits<T>::num, nr, nc);

            typedef typename pixel_traits<T>::basic_pixel_type bptype;

            const size_t offset = nr*nc;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        auto temp = pixel_to_vector<float>((*i)(r,c));
                        auto p = ptr++;
                        for (long j = 0; j < temp.size(); ++j)
                        {
                            if (is_same_type<bptype,unsigned char>::value)
                                *p = temp(j)/256.0;
                            else
                                *p = temp(j);
                            p += offset;
                        }
                    }
                }
                ptr += offset*(data.k()-1);
            }

        }

        friend void serialize(const input& /*item*/, std::ostream& out)
        {
            serialize("input<matrix>", out);
        }

        friend void deserialize(input& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input<matrix>")
                throw serialization_error("Unexpected version found while deserializing dlib::input.");
        }

        friend std::ostream& operator<<(std::ostream& out, const input& /*item*/)
        {
            out << "input<matrix>";
            return out;
        }

        friend void to_xml(const input& /*item*/, std::ostream& out)
        {
            out << "<input/>\n";
        }
    };

// ----------------------------------------------------------------------------------------

    template <typename T, long NR, long NC, typename MM, typename L, size_t K>
    class input<std::array<matrix<T,NR,NC,MM,L>,K>> 
    {
    public:
        typedef std::array<matrix<T,NR,NC,MM,L>,K> input_type;

        input() {}
        input(const input&) {}

        bool image_contained_point ( const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space ( const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space ( const tensor& /*data*/, double /*scale*/, drectangle r ) const { return r; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            DLIB_CASSERT(ibegin->size() != 0, "When using std::array<matrix> inputs you can't give 0 sized arrays.");
            const auto nr = (*ibegin)[0].nr();
            const auto nc = (*ibegin)[0].nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                for (size_t k = 0; k < K; ++k)
                {
                    const auto& arr = *i;
                    DLIB_CASSERT(arr[k].nr()==nr && arr[k].nc()==nc,
                        "\t input::to_tensor()"
                        << "\n\t When using std::array<matrix> as input, all matrices in a batch must have the same dimensions."
                        << "\n\t nr: " << nr
                        << "\n\t nc: " << nc
                        << "\n\t k:  " << k 
                        << "\n\t arr[k].nr(): " << arr[k].nr()
                        << "\n\t arr[k].nc(): " << arr[k].nc()
                    );
                }
            }

            
            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(ibegin,iend), K, nr, nc);

            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (size_t k = 0; k < K; ++k)
                {
                    for (long r = 0; r < nr; ++r)
                    {
                        for (long c = 0; c < nc; ++c)
                        {
                            if (is_same_type<T,unsigned char>::value)
                                *ptr++ = (*i)[k](r,c)/256.0;
                            else
                                *ptr++ = (*i)[k](r,c);
                        }
                    }
                }
            }

        }

        friend void serialize(const input& /*item*/, std::ostream& out)
        {
            serialize("input<array<matrix>>", out);
        }

        friend void deserialize(input& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input<array<matrix>>")
                throw serialization_error("Unexpected version found while deserializing dlib::input<array<matrix>>.");
        }

        friend std::ostream& operator<<(std::ostream& out, const input& /*item*/)
        {
            out << "input<array<matrix>>";
            return out;
        }

        friend void to_xml(const input& /*item*/, std::ostream& out)
        {
            out << "<input/>\n";
        }
    };

// ----------------------------------------------------------------------------------------

    template <typename T, typename MM>
    class input<array2d<T,MM>> 
    {
    public:
        typedef array2d<T,MM> input_type;

        input() {}
        input(const input&) {}

        template <long NR, long NC, typename mm, typename L>
        input(const input<matrix<T,NR,NC,mm,L>>&) {}

        bool image_contained_point ( const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space ( const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space ( const tensor& /*data*/, double /*scale*/, drectangle r ) const { return r; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin,iend) > 0);
            const auto nr = ibegin->nr();
            const auto nc = ibegin->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->nr()==nr && i->nc()==nc,
                    "\t input::to_tensor()"
                    << "\n\t All array2d objects given to to_tensor() must have the same dimensions."
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t i->nr(): " << i->nr()
                    << "\n\t i->nc(): " << i->nc()
                );
            }

            
            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(ibegin,iend), pixel_traits<T>::num, nr, nc);
            typedef typename pixel_traits<T>::basic_pixel_type bptype;

            const size_t offset = nr*nc;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        auto temp = pixel_to_vector<float>((*i)[r][c]);
                        auto p = ptr++;
                        for (long j = 0; j < temp.size(); ++j)
                        {
                            if (is_same_type<bptype,unsigned char>::value)
                                *p = temp(j)/256.0;
                            else
                                *p = temp(j);
                            p += offset;
                        }
                    }
                }
                ptr += offset*(data.k()-1);
            }

        }

        friend void serialize(const input&, std::ostream& out)
        {
            serialize("input<array2d>", out);
        }

        friend void deserialize(input&, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input<array2d>")
                throw serialization_error("Unexpected version found while deserializing dlib::input.");
        }
        friend std::ostream& operator<<(std::ostream& out, const input&)
        {
            out << "input<array2d>";
            return out;
        }

        friend void to_xml(const input&, std::ostream& out)
        {
            out << "<input/>\n";
        }
    };

// ----------------------------------------------------------------------------------------

    namespace detail {
        template <typename PYRAMID_TYPE>
        class input_image_pyramid
        {
        public:

            virtual ~input_image_pyramid() = 0;

            typedef PYRAMID_TYPE pyramid_type;

            unsigned long get_pyramid_padding() const { return pyramid_padding; }
            void set_pyramid_padding(unsigned long value) { pyramid_padding = value; }

            unsigned long get_pyramid_outer_padding() const { return pyramid_outer_padding; }
            void set_pyramid_outer_padding(unsigned long value) { pyramid_outer_padding = value; }

            bool image_contained_point(
                const tensor& data,
                const point& p
            ) const
            {
                auto&& rects = any_cast<std::vector<rectangle>>(data.annotation());
                DLIB_CASSERT(rects.size() > 0);
                return rects[0].contains(p + rects[0].tl_corner());
            }

            drectangle tensor_space_to_image_space(
                const tensor& data,
                drectangle r
            ) const
            {
                auto&& rects = any_cast<std::vector<rectangle>>(data.annotation());
                return tiled_pyramid_to_image<pyramid_type>(rects, r);
            }

            drectangle image_space_to_tensor_space (
                const tensor& data,
                double scale,
                drectangle r
            ) const
            {
                DLIB_CASSERT(0 < scale && scale <= 1, "scale: " << scale);
                auto&& rects = any_cast<std::vector<rectangle>>(data.annotation());
                return image_to_tiled_pyramid<pyramid_type>(rects, scale, r);
            }

        protected:

            template <typename forward_iterator>
            void to_tensor_init (
                forward_iterator ibegin,
                forward_iterator iend,
                resizable_tensor &data,
                unsigned int k
            ) const
            {

                DLIB_CASSERT(std::distance(ibegin, iend) > 0);
                auto nr = ibegin->nr();
                auto nc = ibegin->nc();
                // make sure all the input matrices have the same dimensions
                for (auto i = ibegin; i != iend; ++i)
                {
                    DLIB_CASSERT(i->nr() == nr && i->nc() == nc,
                                 "\t input_grayscale_image_pyramid::to_tensor()"
                                         << "\n\t All matrices given to to_tensor() must have the same dimensions."
                                         << "\n\t nr: " << nr
                                         << "\n\t nc: " << nc
                                         << "\n\t i->nr(): " << i->nr()
                                         << "\n\t i->nc(): " << i->nc()
                    );
                }

                long NR, NC;
                pyramid_type pyr;
                auto& rects = data.annotation().get<std::vector<rectangle>>();
                impl::compute_tiled_image_pyramid_details(pyr, nr, nc, pyramid_padding, pyramid_outer_padding, rects,
                                                          NR, NC);

                // initialize data to the right size to contain the stuff in the iterator range.
                data.set_size(std::distance(ibegin, iend), k, NR, NC);

                // We need to zero the image before doing the pyramid, since the pyramid
                // creation code doesn't write to all parts of the image.  We also take
                // care to avoid triggering any device to hosts copies.
                auto ptr = data.host_write_only();
                for (size_t i = 0; i < data.size(); ++i)
                    ptr[i] = 0;

            }

            // now build the image pyramid into data.  This does the same thing as
            // standard create_tiled_pyramid(), except we use the GPU if one is available.
            void create_tiled_pyramid (
                const std::vector<rectangle>& rects,
                resizable_tensor& data
            ) const
            {
                for (size_t i = 1; i < rects.size(); ++i) {
                    alias_tensor src(data.num_samples(), data.k(), rects[i - 1].height(), rects[i - 1].width());
                    alias_tensor dest(data.num_samples(), data.k(), rects[i].height(), rects[i].width());

                    auto asrc = src(data, data.nc() * rects[i - 1].top() + rects[i - 1].left());
                    auto adest = dest(data, data.nc() * rects[i].top() + rects[i].left());

                    tt::resize_bilinear(adest, data.nc(), data.nr() * data.nc(),
                                        asrc, data.nc(), data.nr() * data.nc());
                }
            }

            unsigned long pyramid_padding = 10;
            unsigned long pyramid_outer_padding = 11;
        };

        template <typename PYRAMID_TYPE>
        input_image_pyramid<PYRAMID_TYPE>::~input_image_pyramid() {}
    }

// ----------------------------------------------------------------------------------------

    template <typename PYRAMID_TYPE>
    class input_grayscale_image_pyramid : public detail::input_image_pyramid<PYRAMID_TYPE>
    {
    public:
        typedef matrix<unsigned char> input_type;
        typedef PYRAMID_TYPE pyramid_type;

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            this->to_tensor_init(ibegin, iend, data, 1);

            const auto rects = data.annotation().get<std::vector<rectangle>>();
            if (rects.size() == 0)
                return;

            // copy the first raw image into the top part of the tiled pyramid.  We need to
            // do this for each of the input images/samples in the tensor.
            auto ptr = data.host_write_only();
            for (auto i = ibegin; i != iend; ++i)
            {
                auto& img = *i;
                ptr += rects[0].top()*data.nc();
                for (long r = 0; r < img.nr(); ++r)
                {
                    auto p = ptr+rects[0].left();
                    for (long c = 0; c < img.nc(); ++c)
                        p[c] = (img(r,c))/256.0;
                    ptr += data.nc();
                }
                ptr += data.nc()*(data.nr()-rects[0].bottom()-1);
            }

            this->create_tiled_pyramid(rects, data);
        }

        friend void serialize(const input_grayscale_image_pyramid& item, std::ostream& out)
        {
            serialize("input_grayscale_image_pyramid", out);
            serialize(item.pyramid_padding, out);
            serialize(item.pyramid_outer_padding, out);
        }

        friend void deserialize(input_grayscale_image_pyramid& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input_grayscale_image_pyramid")
                throw serialization_error("Unexpected version found while deserializing dlib::input_grayscale_image_pyramid.");
            deserialize(item.pyramid_padding, in);
            deserialize(item.pyramid_outer_padding, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const input_grayscale_image_pyramid& item)
        {
            out << "input_grayscale_image_pyramid()";
            out << " pyramid_padding="<<item.pyramid_padding;
            out << " pyramid_outer_padding="<<item.pyramid_outer_padding;
            return out;
        }

        friend void to_xml(const input_grayscale_image_pyramid& item, std::ostream& out)
        {
            out << "<input_grayscale_image_pyramid"
                <<"' pyramid_padding='"<<item.pyramid_padding
                <<"' pyramid_outer_padding='"<<item.pyramid_outer_padding
                <<"'/>\n";
        }
    };

// ----------------------------------------------------------------------------------------

    template <typename PYRAMID_TYPE>
    class input_rgb_image_pyramid : public detail::input_image_pyramid<PYRAMID_TYPE>
    {
    public:
        typedef matrix<rgb_pixel> input_type;
        typedef PYRAMID_TYPE pyramid_type;

        input_rgb_image_pyramid (
        ) :
            avg_red(122.782),
            avg_green(117.001),
            avg_blue(104.298)
        {
        }

        input_rgb_image_pyramid (
            float avg_red_,
            float avg_green_,
            float avg_blue_
        ) : avg_red(avg_red_), avg_green(avg_green_), avg_blue(avg_blue_)
        {}

        float get_avg_red()   const { return avg_red; }
        float get_avg_green() const { return avg_green; }
        float get_avg_blue()  const { return avg_blue; }

        template <typename forward_iterator>
        void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            this->to_tensor_init(ibegin, iend, data, 3);

            const auto rects = data.annotation().get<std::vector<rectangle>>();
            if (rects.size() == 0)
                return;

            // copy the first raw image into the top part of the tiled pyramid.  We need to
            // do this for each of the input images/samples in the tensor.
            auto ptr = data.host_write_only();
            for (auto i = ibegin; i != iend; ++i)
            {
                auto& img = *i;
                ptr += rects[0].top()*data.nc();
                for (long r = 0; r < img.nr(); ++r)
                {
                    auto p = ptr+rects[0].left();
                    for (long c = 0; c < img.nc(); ++c)
                        p[c] = (img(r,c).red-avg_red)/256.0;
                    ptr += data.nc();
                }
                ptr += data.nc()*(data.nr()-rects[0].bottom()-1);

                ptr += rects[0].top()*data.nc();
                for (long r = 0; r < img.nr(); ++r)
                {
                    auto p = ptr+rects[0].left();
                    for (long c = 0; c < img.nc(); ++c)
                        p[c] = (img(r,c).green-avg_green)/256.0;
                    ptr += data.nc();
                }
                ptr += data.nc()*(data.nr()-rects[0].bottom()-1);

                ptr += rects[0].top()*data.nc();
                for (long r = 0; r < img.nr(); ++r)
                {
                    auto p = ptr+rects[0].left();
                    for (long c = 0; c < img.nc(); ++c)
                        p[c] = (img(r,c).blue-avg_blue)/256.0;
                    ptr += data.nc();
                }
                ptr += data.nc()*(data.nr()-rects[0].bottom()-1);
            }

            this->create_tiled_pyramid(rects, data);
        }

        friend void serialize(const input_rgb_image_pyramid& item, std::ostream& out)
        {
            serialize("input_rgb_image_pyramid2", out);
            serialize(item.avg_red, out);
            serialize(item.avg_green, out);
            serialize(item.avg_blue, out);
            serialize(item.pyramid_padding, out);
            serialize(item.pyramid_outer_padding, out);
        }

        friend void deserialize(input_rgb_image_pyramid& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input_rgb_image_pyramid" && version != "input_rgb_image_pyramid2")
                throw serialization_error("Unexpected version found while deserializing dlib::input_rgb_image_pyramid.");
            deserialize(item.avg_red, in);
            deserialize(item.avg_green, in);
            deserialize(item.avg_blue, in);
            if (version == "input_rgb_image_pyramid2")
            {
                deserialize(item.pyramid_padding, in);
                deserialize(item.pyramid_outer_padding, in);
            }
            else
            {
                item.pyramid_padding = 10;
                item.pyramid_outer_padding = 11;
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const input_rgb_image_pyramid& item)
        {
            out << "input_rgb_image_pyramid("<<item.avg_red<<","<<item.avg_green<<","<<item.avg_blue<<")";
            out << " pyramid_padding="<<item.pyramid_padding;
            out << " pyramid_outer_padding="<<item.pyramid_outer_padding;
            return out;
        }

        friend void to_xml(const input_rgb_image_pyramid& item, std::ostream& out)
        {
            out << "<input_rgb_image_pyramid r='"<<item.avg_red<<"' g='"<<item.avg_green
                <<"' b='"<<item.avg_blue
                <<"' pyramid_padding='"<<item.pyramid_padding
                <<"' pyramid_outer_padding='"<<item.pyramid_outer_padding
                <<"'/>\n";
        }

    private:
        float avg_red;
        float avg_green;
        float avg_blue;
    };

// ----------------------------------------------------------------------------------------

    class input_grayscale_image_stack
    {
    public:
        typedef std::vector<matrix<uint8_t>> input_type;

        input_grayscale_image_stack( // If average values are not passed, then 127.5f is assumed
        )
        {}

        input_grayscale_image_stack(
            const std::vector<float>& average_values_
        ) : average_values(average_values_)
        {}

        const std::vector<float>& get_average_values() const { return average_values; }

        bool image_contained_point(const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space(const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space(const tensor& /*data*/, double /*scale*/, drectangle r) const { return r; }

        template <typename forward_iterator>
        void to_tensor(
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin, iend) > 0);
            DLIB_CASSERT(!ibegin->empty());
            const auto images_in_stack = ibegin->size();
            const auto nr = ibegin->begin()->nr();
            const auto nc = ibegin->begin()->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->size() == images_in_stack,
                    "\t input_grayscale_image_stack::to_tensor()"
                    << "\n\t All image stacks given to to_tensor() must have the same size."
                    << "\n\t images_in_stack: " << images_in_stack
                    << "\n\t i->size: " << i->size()
                );

                for (auto j = i->begin(); j != i->end(); ++j)
                {
                    DLIB_CASSERT(j->nr() == nr && j->nc() == nc,
                        "\t input_grayscale_image_stack::to_tensor()"
                        << "\n\t All matrices given to to_tensor() must have the same dimensions."
                        << "\n\t nr: " << nr
                        << "\n\t nc: " << nc
                        << "\n\t j->nr(): " << j->nr()
                        << "\n\t j->nc(): " << j->nc()
                    );
                }
            }


            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(ibegin, iend), images_in_stack, nr, nc);

            // if average values are supplied, then they need to match the stack size
            // (if they are not supplied, then 127.5f is assumed)
            DLIB_CASSERT(average_values.empty() || average_values.size() == images_in_stack);
            const bool has_average_values = !average_values.empty();

            const size_t offset = nr*nc;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        auto p = ptr++;
                        for (size_t j = 0; j < images_in_stack; ++j)
                        {
                            const float avg = has_average_values ? average_values[j] : 127.5f;
                            const uint8_t temp = (*i)[j](r, c);
                            *p = (temp - avg) / 256.f;
                            p += offset;
                        }
                    }
                }
                ptr += offset*(data.k() - 1);
            }

        }

        friend void serialize(const input_grayscale_image_stack& item, std::ostream& out)
        {
            serialize("input_grayscale_image_stack", out);
            serialize(item.average_values, out);
        }

        friend void deserialize(input_grayscale_image_stack& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input_grayscale_image_stack")
                throw serialization_error("Unexpected version found while deserializing dlib::input_grayscale_image_stack.");
            deserialize(item.average_values, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const input_grayscale_image_stack& item)
        {
            out << "input_grayscale_image_stack(";
            for (size_t j = 0; j < item.average_values.size(); ++j) {
                if (j > 0) {
                    out << ",";
                }
                out << item.average_values[j];
            }
            out << ")";
            return out;
        }

        friend void to_xml(const input_grayscale_image_stack& item, std::ostream& out)
        {
            out << "<input_grayscale_image_stack>";
            for (size_t j = 0; j < item.average_values.size(); ++j) {
                out << "<stack_position index=\"" << j << "\" average_value=\"" << item.average_values[j] << "/>";
            }
            out << "</input_grayscale_image_stack>";
        }

    private:
        std::vector<float> average_values;
    };

// ----------------------------------------------------------------------------------------

    class input_rgb_image_stack
    {
    public:
        typedef std::vector<matrix<rgb_pixel>> input_type;

        struct rgb_pixel_float_values {
            float red   = std::numeric_limits<float>::quiet_NaN();
            float green = std::numeric_limits<float>::quiet_NaN();
            float blue  = std::numeric_limits<float>::quiet_NaN();
        };

        input_rgb_image_stack( // If average values are not passed, then the values defined already in input_rgb_image are assumed
        )
        {}

        input_rgb_image_stack(
            const std::vector<rgb_pixel_float_values>& average_values_
        ) : average_values(average_values_)
        {}

        const std::vector<rgb_pixel_float_values>& get_average_values() const { return average_values; }

        bool image_contained_point(const tensor& data, const point& p) const { return get_rect(data).contains(p); }
        drectangle tensor_space_to_image_space(const tensor& /*data*/, drectangle r) const { return r; }
        drectangle image_space_to_tensor_space(const tensor& /*data*/, double /*scale*/, drectangle r) const { return r; }

        template <typename forward_iterator>
        void to_tensor(
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
        ) const
        {
            DLIB_CASSERT(std::distance(ibegin, iend) > 0);
            DLIB_CASSERT(!ibegin->empty());
            const auto images_in_stack = ibegin->size();
            const auto nr = ibegin->begin()->nr();
            const auto nc = ibegin->begin()->nc();
            // make sure all the input matrices have the same dimensions
            for (auto i = ibegin; i != iend; ++i)
            {
                DLIB_CASSERT(i->size() == images_in_stack,
                    "\t input_rgb_image_stack::to_tensor()"
                    << "\n\t All image stacks given to to_tensor() must have the same size."
                    << "\n\t images_in_stack: " << images_in_stack
                    << "\n\t i->size: " << i->size()
                );

                for (auto j = i->begin(); j != i->end(); ++j)
                {
                    DLIB_CASSERT(j->nr() == nr && j->nc() == nc,
                        "\t input_rgb_image_stack::to_tensor()"
                        << "\n\t All matrices given to to_tensor() must have the same dimensions."
                        << "\n\t nr: " << nr
                        << "\n\t nc: " << nc
                        << "\n\t j->nr(): " << j->nr()
                        << "\n\t j->nc(): " << j->nc()
                    );
                }
            }


            // initialize data to the right size to contain the stuff in the iterator range.
            data.set_size(std::distance(ibegin, iend), 3 * images_in_stack, nr, nc);

            // if average values are supplied, then they need to match the stack size
            // (if they are not supplied, then values defined already in input_rgb_image are assumed)
            DLIB_CASSERT(average_values.empty() || average_values.size() == images_in_stack);
            const bool has_average_values = !average_values.empty();

            const size_t offset = nr*nc;
            auto ptr = data.host();
            for (auto i = ibegin; i != iend; ++i)
            {
                for (long r = 0; r < nr; ++r)
                {
                    for (long c = 0; c < nc; ++c)
                    {
                        auto p = ptr++;
                        for (size_t j = 0; j < images_in_stack; ++j)
                        {
                            const float avg_red   = has_average_values ? average_values[j].red   : 122.782;
                            const float avg_green = has_average_values ? average_values[j].green : 117.001;
                            const float avg_blue  = has_average_values ? average_values[j].blue  : 104.298;

                            const rgb_pixel& temp = (*i)[j](r, c);
                            *p = (temp.red   - avg_red  ) / 256.0;
                            p += offset;
                            *p = (temp.green - avg_green) / 256.0;
                            p += offset;
                            *p = (temp.blue  - avg_blue ) / 256.0;
                            p += offset;
                        }
                    }
                }
                ptr += offset*(data.k() - 1);
            }
        }

        friend void serialize(const input_rgb_image_stack& item, std::ostream& out)
        {
            serialize("input_rgb_image_stack", out);
            
            // TODO: this should use std::vector's serialization instead, but didn't know
            //       where to put rgb_pixel_float_values's serialization so that it would
            //       have been found correctly.
            serialize(item.average_values.size(), out);
            for (size_t j = 0; j < item.average_values.size(); ++j) {
                const auto& i = item.average_values[j];
                serialize(i.red,   out);
                serialize(i.green, out);
                serialize(i.blue,  out);
            }
        }

        friend void deserialize(input_rgb_image_stack& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "input_rgb_image_stack")
                throw serialization_error("Unexpected version found while deserializing dlib::input_rgb_image_stack.");

            // TODO: this should use std::vector's deserialization instead, but didn't know
            //       where to put rgb_pixel_float_values's deserialization so that it would
            //       have been found correctly.
            size_t size;
            deserialize(size, in);
            item.average_values.resize(size);
            for (size_t j = 0; j < size; ++j) {
                auto& i = item.average_values[j];
                deserialize(i.red,   in);
                deserialize(i.green, in);
                deserialize(i.blue,  in);
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const input_rgb_image_stack& item)
        {
            out << "input_rgb_image_stack(";
            for (size_t j = 0; j < item.average_values.size(); ++j) {
                if (j > 0) {
                    out << ";";
                }
                out << "(" 
                    << item.average_values[j].red   << "," 
                    << item.average_values[j].green << "," 
                    << item.average_values[j].blue  << ")";
            }
            out << ")";
            return out;
        }

        friend void to_xml(const input_rgb_image_stack& item, std::ostream& out)
        {
            out << "<input_rgb_image_stack>";
            for (size_t j = 0; j < item.average_values.size(); ++j) {
                out << "<stack_position index=\"" << j << "\" "
                    << " average_red=\""   << item.average_values[j].red   << "\""
                    << " average_green=\"" << item.average_values[j].green << "\""
                    << " average_blue=\""  << item.average_values[j].blue  << "\"/>";
            }
            out << "</input_rgb_image_stack>";
        }

    private:
        std::vector<rgb_pixel_float_values> average_values;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_INPUT_H_

