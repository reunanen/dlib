// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// Copyright (C) 2018  Juha Reunanen (juha.reunanen@tomaattinen.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FLEXIBLE_PIPE_KERNEl_1_ 
#define DLIB_FLEXIBLE_PIPE_KERNEl_1_ 

#include "../algs.h"
#include "../threads.h"

namespace dlib
{

    template <
        typename T
        >
    class dynamic_pipe 
    {
        /*!
            INITIAL VALUE
                - enabled == true
                - data == an empty deque of T objects.
                - dequeue_waiters == 0
                - enqueue_waiters == 0
                - unblock_sig_waiters == 0

            CONVENTION
                - is_enabled() == enabled

                - m == the mutex used to lock access to all the members of this class

                - dequeue_waiters == the number of threads blocked on calls to dequeue()
                - enqueue_waiters == the number of threads blocked on calls to wait_until_empty()
                - unblock_sig_waiters == the number of threads blocked on calls to 
                  wait_for_num_blocked_dequeues() and the destructor.  (i.e. the number of
                  blocking calls to unblock_sig.wait())

                - dequeue_sig == the signaler that threads blocked on calls to dequeue() wait on
                - enqueue_sig == the signaler that threads blocked on calls to wait_until_empty() wait on.
                - unblock_sig == the signaler that is signaled when a thread stops blocking on a call
                  to dequeue().  It is also signaled when a dequeue that will probably block is called.
                  The destructor and wait_for_num_blocked_dequeues are the only things that will wait on
                  this signaler.
        !*/

    public:
        typedef T type;

        explicit dynamic_pipe (
        );

        virtual ~dynamic_pipe(
        );

        void clear (
        );

        void wait_until_empty (
        ) const;

        void wait_for_num_blocked_dequeues (
            unsigned long num
        )const;

        void enable (
        );

        void disable (
        );

        bool is_enqueue_enabled (
        ) const;

        void disable_enqueue (
        );

        void enable_enqueue (
        );

        bool is_dequeue_enabled (
        ) const;

        void disable_dequeue (
        );

        void enable_dequeue (
        );

        bool is_enabled (
        ) const;

        size_t size (
        ) const;

        bool enqueue (
            T&& item
        );

        bool dequeue (
            T& item
        );

        bool dequeue_or_timeout (
            T& item,
            unsigned long timeout
        );

    private:

        bool enabled;

        std::deque<T> data;

        mutex m;
        signaler dequeue_sig;
        signaler enqueue_sig;
        signaler unblock_sig;

        unsigned long dequeue_waiters;
        mutable unsigned long enqueue_waiters;
        mutable unsigned long unblock_sig_waiters;
        bool enqueue_enabled;
        bool dequeue_enabled;

        // restricted functions
        dynamic_pipe(const dynamic_pipe&);        // copy constructor
        dynamic_pipe& operator=(const dynamic_pipe&);    // assignment operator

    };    

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                      member function definitions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    dynamic_pipe<T>::
    dynamic_pipe (  
    ) : 
        enabled(true),
        dequeue_sig(m),
        enqueue_sig(m),
        unblock_sig(m),
        dequeue_waiters(0),
        enqueue_waiters(0),
        unblock_sig_waiters(0),
        enqueue_enabled(true),
        dequeue_enabled(true)
    {
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    dynamic_pipe<T>::
    ~dynamic_pipe (
    )
    {
        auto_mutex M(m);
        ++unblock_sig_waiters;

        // first make sure no one is blocked on any calls to enqueue() or dequeue()
        enabled = false;
        dequeue_sig.broadcast();
        enqueue_sig.broadcast();
        unblock_sig.broadcast();

        // wait for all threads to unblock
        while (dequeue_waiters > 0 || enqueue_waiters > 0 || unblock_sig_waiters > 1)
            unblock_sig.wait();

        data.clear();
        --unblock_sig_waiters;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void dynamic_pipe<T>::
    clear (
    )
    {
        auto_mutex M(m);
        data.clear();

        // let any calls to enqueue() know that the pipe is now empty
        if (enqueue_waiters > 0)
            enqueue_sig.broadcast();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void dynamic_pipe<T>::
    wait_until_empty (
    ) const
    {
        auto_mutex M(m);
        // this function is sort of like a call to enqueue so treat it like that
        ++enqueue_waiters;

        while (!data.empty() && enabled && dequeue_enabled)
            enqueue_sig.wait();

        // let the destructor know we are ending if it is blocked waiting
        if (enabled == false)
            unblock_sig.broadcast();

        --enqueue_waiters;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void dynamic_pipe<T>::
    enable (
    )
    {
        auto_mutex M(m);
        enabled = true;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void dynamic_pipe<T>::
    disable (
    )
    {
        auto_mutex M(m);
        enabled = false;
        dequeue_sig.broadcast();
        enqueue_sig.broadcast();
        unblock_sig.broadcast();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    bool dynamic_pipe<T>::
    is_enabled (
    ) const
    {
        auto_mutex M(m);
        return enabled;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    size_t dynamic_pipe<T>::
    size (
    ) const
    {
        auto_mutex M(m);
        return data.size();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    bool dynamic_pipe<T>::
    enqueue (
        T&& item
    )
    {
        auto_mutex M(m);

        if (enabled == false || enqueue_enabled == false)
        {
            // let the destructor know we are unblocking
            unblock_sig.broadcast();
            return false;
        }

        data.push_back(std::move(item));

        // wake up a call to dequeue() if there are any currently blocked
        if (dequeue_waiters > 0)
            dequeue_sig.signal();

        return true;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    bool dynamic_pipe<T>::
    dequeue (
        T& item
    )
    {
        auto_mutex M(m);
        ++dequeue_waiters;

        if (data.empty())
        {
            // notify wait_for_num_blocked_dequeues()
            if (unblock_sig_waiters > 0)
                unblock_sig.broadcast();

            // notify any blocked enqueue_or_timeout() calls
            if (enqueue_waiters > 0)
                enqueue_sig.broadcast();
        }

        // wait until there is something in the pipe or we are disabled 
        while (data.empty() && enabled && dequeue_enabled)
            dequeue_sig.wait();

        if (enabled == false || dequeue_enabled == false)
        {
            --dequeue_waiters;
            // let the destructor know we are unblocking
            unblock_sig.broadcast();
            return false;
        }

        item = std::move(data.front());
        data.pop_front();

        // wake up a call to enqueue() if there are any currently blocked
        if (enqueue_waiters > 0)
            enqueue_sig.broadcast();

        --dequeue_waiters;
        return true;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    bool dynamic_pipe<T>::
    dequeue_or_timeout (
        T& item,
        unsigned long timeout
    )
    {
        auto_mutex M(m);
        ++dequeue_waiters;

        if (data.empty())
        {
            // notify wait_for_num_blocked_dequeues()
            if (unblock_sig_waiters > 0)
                unblock_sig.broadcast();

            // notify any blocked enqueue_or_timeout() calls
            if (enqueue_waiters > 0)
                enqueue_sig.broadcast();
        }

        bool timed_out = false;
        // wait until there is something in the pipe or we are disabled or we timeout.
        while (data.empty() && enabled && dequeue_enabled)
        {
            if (timeout == 0 || dequeue_sig.wait_or_timeout(timeout) == false)
            {
                timed_out = true;
                break;
            }
        }

        if (enabled == false || timed_out || dequeue_enabled == false)
        {
            --dequeue_waiters;
            // let the destructor know we are unblocking
            unblock_sig.broadcast();
            return false;
        }

        item = std::move(data.front());
        data.pop_front();

        // wake up a call to enqueue() if there are any currently blocked
        if (enqueue_waiters > 0)
            enqueue_sig.broadcast();

        --dequeue_waiters;
        return true;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void dynamic_pipe<T>::
    wait_for_num_blocked_dequeues (
        unsigned long num
    )const
    {
        auto_mutex M(m);
        ++unblock_sig_waiters;

        while (dequeue_waiters < num && enabled && dequeue_enabled)
            unblock_sig.wait();

        // let the destructor know we are ending if it is blocked waiting
        if (enabled == false)
            unblock_sig.broadcast();

        --unblock_sig_waiters;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    bool dynamic_pipe<T>::
    is_enqueue_enabled (
    ) const
    {
        auto_mutex M(m);
        return enqueue_enabled;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void dynamic_pipe<T>::
    disable_enqueue (
    )
    {
        auto_mutex M(m);
        enqueue_enabled = false;
        enqueue_sig.broadcast();
    }


// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void dynamic_pipe<T>::
    enable_enqueue (
    )
    {
        auto_mutex M(m);
        enqueue_enabled = true;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    bool dynamic_pipe<T>::
    is_dequeue_enabled (
    ) const
    {
        auto_mutex M(m);
        return dequeue_enabled;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void dynamic_pipe<T>::
    disable_dequeue (
    )
    {
        auto_mutex M(m);
        dequeue_enabled = false;
        dequeue_sig.broadcast();
    }


// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void dynamic_pipe<T>::
    enable_dequeue (
    )
    {
        auto_mutex M(m);
        dequeue_enabled = true;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FLEXIBLE_PIPE_KERNEl_1_

