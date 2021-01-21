#ifndef __UTIL_HDR
#define __UTIL_HDR

#if defined(__APPLE__) || defined(__MACOSX)
#include <sys/time.h>
#else
#include <stdint.h>
#include <unistd.h>
#endif

#include <iostream>
#include <fstream>
#include <string>

#include <cstdlib>

namespace util {

  inline std::string loadProgram(std::string input)
  {
    std::ifstream stream(input.c_str());
    if (!stream.is_open()) {
      std::cout << "Cannot open file: " << input << std::endl;
      exit(1);
    }

    return std::string(
        std::istreambuf_iterator<char>(stream),
        (std::istreambuf_iterator<char>()));
  }

#if 1
  class Timer
  {
    private:
#if defined(__APPLE__) || defined(__MACOSX)
      struct timeval startTime_;
#else
      struct timespec startTime_;
#endif

      template <typename T>
        T _max(T a,T b)
        {
          return (a > b ? a : b);
        }

      uint64_t getTime(unsigned long long scale)
      {
        uint64_t ticks;
#if defined(__APPLE__) || defined(__MACOSX)
        // WARNING: THIS IS PROBABLY BROKEN
        struct timeval tv;
        gettimeofday(&tv, 0);
        // check for overflow
        if ((tv.tv_usec - startTime_.tv_usec) < 0)
        {
          // Remove a second from the second field and add it to the
          // microseconds fields to prevent overflow.
          // Then scale.
          ticks = (uint64_t) (tv.tv_sec - startTime_.tv_sec - 1) * scale
            + (uint64_t) ((1000ULL * 1000ULL) + tv.tv_usec - startTime_.tv_usec)
            * scale / (1000ULL * 1000ULL);
        }
        else
        {
          ticks = (uint64_t) (tv.tv_sec - startTime_.tv_sec) * scale
            + (uint64_t) (tv.tv_usec - startTime_.tv_usec) * scale / (1000ULL * 1000ULL);
        }
#else
        struct timespec tp;
        ::clock_gettime(CLOCK_MONOTONIC, &tp);
        // check for overflow
        if ((tp.tv_nsec - startTime_.tv_nsec) < 0)
        {
          // Remove a second from the second field and add it to the
          // nanoseconds field to prevent overflow.
          // Then scale
          ticks = (uint64_t) (tp.tv_sec - startTime_.tv_sec - 1) * scale
            + (uint64_t) ((1000ULL * 1000ULL * 1000ULL) + tp.tv_nsec - startTime_.tv_nsec)
            * scale / (1000ULL * 1000ULL * 1000ULL);
        }
        else
        {
          ticks = (uint64_t) (tp.tv_sec - startTime_.tv_sec) * scale
            + (uint64_t) (tp.tv_nsec - startTime_.tv_nsec) * scale / (1000ULL * 1000ULL * 1000ULL);
        }
#endif //__APPLE__

        return ticks;
      }

    public:
      //! Constructor
      Timer()
      {
        reset();
      }

      //! Destructor
      ~Timer()
      {
      }

      /*!
       * \brief Resets timer such that in essence the elapsed time is zero
       * from this point.
       */
      void reset()
      {
#if defined(__APPLE__) || defined(__MACOSX)
        gettimeofday(&startTime_, 0);
#else
        ::clock_gettime(CLOCK_MONOTONIC, &startTime_);
#endif //__APPLE__
      }

      /*!
       * \brief Calculates the time since the last reset.
       * \returns The time in milli seconds since the last reset.
       */
      uint64_t getTimeMilliseconds(void)
      {
        return getTime(1000ULL);
      }

      /*!
       * \brief Calculates the time since the last reset.
       * \returns The time in nano seconds since the last reset.
       */
      uint64_t getTimeNanoseconds(void)
      {
        return getTime(1000ULL * 1000ULL * 1000ULL);
      }

      /*!
       * \brief Calculates the time since the last reset.
       * \returns The time in micro seconds since the last reset.
       */
      uint64_t getTimeMicroseconds(void)
      {
        return getTime(1000ULL * 1000ULL);
      }

      /*!
       * \brief Calculates the tick rate for millisecond counter.
       */
      float getMillisecondsTickRate(void)
      {
        return 1000.f;
      }

      /*!
       * \brief Calculates the tick rate for nanosecond counter.
       */
      float getNanosecondsTickRate(void)
      {
        return (float) (1000ULL * 1000ULL * 1000ULL);
      }

      /*!
       * \brief Calculates the tick rate for microsecond counter.
       */
      float getMicrosecondsTickRate(void)
      {
        return (float) (1000ULL * 1000ULL);
      }
  };
#endif

} // namespace util

#endif // __UTIL_HDR
