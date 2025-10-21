#include <cstdio>
#include <thread>
#include <vector>
#include <random>
#include <limits>
#include "parallel_hashmap/phmap.h"

// Your increment_saturating function
template <typename CounterType>
static inline void increment_saturating(CounterType &counter, uint64_t key)
{
    auto result = counter.try_emplace(key, 1);
    bool inserted = result.second;
    // if (!inserted)
    // {
    //     counter.modify_if(key, [](typename CounterType::value_type &kv)
    //                       {
    //         if (kv.second != std::numeric_limits<typename CounterType::mapped_type>::max())
    //             ++kv.second; });
    // }
}

int main()
{
    constexpr size_t num_threads = 8;
    constexpr size_t num_ops_per_thread = 100000;

    phmap::parallel_flat_hash_map<uint64_t, uint16_t, std::hash<uint64_t>, std::equal_to<uint64_t>,
                                  std::allocator<std::pair<const uint64_t, uint16_t>>, 4, std::mutex>
        counter;

    // counter.try_emplace_l()

    auto worker = [&counter, num_ops_per_thread]()
    {
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<uint64_t> dist(0, 127);

        for (size_t i = 0; i < num_ops_per_thread; ++i)
        {
            uint64_t key = dist(rng);
            increment_saturating(counter, key);
        }
    };

    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back(worker);

    for (auto &th : threads)
        th.join();

    // Print counts
    for (auto &kv : counter)
        std::printf("Key: %llu Count: %u\n", kv.first, kv.second);

    return 0;
}
