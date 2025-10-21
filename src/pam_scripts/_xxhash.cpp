#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <thread>
#include <vector>

namespace py = pybind11;

static inline uint64_t hash_u64(uint64_t x, uint64_t seed = 42)
{
    x += seed;
    x ^= x >> 33;
    x *= 0x9e3779b97f4a7c15ULL;
    x ^= x >> 33;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 33;
    return x;
}

static inline void hash_buf(
    const uint64_t *input, uint64_t *output, size_t n, uint64_t seed, unsigned int num_threads)
{
    num_threads = std::min<size_t>(num_threads, n / 1000);
    if (num_threads < 2)
    {
        for (size_t i = 0; i < n; i++)
            output[i] = hash_u64(input[i], seed);
        return;
    }
    std::vector<std::thread> threads(num_threads);
    size_t base_size = n / num_threads;
    size_t remainder = n % num_threads; // extra elements to distribute
    size_t start = 0;
    for (unsigned int t = 0; t < num_threads; t++)
    {
        size_t size = base_size + (t < remainder); // distribute remainder
        size_t stop = start + size;
        threads[t] = std::thread([input, output, seed, start, stop]()
                                 {
            for (size_t i = start; i < stop; i++) {
                output[i] = hash_u64(input[i], seed);
            } });
        start = stop;
    }
    if (start != n)
        throw std::runtime_error("Not finished");
    for (std::thread &thread : threads)
        thread.join();
}

py::array_t<uint64_t> hash_kmers(py::array_t<uint64_t> kmers, uint64_t seed = 42, unsigned int num_threads = 1)
{
    auto kmers_buf = kmers.unchecked<1>();
    size_t num_kmers = kmers_buf.shape(0);
    py::array_t<uint64_t> hashed_kmers({(py::ssize_t)num_kmers}); // Allocate NumPy array
    if (num_kmers != 0)
    {
        auto hashed_kmers_buf = hashed_kmers.mutable_unchecked<1>();
        hash_buf(kmers_buf.data(0), hashed_kmers_buf.mutable_data(0), num_kmers, seed, num_threads);
    }
    return hashed_kmers;
}

PYBIND11_MODULE(_xxhash, m)
{
    m.doc() = "Hash NumPy uint64_t array of kmers";
    m.def("hash_kmers", &hash_kmers,
          py::arg("kmers"), py::arg("num_threads") = 0, py::arg("seed") = 42,
          "Hash NumPy uint64_t array of kmers.");
}
