#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <iostream>

namespace py = pybind11;

static inline double estimate_occupancy(uint64_t filled, uint64_t total)
{
    return log1p(-static_cast<double>(filled) / static_cast<double>(total)) /
           log1p(-1.0 / static_cast<double>(total));
}

static inline std::pair<std::vector<size_t>, std::vector<size_t>> get_below_diagonal_partitions(
    size_t N, size_t num_threads)
{
    std::vector<size_t> i_parts, j_parts;
    i_parts.reserve(num_threads + 1);
    j_parts.reserve(num_threads + 1);
    i_parts.push_back(0);
    j_parts.push_back(0);
    size_t i = 1;
    size_t total = 0;
    size_t final_total = N * (N - 1) / 2;
    for (size_t t = 1; t < num_threads; t++)
    {
        size_t target = t * final_total / num_threads;
        size_t next_total;
        while ((next_total = total + i) <= target)
        {
            total = next_total;
            ++i;
        }
        i_parts.push_back(i);
        j_parts.push_back(target - total);
    }
    i_parts.push_back(N);
    j_parts.push_back(0);
    return {i_parts, j_parts};
}

static inline size_t get_intersection_size(const uint64_t *A, size_t nA, const uint64_t *B, size_t nB)
{
    size_t iA = 0, iB = 0, intersection_size = 0;
    while ((iA < nA) && (iB < nB))
    {
        uint64_t a = A[iA], b = B[iB];
        intersection_size += (a == b);
        iA += (a <= b);
        iB += (a >= b);
    }
    return intersection_size;
}

static inline double get_jaccard_index(const uint64_t *A, size_t nA, const uint64_t *B, size_t nB)
{
    if ((nA == 0) || (nB == 0))
        return 0.0;
    size_t intersection_size = get_intersection_size(A, nA, B, nB);
    size_t union_size = nA + nB - intersection_size;
    return static_cast<double>(intersection_size) / static_cast<double>(union_size);
}

static inline double get_distance(double jaccard_index, unsigned int kmer_length)
{
    if (jaccard_index == 0.0)
        return INFINITY;
    if ((jaccard_index < 0.0) || (jaccard_index > 1.0))
        return NAN;
    return -log((2.0 * jaccard_index) / (1.0 + jaccard_index)) / static_cast<double>(kmer_length);
}

py::array_t<double> get_pairwise_distances(
    const std::vector<py::array_t<uint64_t>> arrays, unsigned int kmer_length)
{
    size_t N = arrays.size();
    py::array_t<double> pairwise_distances({N, N});
    auto buf = pairwise_distances.mutable_unchecked<2>();
    for (size_t i = 0; i < N; i++)
    {
        std::cout << "Processing row " << i + 1 << " / " << N << std::endl;
        buf(i, i) = 0.0; // diagonal is zero
        const py::array_t<uint64_t> &a = arrays[i];
        for (size_t j = 0; j < i; j++)
        {
            const py::array_t<uint64_t> &b = arrays[j]; // same here
            double jaccard_index = get_jaccard_index(a.data(), a.size(), b.data(), b.size());
            double distance = get_distance(jaccard_index, kmer_length);
            buf(i, j) = distance;
            buf(j, i) = distance; // symmetric
        }
    }
    return pairwise_distances;
}

py::array_t<double> get_distances(
    const std::vector<py::array_t<uint64_t>> &reference_arrays,
    const std::vector<py::array_t<uint64_t>> &query_arrays,
    unsigned int kmer_length)
{
    size_t N = reference_arrays.size(); // number of reference arrays (columns)
    size_t M = query_arrays.size();     // number of query arrays (rows)
    py::array_t<double> distances({M, N});
    auto buf = distances.mutable_unchecked<2>();
    for (size_t i = 0; i < M; i++)
    {
        std::cout << "Processing query " << i + 1 << " / " << M << std::endl;
        const py::array_t<uint64_t> &query = query_arrays[i];
        for (size_t j = 0; j < N; j++)
        {
            const py::array_t<uint64_t> &ref = reference_arrays[j];
            double jaccard_index = get_jaccard_index(query.data(), query.size(), ref.data(), ref.size());
            buf(i, j) = get_distance(jaccard_index, kmer_length);
        }
    }
    return distances;
}

PYBIND11_MODULE(_jaccard, m)
{
    m.def("get_pairwise_distances", &get_pairwise_distances);
    m.def("get_distances", &get_distances);
}
