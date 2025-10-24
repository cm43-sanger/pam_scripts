#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>

namespace py = pybind11;

size_t get_intersection_size(const uint64_t *A, size_t nA, const uint64_t *B, size_t nB)
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

double get_jaccard_index(const uint64_t *A, size_t nA, const uint64_t *B, size_t nB)
{
    if ((nA == 0) || (nB == 0))
        return 0.0;
    size_t intersection_size = get_intersection_size(A, nA, B, nB);
    size_t union_size = nA + nB - intersection_size;
    return static_cast<double>(intersection_size) / static_cast<double>(union_size);
}

py::array_t<double> get_pairwise_jaccard_distances(const std::vector<py::array_t<uint64_t>> arrays)
{
    size_t N = arrays.size();
    py::array_t<double> pairwise_jaccard_distance({N, N});
    auto buf = pairwise_jaccard_distance.mutable_unchecked<2>();
    for (size_t i = 0; i < N; i++)
    {
        std::cout << "Processing row " << i + 1 << " / " << N << std::endl;
        buf(i, i) = 0.0; // diagonal is zero
        const py::array_t<uint64_t> &a = arrays[i];
        for (size_t j = 0; j < i; j++)
        {
            const py::array_t<uint64_t> &b = arrays[j]; // same here
            double jaccard_distance = 1.0 - get_jaccard_index(a.data(), a.size(), b.data(), b.size());
            buf(i, j) = jaccard_distance;
            buf(j, i) = jaccard_distance; // symmetric
        }
    }
    return pairwise_jaccard_distance;
}

py::array_t<double> get_jaccard_distances(
    const std::vector<py::array_t<uint64_t>> &reference_arrays,
    const std::vector<py::array_t<uint64_t>> &query_arrays)
{
    size_t N = reference_arrays.size(); // number of reference arrays (columns)
    size_t M = query_arrays.size();     // number of query arrays (rows)
    py::array_t<double> jaccard_distances({M, N});
    auto buf = jaccard_distances.mutable_unchecked<2>();
    for (size_t i = 0; i < M; i++)
    {
        std::cout << "Processing query " << i + 1 << " / " << M << std::endl;
        const py::array_t<uint64_t> &query = query_arrays[i];
        for (size_t j = 0; j < N; j++)
        {
            const py::array_t<uint64_t> &ref = reference_arrays[j];
            buf(i, j) = 1.0 - get_jaccard_index(query.data(), query.size(), ref.data(), ref.size());
        }
    }
    return jaccard_distances;
}

PYBIND11_MODULE(_jaccard, m)
{
    m.def("get_pairwise_jaccard_distances", &get_pairwise_jaccard_distances);
    m.def("get_jaccard_distances", &get_jaccard_distances);
}
