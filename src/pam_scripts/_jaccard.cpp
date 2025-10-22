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

// py::array_t<double> get_pairwise_jaccard(std::vector<py::array_t<uint64_t>> arrays)
// {
//     size_t N = arrays.size();
//     py::array_t<double> pairwise_jaccard({N, N});
//     auto buf = pairwise_jaccard.mutable_unchecked<2>();
//     for (size_t i = 0; i < N; i++)
//     {
//         buf(i, i) = 0.0; // diagonal is zero
//         auto a = arrays[i].unchecked<1>();
//         for (size_t j = 0; j < i; j++)
//         {
//             auto b = arrays[j].unchecked<1>();
//             double jaccard_index = get_jaccard_index(a.data(), a.shape(0), b.data(), b.shape(0));
//             buf(i, j) = jaccard_index;
//             buf(j, i) = jaccard_index; // symmetric
//         }
//     }
//     return pairwise_jaccard;
// }

py::array_t<double> get_pairwise_jaccard(std::vector<py::array_t<uint64_t>> arrays)
{
    size_t N = arrays.size();
    py::array_t<double> pairwise_jaccard({N, N});
    auto buf = pairwise_jaccard.mutable_unchecked<2>();
    for (size_t i = 0; i < N; i++)
    {
        std::cout << "Processing row " << i + 1 << " / " << N << std::endl;
        buf(i, i) = 0.0; // diagonal is zero
        py::array_t<uint64_t> &a = arrays[i];
        for (size_t j = 0; j < i; j++)
        {
            py::array_t<uint64_t> &b = arrays[j]; // same here
            double jaccard_index_val = get_jaccard_index(a.data(), a.size(), b.data(), b.size());
            buf(i, j) = jaccard_index_val;
            buf(j, i) = jaccard_index_val; // symmetric
        }
    }
    return pairwise_jaccard;
}

PYBIND11_MODULE(_jaccard, m)
{
    m.def("get_pairwise_jaccard", &get_pairwise_jaccard);
}
