#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kmc_api/kmer_api.h"
#include "kmc_api/kmc_file.h"
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

// Read all kmers from a KMC database and estimate the coverage
double estimate_coverage(const std::string &db_path)
{
    CKMCFile kmc_file;
    if (!kmc_file.OpenForListing(db_path))
        throw std::runtime_error("Failed to open KMC database: " + db_path);
    uint32_t kmer_len = kmc_file.KmerLength();
    if (kmer_len > 31)
        throw std::runtime_error("k<=31 is required");
    uint64_t num_kmers = kmc_file.KmerCount();
    uint64_t total = 0;
    CKmerAPI kmer(kmer_len); // ignored
    uint32_t count;
    uint64_t i = 0;
    for (; kmc_file.ReadNextKmer(kmer, count); i++)
        total += count;
    if (i != num_kmers)
        throw std::runtime_error("Insufficient kmers in database");
    kmc_file.Close();
    return static_cast<double>(total) / static_cast<double>(num_kmers);
}

// Helper to pack a k<33 CKmerAPI object into a uint64_t word
uint64_t pack_kmer(CKmerAPI &kmer, uint32_t kmer_len)
{
    uint64_t word = 0;
    for (uint32_t pos = 0; pos < kmer_len; pos++)
        word = (word << 2) | kmer.get_num_symbol(pos); // 0=A, 1=C, 2=G, 3=T
    return word;
}

// Read all kmers from a KMC database and return a NumPy array of uint64_t
py::array_t<uint64_t> load_kmers(const std::string &db_path)
{
    CKMCFile kmc_file;
    if (!kmc_file.OpenForListing(db_path))
        throw std::runtime_error("Failed to open KMC database: " + db_path);
    uint32_t kmer_len = kmc_file.KmerLength();
    if (kmer_len > 31)
        throw std::runtime_error("k<=31 is required");
    uint64_t num_kmers = kmc_file.KmerCount();
    py::array_t<uint64_t> kmers({(py::ssize_t)num_kmers}); // Allocate NumPy array
    auto kmers_buf = kmers.mutable_unchecked<1>();
    CKmerAPI kmer(kmer_len);
    uint32_t count; // ignored
    uint64_t i = 0;
    for (; kmc_file.ReadNextKmer(kmer, count); i++)
        kmers_buf(i) = pack_kmer(kmer, kmer_len);
    if (i != num_kmers)
        throw std::runtime_error("Insufficient kmers in database");
    kmc_file.Close();
    return kmers;
}

PYBIND11_MODULE(_kmc, m)
{
    m.doc() = "Read all k-mers from a KMC database into a NumPy uint64_t array";
    m.def("estimate_coverage", &estimate_coverage,
          py::arg("path"),
          "Estimate the coverage of a KMC database.");
    m.def("load_kmers", &load_kmers,
          py::arg("path"),
          "Load all k-mers from a KMC database.");
}
