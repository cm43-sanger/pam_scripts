#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kmc_api/kmer_api.h"
#include "kmc_api/kmc_file.h"
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

// Helper to pack a k<33 CKmerAPI object into a uint64_t word
uint64_t pack_kmer(CKmerAPI &kmer, uint32_t kmer_len)
{
    uint64_t word = 0;
    for (uint32_t pos = 0; pos < kmer_len; pos++)
    {
        uint64_t sym = kmer.get_num_symbol(pos); // 0=A, 1=C, 2=G, 3=T
        std::cout << sym << ' ';
        word |= (sym << (2 * pos));
    }
    return word;
}

// Read all kmers from a KMC database and return a NumPy array of uint64_t
py::array_t<uint64_t> load_kmc_kmers(const std::string &db_path)
{
    std::cout << '\n'
              << db_path << '\n';
    CKMCFile kmc_file;
    if (!kmc_file.OpenForListing(db_path))
        throw std::runtime_error("Failed to open KMC database: " + db_path);
    uint32_t kmer_len = kmc_file.KmerLength();
    if (kmer_len > 32)
        throw std::runtime_error("pack_kmer only supports k <= 32");
    uint64_t num_kmers = kmc_file.KmerCount();

    std::cout << '\n'
              << kmer_len << ' ' << num_kmers << '\n';

    // Allocate NumPy array (num_kmers × words_per_kmer)
    py::array_t<uint64_t> kmers({(py::ssize_t)num_kmers});
    auto kmers_buf = kmers.mutable_unchecked<1>();

    CKmerAPI kmer(kmer_len);
    uint32_t count; // ignored
                    // for (uint64_t i = 0; kmc_file.ReadNextKmer(kmer, count); i++)
                    //     kmers_buf(i) = pack_kmer(kmer, kmer_len);
    std::cout << kmc_file.ReadNextKmer(kmer, count) << '\n';
    for (uint64_t i = 0; kmc_file.ReadNextKmer(kmer, count); i++)
    {
        std::cout << i << '\n';
        kmers_buf(i) = pack_kmer(kmer, kmer_len);
    }

    kmc_file.Close();
    return kmers;
}

PYBIND11_MODULE(_kmers, m)
{
    m.doc() = "Read all k-mers from a KMC database into a NumPy uint64_t array";
    m.def("load_kmc_kmers", &load_kmc_kmers,
          py::arg("path"),
          "Load all k-mers from a KMC database (ignoring counts).");
}