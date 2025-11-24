#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

#define INVALID_BASE_CODE UINT8_MAX

static const size_t k = 31;

static inline uint8_t encode_base(char c)
{
    switch (c)
    {
    case 'A':
    case 'a':
        return 0b00;
    case 'C':
    case 'c':
        return 0b01;
    case 'G':
    case 'g':
        return 0b10;
    case 'T':
    case 't':
        return 0b11;
    default:
        return INVALID_BASE_CODE;
    }
}

static inline size_t encode_kmer(
    std::vector<uint8_t> &kmer, const std::string &s, size_t k)
{
    if (s.size() != k)
    {
        std::cerr << "Invalid kmer of length " << s.size() << ": "
                  << s << "\n";
        return 0;
    }
    kmer.assign((k + 3) / 4, 0);
    size_t valid_bases = 0;
    for (; valid_bases < k; valid_bases++)
    {
        uint8_t code = encode_base(s[valid_bases]);
        if (code == INVALID_BASE_CODE)
        {
            std::cerr << "Invalid base (" << s[valid_bases] << ") at position "
                      << valid_bases + 1 << " of kmer: " << s << "\n";
            break;
        }
        kmer[valid_bases / 4] |= code << (2 * (valid_bases % 4));
    }
    return valid_bases;
}

int main()
{
    std::vector<std::pair<std::vector<uint8_t>, uint64_t>> entries;
    std::ifstream in("./src/pam_scripts/data.txt");
    if (!in)
    {
        std::cerr << "Cannot open file\n";
        return 1;
    }
    std::string s;
    uint64_t count;
    while (in >> s >> count)
    {
        std::vector<uint8_t> kmer;
        size_t valid_bases = encode_kmer(kmer, s, k);
        if (valid_bases == k)
            entries.emplace_back(kmer, count);
    }
    std::cout << "Loaded " << entries.size() << " entries.\n";
    for (const auto &entry : entries)
    {
        const auto &kmer = entry.first;
        uint64_t count = entry.second;
        std::cout << kmer.size() << " " << count << "\n";
    }
    return 0;
}
