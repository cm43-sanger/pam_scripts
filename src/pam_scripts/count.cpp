#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <zlib.h>
#include "kseq.h"

KSEQ_INIT(gzFile, gzread)

uint8_t encode_base(uint8_t base)
{
    switch (base)
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
        return UINT8_MAX;
    }
}

// std::vector<uint64_t> extract_kmers_destructive(kseq_t *seq, uint8_t k)
// {
//     std::vector<uint64_t> kmers;
//     size_t l = seq->seq.l;
//     if (l < k)
//         return kmers;
//     uint8_t *s = reinterpret_cast<uint8_t *>(seq->seq.s);
//     for (size_t i = 0; i < l; i++)
//         s[i] = encode_base(s[i]);
//     size_t stop = -1;
//     int shift = 2 * (k - 1);
//     uint64_t mask = (1ULL << (2 * k)) - 1, flip = 0b11;
//     do
//     {
//         size_t start = ++stop; // skip invalid base
//         for (; (stop < l) && (s[stop] != UINT8_MAX); stop++)
//         {
//         }
//         if ((stop - start) < k)
//             continue;
//         size_t i = start;
//         uint64_t forward = 0, reverse = 0;
//         for (; i < start + k; i++)
//         {
//             uint64_t base = s[i];
//             forward = (forward << 2) | base;
//             reverse = (reverse >> 2) | ((flip ^ base) << shift);
//         }
//         kmers.push_back(forward < reverse ? forward : reverse);
//         for (; i < stop; i++)
//         {
//             uint64_t base = s[i];
//             forward = mask & ((forward << 2) | base);
//             reverse = (reverse >> 2) | ((flip ^ base) << shift);
//             kmers.push_back(forward < reverse ? forward : reverse);
//         }
//     } while (stop < l);
//     return kmers;
// }

std::vector<uint64_t> extract_kmers_destructive(kseq_t *seq, uint8_t k)
{
    std::vector<uint64_t> kmers;
    size_t l = seq->seq.l;
    if (l < k)
        return kmers;
    uint8_t *s = reinterpret_cast<uint8_t *>(seq->seq.s);
    for (size_t i = 0; i < l; i++)
        s[i] = encode_base(s[i]);
    int shift = 2 * (k - 1);
    uint64_t mask = (1ULL << (2 * k)) - 1;
    uint64_t flip = 0b11;
    for (size_t stop = 0; stop < l; stop++)
    { // extra increment of stop skips invalid bases
        size_t start = stop;
        for (; (stop < l) && (s[stop] != UINT8_MAX); stop++)
        {
        }
        if ((stop - start) < k)
            continue;
        size_t i = start;
        uint64_t forward = 0, reverse = 0;
        for (; i < start + k; i++)
        {
            uint64_t base = s[i];
            forward = (forward << 2) | base;
            reverse = (reverse >> 2) | ((flip ^ base) << shift);
        }
        kmers.push_back(forward < reverse ? forward : reverse);
        for (; i < stop; i++)
        {
            uint64_t base = s[i];
            forward = mask & ((forward << 2) | base);
            reverse = (reverse >> 2) | ((flip ^ base) << shift);
            kmers.push_back(forward < reverse ? forward : reverse);
        }
    }
    return kmers;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "Usage: %s <in.fastq.gz|in.fastq|->\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    gzFile fp = std::strcmp(path, "-") ? gzopen(path, "r") : gzdopen(fileno(stdin), "r");
    if (!fp)
        return 1;

    kseq_t *seq = kseq_init(fp);
    size_t count = 0;
    while (kseq_read(seq) > -1)
    {
        ++count;
        // std::fprintf(stderr, "%zu\n", count);
        std::vector<uint64_t> kmers = extract_kmers_destructive(seq, 31);
    }

    std::fprintf(stderr, "Total reads: %zu\n", count);

    kseq_destroy(seq);
    gzclose(fp);
    return 0;
}
