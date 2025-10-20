#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <zlib.h>
#include <thread>
#include <atomic>
#include "kseq.h"
#include "concurrentqueue.h"

KSEQ_INIT(gzFile, gzread)

static inline uint8_t encode_base(uint8_t base)
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

static inline std::vector<uint64_t> extract_kmers_destructive(
    kstring_t *seq, uint8_t k)
{
    std::vector<uint64_t> kmers;
    size_t l = seq->l;
    if (l < k)
        return kmers;
    uint8_t *s = reinterpret_cast<uint8_t *>(seq->s);
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

static inline void worker_func(
    std::vector<kstring_t> &buckets,
    moodycamel::ConcurrentQueue<size_t> &filled_buckets,
    moodycamel::ConcurrentQueue<size_t> &free_buckets,
    std::atomic<bool> &done,
    uint8_t k)
{
    size_t i;
    while (true)
    {
        if (filled_buckets.try_dequeue(i))
        {
            std::vector<uint64_t> kmers = extract_kmers_destructive(&buckets[i], k);
            free_buckets.enqueue(i);
        }
        else if (done.load())
            break; // no more work coming
        else
            std::this_thread::yield(); // wait for new items
    }
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

    const size_t num_buckets = 128;
    const size_t num_threads = std::thread::hardware_concurrency();
    const uint8_t k = 31;
    std::fprintf(stderr, "Total threads: %zu\n", num_threads);

    std::vector<kstring_t> buckets(num_buckets, kstring_t{0, 0, nullptr});
    moodycamel::ConcurrentQueue<size_t> free_buckets;
    moodycamel::ConcurrentQueue<size_t> filled_buckets;
    std::atomic<bool> done(false);

    for (size_t i = 0; i < num_buckets; i++)
        free_buckets.enqueue(i);

    std::vector<std::thread> workers;
    for (size_t t = 0; t < num_threads; t++)
        workers.emplace_back(worker_func,
                             std::ref(buckets),
                             std::ref(filled_buckets),
                             std::ref(free_buckets),
                             std::ref(done),
                             k);

    kseq_t *ks = kseq_init(fp);
    size_t count = 0;
    while (1)
    {
        size_t i;
        if (!free_buckets.try_dequeue(i)) // No free bucket available, wait
        {
            std::this_thread::yield();
            continue;
        }
        ks->seq = buckets[i];
        if (kseq_read(ks) < 0)
            break;
        buckets[i] = ks->seq;
        filled_buckets.enqueue(i);
        ++count;
    }

    done.store(true);
    for (std::thread &w : workers)
        w.join();

    std::fprintf(stderr, "Total reads: %zu\n", count);

    for (kstring_t &bucket : buckets)
        free(bucket.s);
    ks->seq.s = NULL; // pointer has already been freed
    kseq_destroy(ks);
    gzclose(fp);
    return 0;
}
