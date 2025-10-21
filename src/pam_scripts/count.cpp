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

static inline void update_forward_reverse_no_mask(uint64_t &forward, uint64_t &reverse,
                                                  uint64_t base, uint8_t k)
{
    forward = (forward << 2) | base;
    reverse = (reverse >> 2) | ((0b11ULL ^ base) << (2 * k - 2));
}

static inline uint64_t canonical_kmer(uint64_t forward, uint64_t reverse)
{
    return (forward > reverse) ? reverse : forward;
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
    uint64_t mask = (1ULL << (2 * k)) - 1;
    size_t next_start = 0;
    size_t first_invalid_start = l - k + 1;
    do
    {
        while ((next_start < first_invalid_start) && (s[next_start] == UINT8_MAX))
            ++next_start;
        if (next_start == first_invalid_start)
            return kmers;
        size_t stop = next_start;
        while ((stop < l) && (s[stop] != UINT8_MAX))
            ++stop;
        size_t start = next_start;
        next_start = stop + 1; // skip non-base character if unfinished (doesn't matter if finished)
        if ((stop - start) < k)
            continue;
        size_t i = start;
        uint64_t forward = 0, reverse = 0;
        for (; i < start + k; i++)
            update_forward_reverse_no_mask(forward, reverse, s[i], k);
        kmers.push_back(canonical_kmer(forward, reverse));
        for (; i < stop; i++)
        {
            update_forward_reverse_no_mask(forward, reverse, s[i], k);
            forward &= mask;
            kmers.push_back(canonical_kmer(forward, reverse));
        }
    } while (next_start < first_invalid_start);
    return kmers;
}

static inline size_t get_next_bucket(moodycamel::ConcurrentQueue<size_t> &buckets)
{
    size_t i;
    while (!buckets.try_dequeue(i)) // try getting filled bucket
        std::this_thread::yield();  // otherwise wait for new items
    return i;
}

static inline void worker_func(
    std::vector<kstring_t> &buckets,
    moodycamel::ConcurrentQueue<size_t> &filled_buckets,
    moodycamel::ConcurrentQueue<size_t> &free_buckets,
    uint8_t k)
{
    while (true)
    {
        size_t i = get_next_bucket(filled_buckets);                              // get filled bucket
        if (buckets[i].l == 0)                                                   // if received done signal
            return;                                                              // then exit
        std::vector<uint64_t> kmers = extract_kmers_destructive(&buckets[i], k); // otherwise get kmers
        free_buckets.enqueue(i);                                                 // then return empty bucket
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

    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t num_buckets = 6 * num_threads;
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
                             k);

    kseq_t *ks = kseq_init(fp);
    size_t count = 0;
    size_t i;
    while (true)
    {
        i = get_next_bucket(free_buckets); // get free bucket
        ks->seq = buckets[i];
        int l;
        while ((l = kseq_read(ks)) == 0) // skip empty reads
            ;
        buckets[i] = ks->seq;
        if (l < 0) // finished
            break;
        filled_buckets.enqueue(i);
        ++count;
    }

    // cleanup workers
    buckets[i].l = 0;          // send stop signal to first thread
    filled_buckets.enqueue(i); //
    for (size_t t = 1; t < num_threads; t++)
    {
        i = get_next_bucket(free_buckets); // get free bucket
        buckets[i].l = 0;                  // send stop signal to other threads
        filled_buckets.enqueue(i);         //
    }
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
