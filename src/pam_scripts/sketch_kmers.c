#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <hdf5.h>
// #include <xxhash.h>
#define XXH_INLINE_ALL
#include "xxhash.h"

#define LINE_WIDTH 512
#define DEFAULT_SIZE 1024
#define DEFAULT_K 21
#define DEFAULT_SCALE 1000
#define DEFAULT_SEED 42
#define BAD_BASE INT8_MAX

// gcc -O3 $(pkg-config --cflags hdf5) src/pam_scripts/sketch_kmers.c -osrc/pam_scripts/sketch_kmers
// gcc -O3 $(pkg-config --cflags hdf5) src/pam_scripts/sketch_kmers.c src/pam_scripts/xxhash.c -osrc/pam_scripts/sketch_kmers

static void print_usage(const char *progname)
{
    printf("Usage: %s [options] [dump]\n", progname);
    printf("\n");
    printf("Estimate coverage from a whitespace-separated histogram file.\n");
    printf("\n");
    printf("Positional arguments:\n");
    printf("  dump                 Kmer dump path (defaults to stdin)\n");
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help           Show this help message and exit\n");
    printf("  -o, --output_sketch  Output sketch location ");
    printf("                         (.h5 extension appended if not present)\n");
    printf("  -k, --kmer_length K  Set kmer length (default %d)\n", DEFAULT_K);
}

static void safe_close(FILE *fp)
{
    if (fp && (fp != stdin))
        fclose(fp);
}

static int close_file_error(FILE *fp, const char *fmt, ...)
{
    safe_close(fp);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    return 1;
}

char process_base(char base)
{
    switch (base)
    {
    case 'A':
    case 'a':
        return 'A';
    case 'C':
    case 'c':
        return 'C';
    case 'G':
    case 'g':
        return 'G';
    case 'T':
    case 't':
        return 'T';
    default:
        return BAD_BASE;
    }
}

int main(int argc, char *argv[])
{
    char *sketch = NULL;
    uint32_t k = DEFAULT_K, scale = DEFAULT_SCALE, seed = DEFAULT_SEED;
    int opt;
    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"output_sketch", required_argument, 0, 'o'},
        {"kmer_length", required_argument, 0, 'k'},
        {"scale", required_argument, 0, 's'},
        {"seed", required_argument, 0, 'd'},
        {0, 0, 0, 0}};
    while ((opt = getopt_long(argc, argv, "hoksd:", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'h':
            print_usage(argv[0]);
            return 0;
        case 'o':
            sketch = optarg;
            break;
        case 'k':
        {
            char *endptr;
            k = strtoul(optarg, &endptr, 10);
            if (*endptr != '\0' || k == 0 || k > 255)
            {
                fprintf(stderr, "Invalid kmer length: %u\n", k);
                return 1;
            }
            break;
        }
        case 's':
        {
            char *endptr;
            scale = strtoul(optarg, &endptr, 10);
            if (*endptr != '\0' || scale == 0)
            {
                fprintf(stderr, "Invalid scale: %u\n", scale);
                return 1;
            }
            break;
        }
        case 'd':
        {
            char *endptr;
            seed = strtoul(optarg, &endptr, 10);
            if (*endptr != '\0' || scale == 0)
            {
                fprintf(stderr, "Invalid seed: %u\n", seed);
                return 1;
            }
            break;
        }
        default:
            print_usage(argv[0]);
            return 1;
        }
    }
    if (sketch == NULL)
    {
        fprintf(stderr, "Output sketch path not provided\n");
        return 1;
    }

    FILE *fp = stdin;
    if (optind < argc)
    {
        const char *filename = argv[optind];
        fp = fopen(filename, "r");
        if (fp == NULL)
        {
            fprintf(stderr, "Couldn't open kmer file '%s'\n", filename);
            return 1;
        }
    }
    else if (isatty(fileno(stdin)))
    {
        fprintf(stderr, "Pipe data to %s\n", argv[0]);
        return 1;
    }

    size_t cutoff = UINT64_MAX / scale;
    size_t length = 0, size = DEFAULT_SIZE;
    uint32_t *hashes = malloc(DEFAULT_SIZE * sizeof(uint32_t));
    char line[LINE_WIDTH];
    for (size_t lineno = 1; fgets(line, LINE_WIDTH, fp); lineno++)
    {
        if (strchr(line, '\n') == NULL)
            return close_file_error(fp,
                                    "Line %zu exceeded buffer size (%d):\n%s\n",
                                    lineno, LINE_WIDTH, line);
        char kmer[256 + 1];
        uint64_t count;
        if (sscanf(line, "%256s %llu", kmer, &count) != 2 || strlen(kmer) != k)
            return close_file_error(fp, "Invalid line %zu:\n%s\n", lineno, line);
        for (uint32_t i = 0; i < k; i++)
        {
            char base = process_base(kmer[i]);
            if (base == BAD_BASE)
                return close_file_error(
                    fp,
                    "Invalid base (%c) at position %u in line %zu:\n%s\n",
                    kmer[i], i, lineno, line);
            kmer[i] = base;
        }
        uint64_t hash64 = XXH64(kmer, k, DEFAULT_SEED);
        if (hash64 > cutoff)
            continue;
        if (length == size)
        {
            size *= 2;
            if ((hashes = realloc(hashes, size * sizeof(uint32_t))) == NULL)
                return close_file_error(
                    fp, "Unable to allocate hashes array of size %zu\n", size);
        }
        hashes[length++] = hash64;
    }
    safe_close(fp);

    return 0;
}
