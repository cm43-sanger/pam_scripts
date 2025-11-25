// #include <stdarg.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <math.h>
// #include <getopt.h>

// #define LINE_WIDTH 256
// #define MAX_COUNT UINT16_MAX
// #define DEFAULT_CUTOFF 0.99

// static void print_usage(const char *progname)
// {
//     printf("Usage: %s [options] [histogram]\n", progname);
//     printf("\n");
//     printf("Estimate coverage from a whitespace-separated histogram file.\n");
//     printf("\n");
//     printf("Positional arguments:\n");
//     printf("  histogram            Histogram file path (defaults to stdin)\n");
//     printf("\n");
//     printf("Options:\n");
//     printf("  -h, --help           Show this help message and exit\n");
//     printf("  -c, --cutoff CUTOFF  Set cutoff (default %g)\n", DEFAULT_CUTOFF);
// }

// static void safe_close(FILE *fp)
// {
//     if (fp && (fp != stdin))
//         fclose(fp);
// }

// static int close_file_error(FILE *fp, const char *fmt, ...)
// {
//     safe_close(fp);
//     va_list args;
//     va_start(args, fmt);
//     vfprintf(stderr, fmt, args);
//     va_end(args);
//     return 1;
// }

// static double estimate_coverage(
//     const uint64_t histogram[MAX_COUNT + 1], uint64_t total, double cutoff)
// {
//     if (total == 0)
//         return 0.0;
//     uint64_t target = ceil(cutoff * total);
//     size_t high_count = 1;
//     uint64_t selected_total = 0, selected_unique = 0;
//     while ((selected_total < target) && (high_count <= MAX_COUNT))
//     {
//         uint64_t frequency = histogram[high_count];
//         selected_total += high_count * frequency;
//         selected_unique += frequency;
//         ++high_count;
//     }
//     size_t low_count = ceil(sqrt(high_count));
//     for (size_t count = 1; count < low_count; count++)
//     {
//         uint64_t frequency = histogram[count];
//         selected_total -= count * frequency;
//         selected_unique -= frequency;
//     }
//     if (selected_unique == 0)
//         return NAN;
//     return (double)selected_total / (double)selected_unique;
// }

// int main(int argc, char *argv[])
// {
//     double cutoff = DEFAULT_CUTOFF;
//     int opt;
//     static struct option long_options[] = {
//         {"help", no_argument, 0, 'h'},
//         {"cutoff", required_argument, 0, 'c'},
//         {0, 0, 0, 0}};
//     while ((opt = getopt_long(argc, argv, "hc:", long_options, NULL)) != -1)
//     {
//         switch (opt)
//         {
//         case 'h':
//             print_usage(argv[0]);
//             return 0;
//         case 'c':
//         {
//             char *endptr;
//             cutoff = strtod(optarg, &endptr);
//             if (*endptr != '\0' || cutoff <= 0.0 || cutoff >= 1.0)
//             {
//                 fprintf(stderr, "Invalid cutoff: %s\n", optarg);
//                 return 1;
//             }
//             break;
//         }
//         default:
//             print_usage(argv[0]);
//             return 1;
//         }
//     }

//     FILE *fp = stdin;
//     if (optind < argc)
//     {
//         const char *filename = argv[optind];
//         fp = fopen(filename, "r");
//         if (fp == NULL)
//         {
//             fprintf(stderr, "Couldn't open histogram file '%s'\n", filename);
//             return 1;
//         }
//     }
//     else if (isatty(fileno(stdin)))
//     {
//         fprintf(stderr, "Pipe data to %s\n", argv[0]);
//         return 1;
//     }

//     uint64_t histogram[MAX_COUNT + 1] = {0};
//     uint64_t total = 0;
//     char line[LINE_WIDTH];
//     for (size_t lineno = 1; fgets(line, LINE_WIDTH, fp); lineno++)
//     {
//         if (line[0] == '#') // skip comment line
//             continue;
//         if (strchr(line, '\n') == NULL)
//             return close_file_error(fp, bins,
//                                     "Line %zu exceeded buffer size (%d):\n%s\n",
//                                     lineno, LINE_WIDTH, line);
//         uint64_t count, frequency;
//         if (sscanf(line, "%llu %llu", &count, &frequency) != 2)
//             return close_file_error(fp, bins, "Line %zu is invalid:\n%s\n", lineno, line);
//         count = count > MAX_COUNT ? MAX_COUNT : count;
//         if (histogram[count])
//             return close_file_error(
//                 fp,
//                 "Line %zu has tried to reset frequency of count %llu (previously %llu):\n%s\n",
//                 lineno, count, histogram[count], line);
//         histogram[count] = frequency;
//         total += count * frequency;
//     }
//     safe_close(fp);

//     printf("%.6f\n", estimate_coverage(histogram, total, cutoff));
//     return 0;
// }

// gcc -O3 src/pam_scripts/estimate_coverage.c -osrc/pam_scripts/estimate_coverage

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEFAULT_SIZE 256
#define LINE_WIDTH 256
#define DEFAULT_CUTOFF 0.99

typedef struct HISTOGRAM_BIN
{
    double count, freq;
} bin_t;

int compare_bins(const void *a, const void *b)
{
    const bin_t *A = a;
    const bin_t *B = b;
    return (A->count > B->count) - (A->count < B->count);
}

static void safe_close(FILE *fp)
{
    if (fp && (fp != stdin))
        fclose(fp);
}

static int load_error(FILE *fp, bin_t *bins, const char *fmt, ...)
{
    safe_close(fp);
    free(bins);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    return 1;
}

static size_t get_stop(const bin_t *bins, double total, double cutoff)
{
    double target = cutoff * total;
    double cum = 0.0;
    size_t stop = 0;
    while ((cum += bins[stop].count * bins[stop].freq) < total)
        ++stop;
    return stop;
}

static size_t get_start(const bin_t *bins, size_t l, double low)
{
    size_t left = 0, right = l;
    while (left < right)
    {
        size_t mid = left + (right - left) / 2;
        if (bins[mid].count > low)
            right = mid;
        else
            left = mid + 1;
    }
    return left;
}

static double estimate_coverage(const bin_t *bins, size_t l, size_t start, size_t stop)
{
    double total = 0.0, unique = 0.0;
    for (size_t i = start; i < stop; i++)
    {
        total += bins[i].count * bins[i].freq;
        unique += bins[i].freq;
    }
    return total / unique;
}

int main(int argc, char *argv[])
{
    char *filename = NULL;
    double cutoff = DEFAULT_CUTOFF;
    for (int i = 1; i < argc; i++)
    {
        if (!strncmp(argv[i], "-h", 2))
        {
            printf("Usage: %s [-h] [-c CUTOFF] [histogram]\n", argv[0]);
            return 0;
        }
        else if (!strncmp(argv[i], "-c", 2))
        {
            char *value = NULL;
            if (argv[i][2] != '\0')
                value = &argv[i][2];
            else if (++i == argc)
            {
                fprintf(stderr, "Cutoff flag ('-c') given without value\n");
                return 1;
            }
            else
                value = argv[i];
            char *endptr;
            cutoff = strtod(value, &endptr);
            if (*endptr != '\0' || cutoff <= 0.0 || cutoff >= 1.0)
            {
                fprintf(stderr, "Invalid cutoff: %s\n", value);
                return 1;
            }
        }
        else
            filename = argv[i];
    }

    FILE *fp = stdin;
    if (filename)
    {
        fp = fopen(filename, "r");
        if (fp == NULL)
        {
            fprintf(stderr, "Couldn't open histogram file '%s'\n", filename);
            return 1;
        }
    }

    size_t l = 0, m = DEFAULT_SIZE;
    bin_t *bins = malloc(DEFAULT_SIZE * sizeof(bin_t));
    if (bins == NULL)
    {
        safe_close(fp);
        fprintf(stderr, "Failed to allocate initial buffer of size %d\n", DEFAULT_SIZE);
        return 1;
    }

    int ordered = 1;
    double total = 0.0;
    double prev_count = -1.0;
    char line[LINE_WIDTH];
    for (size_t lineno = 1; fgets(line, LINE_WIDTH, fp); lineno++)
    {
        if (line[0] == '#') // skip comment line
            continue;
        if (strchr(line, '\n') == NULL && !feof(fp))
            return load_error(fp, bins,
                              "Line %zu exceeded buffer size (%d):\n%s\n",
                              lineno, LINE_WIDTH, line);
        double count, freq;
        if (sscanf(line, "%lf %lf", &count, &freq) != 2 || count < 0.0 || freq < 0.0)
            return load_error(fp, bins, "Line %zu is invalid:\n%s\n", lineno, line);
        if (freq == 0.0)
            continue;
        if (l == m)
        {
            if (m > SIZE_MAX / (2 * sizeof(bin_t)))
                return load_error(
                    fp, bins, "Buffer too large (%zu) to re-allocate at line %zu\n", m, lineno);
            m *= 2;
            bin_t *new_bins = realloc(bins, m * sizeof(bin_t));
            if (new_bins == NULL)
                return load_error(
                    fp, bins, "Failed to re-allocate buffer of size %zu at line %zu\n", m, lineno);
            bins = new_bins;
        }
        bins[l].count = count, bins[l].freq = freq;
        ++l;
        ordered &= count > prev_count;
        total += count * freq;
        prev_count = count;
    }
    safe_close(fp);

    if (l == 0)
    {
        free(bins);
        fprintf(stderr, "Empty histogram file\n");
        return 1;
    }

    if (!ordered)
    {
        fprintf(stderr, "Histogram file was not sorted\n");
        qsort(bins, l, sizeof(bin_t), compare_bins);
    }

    size_t stop = get_stop(bins, total, cutoff);
    double high = bins[stop].count;
    double low = sqrt(high);
    size_t start = get_start(bins, l, low);
    if (stop == start)
    {
        free(bins);
        fprintf(stderr, "Insufficient bins to estimate coverage\n");
        return 1;
    }
    printf("%.6f\n", estimate_coverage(bins, l, start, stop));

    free(bins);
    return 0;
}
