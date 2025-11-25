#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_SIZE 256
#define LINE_WIDTH 256
#define DEFAULT_CUTOFF 0.99

typedef struct HISTOGRAM_BIN
{
    double count, total, unique;
} bin_t;

static inline int compare_bins(const void *a, const void *b)
{
    const bin_t *A = a;
    const bin_t *B = b;
    return (A->count > B->count) - (A->count < B->count);
}

static int print_usage(int code, const char *progname)
{
    FILE *fp = code ? stderr : stdout;
    fprintf(fp, "Usage: %s [-h] [-c CUTOFF] histogram\n", progname);
    fprintf(fp, "\n");
    fprintf(fp, "Estimate read coverage from a whitespace-separated kmer count histogram file, \n"
                "  reducing the effects of low and high-count kmers by only considering those\n"
                "  with counts in the range sqrt(THRESHOLD) to THRESHOLD, where the total number\n"
                "  of kmers (including duplicates) with count below THRESHOLD is a proportion\n"
                "  CUTOFF of the total.\n");
    fprintf(fp, "\n");
    fprintf(fp, "Positional arguments:\n");
    fprintf(fp, "  histogram            Histogram file path ('-' for stdin)\n");
    fprintf(fp, "\n");
    fprintf(fp, "Options:\n");
    fprintf(fp, "  -h, --help           Show this help message and exit\n");
    fprintf(fp, "  -c, --cutoff CUTOFF  Set cutoff (default %g, 0<cutoff<1)\n", DEFAULT_CUTOFF);
    return code;
}

static int error_usage(const char *progname, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n\n");
    va_end(args);
    return print_usage(1, progname);
}

static int is_flag(const char *arg, const char *flag)
{
    return !strcmp(arg, flag);
}

static int has_flag(const char *arg, const char *flag, const char **value)
{
    size_t flag_len = strlen(flag);
    if (strncmp(arg, flag, flag_len))
        return 0;
    flag_len += arg[flag_len] == '-';
    *value = &arg[flag_len];
    return 1;
}

static int safe_close(FILE *fp)
{
    if (fp == stdin)
        return 0;
    return fclose(fp);
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

static size_t get_stop(const bin_t *bins, size_t l, double total, double cutoff)
{
    double target = cutoff * total;
    size_t left = 0;
    size_t right = l;
    while (left < right)
    {
        size_t mid = left + (right - left) / 2;
        if (bins[mid].total > target)
            right = mid;
        else
            left = mid + 1;
    }
    return left;
}

static size_t get_start(const bin_t *bins, size_t stop)
{
    double threshold = bins[stop - 1].count;
    double min_count = sqrt(threshold);
    size_t left = 0;
    size_t right = stop;
    while (left < right)
    {
        size_t mid = left + (right - left) / 2;
        if (bins[mid].count > min_count)
            right = mid;
        else
            left = mid + 1;
    }
    return left;
}

static double estimate_coverage(const bin_t *bins, size_t l, size_t start, size_t stop)
{
    return (bins[stop - 1].total - bins[start].total) / (bins[stop - 1].unique - bins[start].unique);
}

int main(int argc, char *argv[])
{
    const char *progname = argv[0];
    if (argc == 1)
        return print_usage(1, progname);
    const char *filename = NULL;
    double cutoff = DEFAULT_CUTOFF;
    for (int i = 1; i < argc; i++)
    {
        const char *arg = argv[i];
        const char *value;
        if (is_flag(arg, "-h") || is_flag(arg, "--help"))
            return print_usage(0, progname);
        else if (has_flag(arg, "-c", &value) || has_flag(arg, "--cutoff", &value))
        {
            if (value[0] == '\0' && ++i == argc)
                return error_usage(progname, "Truncated cutoff");
            char *endptr;
            cutoff = strtod(value, &endptr);
            if (*endptr != '\0' || isnan(cutoff) || cutoff <= 0.0 || cutoff >= 1.0)
                return error_usage(progname, "Invalid cutoff: %s", value);
        }
        else if (arg[0] == '-')
            return error_usage(progname, "Unrecognized argument: %s", arg);
        else if (filename)
            return error_usage(progname, "Multiple filenames");
        else
            filename = arg;
    }
    if (!filename)
        return error_usage(progname, "Missing filename");

    FILE *fp = stdin;
    if (strcmp(filename, "-") && !(fp = fopen(filename, "r")))
    {
        fprintf(stderr, "Couldn't open histogram file: %s\n", filename);
        return 1;
    }

    size_t l = 0, m = DEFAULT_SIZE;
    bin_t *bins = malloc(DEFAULT_SIZE * sizeof(bin_t));
    if (!bins)
    {
        safe_close(fp);
        fprintf(stderr, "Failed to allocate initial buffer of size %d\n", DEFAULT_SIZE);
        return 1;
    }

    double total = 0.0;
    double unique = 0.0;
    double prev_count = -1.0;
    char line[LINE_WIDTH];
    for (size_t lineno = 1; fgets(line, LINE_WIDTH, fp); lineno++)
    {
        if (line[0] == '#') // skip comment line
            continue;
        if (!strchr(line, '\n') && !feof(fp))
            return load_error(fp, bins,
                              "Line %zu exceeded buffer size (%d):\n%s\n",
                              lineno, LINE_WIDTH, line);
        double count, freq;
        if (sscanf(line, "%lf %lf", &count, &freq) != 2 ||
            !isfinite(count) || count < 0.0 ||
            !isfinite(freq) || freq < 0.0)
            return load_error(fp, bins, "Line %zu is invalid:\n%s\n", lineno, line);
        if (count <= prev_count)
            return load_error(
                fp, bins, "Histogram count was not strictly increasing at line %zu\n", lineno);
        if (freq == 0.0)
            continue;
        if (l == m)
        {
            if (m > SIZE_MAX / (2 * sizeof(bin_t)))
                return load_error(
                    fp, bins, "Buffer too large (%zu) to re-allocate at line %zu\n", m, lineno);
            m *= 2;
            bin_t *new_bins = realloc(bins, m * sizeof(bin_t));
            if (!new_bins)
                return load_error(
                    fp, bins, "Failed to re-allocate buffer of size %zu at line %zu\n", m, lineno);
            bins = new_bins;
        }
        bins[l].count = count;
        bins[l].total = total;
        bins[l].unique = unique;
        total += count * freq;
        unique += freq;
        prev_count = count;
        ++l;
    }
    safe_close(fp);

    if (l == 0)
    {
        free(bins);
        fprintf(stderr, "Empty histogram file\n");
        return 1;
    }
    if (!isfinite(total))
    {
        free(bins);
        fprintf(stderr, "Overflow while accumulating histogram total\n");
        return 1;
    }

    size_t stop = get_stop(bins, l, total, cutoff);
    if (stop == 0)
    {
        free(bins);
        fprintf(stderr, "Cutoff (%g) was too small\n", cutoff);
        return 1;
    }

    size_t start = get_start(bins, stop);
    if (stop == start)
    {
        free(bins);
        fprintf(stderr, "Insufficient bins to estimate coverage\n");
        return 1;
    }

    printf("%.6lf\n", estimate_coverage(bins, l, start, stop));
    free(bins);
    return 0;
}
