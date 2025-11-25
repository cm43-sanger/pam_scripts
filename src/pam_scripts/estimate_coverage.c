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

static int compare_bins(const void *a, const void *b)
{
    const bin_t *A = a;
    const bin_t *B = b;
    return (A->count > B->count) - (A->count < B->count);
}

static int is_flag(const char *arg, const char *flag, const char **value)
{
    size_t flag_len = strlen(flag);
    if (strncmp(arg, flag, flag_len))
        return 0;
    *value = &arg[flag_len];
    return 1;
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

static size_t get_stop(const bin_t *bins, size_t l, double total, double cutoff)
{
    double target = cutoff * total;
    size_t i = 0;
    double cum = 0.0;
    while (i < l && cum < target)
    {
        cum += bins[i].count * bins[i].freq;
        ++i;
    }
    return i;
}

static size_t get_start(const bin_t *bins, size_t stop)
{
    double high = bins[stop - 1].count;
    double low = sqrt(high);
    size_t left = 0, right = stop;
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
    const char *filename = NULL;
    double cutoff = DEFAULT_CUTOFF;
    for (int i = 1; i < argc; i++)
    {
        const char *arg = argv[i];
        const char *value;
        if (is_flag(arg, "-h", &value) || is_flag(arg, "--help", &value))
        {
            if (value[0] != '\0')
            {
                fprintf(stderr, "Unrecognized argument: %s\n", arg);
                return 1;
            }
            printf("Usage: %s [-h] [-c CUTOFF] histogram\n", argv[0]);
            printf("\n");
            printf("Estimate coverage from a whitespace-separated histogram file.\n");
            printf("\n");
            printf("Positional arguments:\n");
            printf("  histogram            Histogram file path ('-' for stdin)\n");
            printf("\n");
            printf("Options:\n");
            printf("  -h, --help           Show this help message and exit\n");
            printf("  -c, --cutoff CUTOFF  Set cutoff (default %g, 0<cutoff<1)\n", DEFAULT_CUTOFF);
            return 0;
        }
        else if (is_flag(arg, "-c", &value) || is_flag(arg, "--cutoff", &value))
        {
            if (value[0] == '=')
                ++value;
            if (value[0] == '\0' && ++i == argc)
            {
                fprintf(stderr, "Truncated cutoff\n");
                return 1;
            }
            char *endptr;
            cutoff = strtod(value, &endptr);
            if (*endptr != '\0' || isnan(cutoff) || cutoff <= 0.0 || cutoff >= 1.0)
            {
                fprintf(stderr, "Invalid cutoff: %s\n", value);
                return 1;
            }
        }
        else if (arg[0] == '-')
        {
            fprintf(stderr, "Unrecognized argument: %s\n", arg);
            return 1;
        }
        else if (filename)
        {
            fprintf(stderr, "Multiple filenames\n");
            return 1;
        }
        else
            filename = arg;
    }
    if (!filename)
    {
        fprintf(stderr, "Usage: %s [-h] [-c CUTOFF] histogram\n", argv[0]);
        return 1;
    }

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

    int ordered = 1;
    double total = 0.0;
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
    if (!isfinite(total))
    {
        free(bins);
        fprintf(stderr, "Overflow while accumulating histogram total\n");
        return 1;
    }

    if (!ordered)
    {
        fprintf(stderr, "Histogram file was not sorted\n");
        qsort(bins, l, sizeof(bin_t), compare_bins);
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
