#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_SIZE 256
#define LINE_WIDTH 256
#define DEFAULT_CUTOFF 0.99

typedef struct CUMULATIVE_POINT
{
    double count, total, unique;
} point_t;

static void print_usage(FILE *fp, const char *progname)
{
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
}

static int error_usage(const char *progname, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n\n");
    va_end(args);
    print_usage(stderr, progname);
    return 1;
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
    flag_len += arg[flag_len] == '=';
    *value = &arg[flag_len];
    return 1;
}

static int safe_close(FILE *fp)
{
    if (fp == stdin)
        return 0;
    return fclose(fp);
}

static int load_error(FILE *fp, point_t *points, const char *fmt, ...)
{
    safe_close(fp);
    free(points);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    return 1;
}

static int parse_line(const char *line, double *count, double *freq)
{
    long long count_lld, freq_lld;
    int consumed;
    if (sscanf(line, "%lld %lld %n", &count_lld, &freq_lld, &consumed) != 2 ||
        line[consumed] != '\0' ||
        count_lld < 0 || freq_lld < 0)
        return 1;
    *count = count_lld;
    *freq = freq_lld;
    return 0;
}

static double estimate_threshold(const point_t *points, size_t l, double total, double cutoff)
{
    double target = cutoff * total;
    size_t left = 0;
    size_t right = l;
    while (left < right)
    {
        size_t mid = left + (right - left) / 2;
        if (points[mid].total > target)
            right = mid;
        else
            left = mid + 1;
    }
    if (left == 0)
        return 0.0;
    double i = (target - points[left - 1].total) / (points[left].total - points[left - 1].total);
    return points[left - 1].count + i * (points[left].count - points[left - 1].count);
}

static int interpolate(
    const point_t *points, size_t l, double count, double *total, double *unique)
{
    if (count < points[0].count || count > points[l - 1].count)
        return 1;
    size_t left = 1;
    size_t right = l;
    while (left < right)
    {
        size_t mid = left + (right - left) / 2;
        if (points[mid].count > count)
            right = mid;
        else
            left = mid + 1;
    }
    double i = (count - points[left - 1].count) / (points[left].count - points[left - 1].count);
    *total = points[left - 1].total + i * (points[left].total - points[left - 1].total);
    *unique = points[left - 1].unique + i * (points[left].unique - points[left - 1].unique);
    return 0;
}

int main(int argc, char *argv[])
{
    const char *progname = argv[0];
    const char *path = NULL;
    double cutoff = DEFAULT_CUTOFF;
    for (int i = 1; i < argc; i++)
    {
        const char *arg = argv[i];
        const char *value;
        if (is_flag(arg, "-h") || is_flag(arg, "--help"))
        {
            print_usage(stdout, progname);
            return 0;
        }
        else if (has_flag(arg, "-c", &value) || has_flag(arg, "--cutoff", &value))
        {
            if (value[0] == '\0')
            {
                if (++i == argc)
                    return error_usage(progname, "Truncated cutoff");
                value = argv[i];
            }
            char *endptr;
            cutoff = strtod(value, &endptr);
            if (*endptr != '\0' || isnan(cutoff) || cutoff <= 0.0 || cutoff >= 1.0)
                return error_usage(progname, "Invalid cutoff: %s", value);
        }
        else if (arg[0] == '-' && strlen(arg) != 1)
            return error_usage(progname, "Unrecognized argument: %s", arg);
        else if (path)
            return error_usage(progname, "Multiple histogram file paths");
        else
            path = arg;
    }
    if (!path)
        return error_usage(progname, "Missing histogram file path");

    FILE *fp = stdin;
    if (strcmp(path, "-") && !(fp = fopen(path, "r")))
    {
        fprintf(stderr, "Couldn't open histogram file path: %s\n", path);
        return 1;
    }

    size_t l = 0, m = DEFAULT_SIZE;
    point_t *points = malloc(DEFAULT_SIZE * sizeof(point_t));
    if (!points)
    {
        safe_close(fp);
        fprintf(stderr, "Failed to allocate initial buffer of size %d\n", DEFAULT_SIZE);
        return 1;
    }

    double total = 0.0;
    double unique = 0.0;
    {
        double prev_count = -1.0;
        char line[LINE_WIDTH];
        for (size_t lineno = 1; fgets(line, LINE_WIDTH, fp); lineno++)
        {
            if (line[0] == '#') // skip comment line
                continue;
            if (!strchr(line, '\n') && !feof(fp))
                return load_error(fp, points,
                                  "Line %zu exceeded buffer size (%d):\n%s\n",
                                  lineno, LINE_WIDTH, line);
            double count, freq;
            if (parse_line(line, &count, &freq))
                return load_error(fp, points, "Line %zu is invalid:\n%s\n", lineno, line);
            if (count <= prev_count)
                return load_error(
                    fp, points, "Histogram count was not strictly increasing at line %zu\n", lineno);
            prev_count = count;
            if (count == 0.0 || freq == 0.0)
                continue;
            if (l == m)
            {
                if (m > SIZE_MAX / (2 * sizeof(point_t)))
                    return load_error(
                        fp, points, "Buffer of size %zu too large to re-allocate at line %zu\n", m, lineno);
                m *= 2;
                point_t *new_points = realloc(points, m * sizeof(point_t));
                if (!new_points)
                    return load_error(
                        fp, points, "Failed to re-allocate buffer of size %zu at line %zu\n", m, lineno);
                points = new_points;
            }
            total += count * freq;
            unique += freq;
            if (!isfinite(total) || !isfinite(unique))
                return load_error(
                    fp, points, "Overflow while accumulating histogram total at line %zu\n", lineno);
            points[l].count = count;
            points[l].total = total;
            points[l].unique = unique;
            ++l;
        }
    }
    safe_close(fp);

    if (l < 2)
    {
        free(points);
        fprintf(stderr, "Insufficient non-zero entries in histogram file\n");
        return 1;
    }

    double threshold = estimate_threshold(points, l, total, cutoff);
    if (threshold == 0.0)
    {
        fprintf(stderr,
                "THRESHOLD (%.3lf) is less than smallest count (%.0lf):\n"
                "no filtering\n",
                threshold, points[0].count);
        printf("%.6lf\n", total / unique);
        free(points);
        return 0;
    }

    double low_total = 0.0, low_unique = 0.0;
    if (interpolate(points, l, sqrt(threshold), &low_total, &low_unique))
        fprintf(stderr,
                "sqrt(THRESHOLD) (%.3lf) is less than smallest count (%.0lf):\n"
                "no low-count filtering\n",
                sqrt(threshold), points[0].count);

    double high_total, high_unique;
    if (interpolate(points, l, threshold, &high_total, &high_unique) ||
        high_unique == low_unique)
    {
        free(points);
        fprintf(stderr, "Insufficient points to estimate coverage\n");
        return 1;
    }

    free(points);

    double coverage = (high_total - low_total) / (high_unique - low_unique);
    printf("%.6lf\n", coverage);

    fprintf(stderr, "Cutoff: %g\n", cutoff);
    fprintf(stderr, "Threshold: %.3lf\n", threshold);
    fprintf(stderr, "Total kmers: %.0lf\n", total);
    fprintf(stderr, "Unique kmers: %.0lf\n", unique);
    fprintf(stderr, "Total kmers below sqrt(threshold): %.0lf\n", low_total);
    fprintf(stderr, "Unique kmers below sqrt(threshold): %.0lf\n", low_unique);
    fprintf(stderr, "Total kmers above threshold: %.0lf\n", total - high_total);
    fprintf(stderr, "Unique kmers above threshold: %.0lf\n", unique - high_unique);
    fprintf(stderr, "Coverage: %.3lf\n", coverage);

    return 0;
}
