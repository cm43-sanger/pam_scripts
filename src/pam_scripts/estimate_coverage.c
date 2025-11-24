#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

#define LINE_WIDTH 256
#define MAX_COUNT UINT16_MAX
#define DEFAULT_CUTOFF 0.99

static void print_usage(const char *progname)
{
    printf("Usage: %s [options] [histogram]\n", progname);
    printf("\n");
    printf("Estimate coverage from a whitespace-separated histogram file.\n");
    printf("\n");
    printf("Positional arguments:\n");
    printf("  histogram            Histogram file path (defaults to stdin)\n");
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help           Show this help message and exit\n");
    printf("  -c, --cutoff CUTOFF  Set cutoff (default %g)\n", DEFAULT_CUTOFF);
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

static double estimate_coverage(
    const uint64_t histogram[MAX_COUNT + 1], uint64_t total, double cutoff)
{
    if (total == 0)
        return 0.0;
    uint64_t target = ceil(cutoff * total);
    size_t high_count = 1;
    uint64_t selected_total = 0, selected_unique = 0;
    while ((selected_total < target) && (high_count <= MAX_COUNT))
    {
        uint64_t frequency = histogram[high_count];
        selected_total += high_count * frequency;
        selected_unique += frequency;
        ++high_count;
    }
    size_t low_count = ceil(sqrt(high_count));
    for (size_t count = 1; count < low_count; count++)
    {
        uint64_t frequency = histogram[count];
        selected_total -= count * frequency;
        selected_unique -= frequency;
    }
    if (selected_unique == 0)
        return NAN;
    return (double)selected_total / (double)selected_unique;
}

int main(int argc, char *argv[])
{
    double cutoff = DEFAULT_CUTOFF;
    int opt;
    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"cutoff", required_argument, 0, 'c'},
        {0, 0, 0, 0}};
    while ((opt = getopt_long(argc, argv, "hc:", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'h':
            print_usage(argv[0]);
            return 0;
        case 'c':
        {
            char *endptr;
            cutoff = strtod(optarg, &endptr);
            if (*endptr != '\0' || cutoff <= 0.0 || cutoff >= 1.0)
            {
                fprintf(stderr, "Invalid cutoff: %s\n", optarg);
                return 1;
            }
            break;
        }
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    FILE *fp = stdin;
    if (optind < argc)
    {
        const char *filename = argv[optind];
        fp = fopen(filename, "r");
        if (fp == NULL)
        {
            fprintf(stderr, "Couldn't open histogram file '%s'\n", filename);
            return 1;
        }
    }
    else if (isatty(fileno(stdin)))
    {
        fprintf(stderr, "Pipe data to %s\n", argv[0]);
        return 1;
    }

    uint64_t histogram[MAX_COUNT + 1] = {0};
    uint64_t total = 0;
    char line[LINE_WIDTH];
    for (size_t lineno = 1; fgets(line, LINE_WIDTH, fp); lineno++)
    {
        if (line[0] == '#') // skip comment line
            continue;
        if (strchr(line, '\n') == NULL)
            return close_file_error(fp,
                                    "Line %zu exceeded buffer size (%d):\n%s\n",
                                    lineno, LINE_WIDTH, line);
        uint64_t count, frequency;
        if (sscanf(line, "%llu %llu", &count, &frequency) != 2)
            return close_file_error(fp, "Line %zu is invalid:\n%s\n", lineno, line);
        count = count > MAX_COUNT ? MAX_COUNT : count;
        if (histogram[count])
            return close_file_error(
                fp,
                "Line %zu has tried to reset frequency of count %llu (previously %llu):\n%s\n",
                lineno, count, histogram[count], line);
        histogram[count] = frequency;
        total += count * frequency;
    }
    safe_close(fp);

    printf("%.6f\n", estimate_coverage(histogram, total, cutoff));
    return 0;
}
