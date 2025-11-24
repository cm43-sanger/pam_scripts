#ifndef SKETCH_DB_H
#define SKETCH_DB_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define SKETCH_M_MIN 256

#define SKETCH_DB_SUCCESS 0
#define SKETCH_DB_EOF 1
#define SKETCH_DB_INVALID_HEADER 2
#define SKETCH_DB_INVALID_MODE 3
#define SKETCH_DB_FAILED_OPEN -1
#define SKETCH_DB_FAILED_CLOSE -2
#define SKETCH_DB_FAILED_IO -3
#define SKETCH_DB_FAILED_ALLOC -4

typedef struct SKETCH
{
    uint32_t l, m;
    uint32_t *a;
} sketch_t;

#define SKETCH_INIT {0, 0, NULL}

static inline int sketch_free(sketch_t *sketch_ptr)
{
    free(sketch_ptr->a);
    sketch_ptr->l = 0, sketch_ptr->m = 0, sketch_ptr->a = NULL;
}

static inline int sketch_resize(sketch_t *sketch_ptr, uint32_t size)
{
    if (size > sketch_ptr->m)
    {
        sketch_ptr->m
    }
    return 0;
}

// static inline int sketch_append(sketch_t *sketch_ptr, uint32_t val)
// {
//     if
// }

typedef struct SKETCHES_DATABASE
{
    gzFile f;
    int w;
    uint8_t k;
    char method[256];
} sketch_db_t;

static inline int load_sketch_db(sketch_db_t *sketch_db_ptr, const char *filename)
{
    sketch_db_ptr->w = 0;
    if ((sketch_db_ptr->f = gzopen(filename, "rb")) == NULL)
        return SKETCH_DB_FAILED_OPEN;
    if (gzread(sketch_db_ptr->f, &sketch_db_ptr->k, sizeof(uint8_t)) != sizeof(uint8_t) ||
        gzread(sketch_db_ptr->f,
               sketch_db_ptr->method,
               sizeof(sketch_db_ptr->method)) != sizeof(sketch_db_ptr->method))
    {
        gzclose(sketch_db_ptr->f);
        return SKETCH_DB_FAILED_IO;
    }
    size_t method_length = strnlen(sketch_db_ptr->method, 256);
    if (method_length == 256)
    {
        gzclose(sketch_db_ptr->f);
        return SKETCH_DB_INVALID_HEADER;
    }
    return SKETCH_DB_SUCCESS;
}

static inline int store_sketch_db(sketch_db_t *sketch_db_ptr, const char *filename,
                                  uint8_t k, const char *method, int compression)
{
    if (compression)
    {
    }
    size_t method_length = strnlen(method, 256);
    if (method_length == 256)
        return SKETCH_DB_INVALID_HEADER;
    sketch_db_ptr->w = 1;
    sketch_db_ptr->k = k;
    memcpy(sketch_db_ptr->method, method, method_length);
    memset(&sketch_db_ptr->method[method_length], '\0', (256 - method_length));
    if ((sketch_db_ptr->f = gzopen(filename, "wb")) == NULL)
        return SKETCH_DB_FAILED_OPEN;
    if (gzwrite(sketch_db_ptr->f, &sketch_db_ptr->k, sizeof(uint8_t)) != sizeof(uint8_t) ||
        gzwrite(sketch_db_ptr->f,
                sketch_db_ptr->method,
                sizeof(sketch_db_ptr->method)) != sizeof(sketch_db_ptr->method))
    {
        gzclose(sketch_db_ptr->f);
        return SKETCH_DB_FAILED_IO;
    }
    return SKETCH_DB_SUCCESS;
}

static inline int close_sketch_db(sketch_db_t *sketch_db_ptr)
{
    if (!sketch_db_ptr->f)
        return SKETCH_DB_SUCCESS;
    int zerr = gzclose(sketch_db_ptr->f);
    sketch_db_ptr->f = NULL;
    return (zerr == Z_OK) ? SKETCH_DB_SUCCESS : SKETCH_DB_FAILED_CLOSE;
}

static inline int read_sketch(sketch_t *sketch_ptr, sketch_db_t *sketch_db_ptr)
{
    if (sketch_db_ptr->f == NULL || sketch_db_ptr->w)
        return SKETCH_DB_INVALID_MODE;
    int bytes_read = gzread(sketch_db_ptr->f, &sketch_ptr->l, sizeof(uint32_t));
    if (bytes_read == 0 && gzeof(sketch_db_ptr->f))
        return SKETCH_DB_EOF;
    if (bytes_read != sizeof(uint32_t))
        return SKETCH_DB_FAILED_IO;
    sketch_ptr->l = ntohl(sketch_ptr->l);
    if (sketch_ptr->m < sketch_ptr->l)
    {
        size_t m_new = 2 * sketch_ptr->m;
        m_new = m_new > SKETCH_M_MIN ? m_new : SKETCH_M_MIN;
        if (m_new > UINT32_MAX ||
            (sketch_ptr->a = (uint32_t *)realloc(sketch_ptr->a, m_new * sizeof(uint32_t))) == NULL)
            return SKETCH_DB_FAILED_ALLOC;
        sketch_ptr->m = m_new;
    }
    size_t num_bytes = sketch_ptr->l * sizeof(uint32_t);
    if (gzread(sketch_db_ptr->f, sketch_ptr->a, num_bytes) != num_bytes)
        return SKETCH_DB_FAILED_IO;
    for (uint32_t j = 0; j < sketch_ptr->l; j++)
        sketch_ptr->a[j] = ntohl(sketch_ptr->a[j]);
    return SKETCH_DB_SUCCESS;
}

static inline int write_sketch(sketch_t *sketch_ptr, sketch_db_t *sketch_db_ptr)
{
    if (sketch_db_ptr->f == NULL || !sketch_db_ptr->w)
        return SKETCH_DB_INVALID_MODE;
    uint32_t l_n = htonl(sketch_ptr->l);
    if (gzwrite(sketch_db_ptr->f, &l_n, sizeof(uint32_t)) != sizeof(uint32_t))
        return SKETCH_DB_FAILED_IO;
    for (uint32_t start = 0; start < sketch_ptr->l; start += 256)
    {
        uint32_t l = sketch_ptr->l - start;
        if (l > 256)
            l = 256;
        uint32_t buf[256];
        for (uint32_t i = 0; i < l; i++)
            buf[i] = htonl(sketch_ptr->a[start + i]);
        size_t num_bytes = l * sizeof(uint32_t);
        if (gzwrite(sketch_db_ptr->f, buf, num_bytes) != num_bytes)
            return SKETCH_DB_FAILED_IO;
    }
    return SKETCH_DB_SUCCESS;
}

#undef SKETCH_M_MIN

#endif
