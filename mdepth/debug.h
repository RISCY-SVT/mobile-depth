#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>

#define DEBUG_PRINT(fmt, ...) \
    do { \
        printf("[DEBUG] [%s:%s:%d] " fmt "\n", __FILE__, __func__, __LINE__, ##__VA_ARGS__); \
    } while (0)

#endif // DEBUG_H
