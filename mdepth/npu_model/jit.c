/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/* auto generate by HHB_VERSION "2.11.4" */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <libgen.h>
#include <unistd.h>
#include "io.h"
#include "shl_ref.h"

void *csinn_(char *params);

void *create_graph(char *params_path) {
    int binary_size;
    char *params = get_binary_from_file(params_path, &binary_size);
    if (params == NULL) {
        return NULL;
    }

    char *suffix = params_path + (strlen(params_path) - 7);
    if (strcmp(suffix, ".params") == 0) {
        // create general graph
        return csinn_(params);
    }

    suffix = params_path + (strlen(params_path) - 3);
    if (strcmp(suffix, ".bm") == 0) {
        struct shl_bm_sections *section = (struct shl_bm_sections *)(params + 4128);
        return csinn_(params + section->params_offset * 4096);
    } else {
        return NULL;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Please set valide args: ./model.elf hhb.bm\n");
        return -1;
    }

    void *sess = create_graph(argv[1]);

    csinn_session_deinit(sess);
    csinn_free_session(sess);

    return 0;
}

