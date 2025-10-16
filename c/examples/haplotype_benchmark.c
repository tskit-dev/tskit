#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <tskit.h>
#include <tskit/genotypes.h>
#include <tskit/tables.h>

#define CHECK_TSK(err)                                                                  \
    do {                                                                                \
        if ((err) < 0) {                                                                \
            fprintf(stderr, "Error: line %d: %s\n", __LINE__, tsk_strerror(err));       \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

#define NUM_ITERATIONS 1
#define MAX_BENCHMARK_NODES 500

int
main(int argc, char **argv)
{
    int ret;
    tsk_table_collection_t tables;
    tsk_treeseq_t treeseq;
    tsk_haplotype_t haplotype_decoder;
    int8_t *haplotype = NULL;
    double elapsed_seconds;
    clock_t start_clock, end_clock;
    uint64_t checksum = 0;

    const char *filename = "../../simulated_chrom_21_100k.ts";
    if (argc > 1) {
        filename = argv[1];
    }

    ret = tsk_table_collection_init(&tables, 0);
    CHECK_TSK(ret);

    ret = tsk_table_collection_load(&tables, filename, 0);
    CHECK_TSK(ret);

    ret = tsk_treeseq_init(&treeseq, &tables, 0);
    CHECK_TSK(ret);

    tsk_size_t num_nodes = tsk_treeseq_get_num_nodes(&treeseq);
    tsk_size_t num_sites = tsk_treeseq_get_num_sites(&treeseq);
    if (num_sites == 0) {
        fprintf(stderr, "Tree sequence has no sites\n");
        exit(EXIT_FAILURE);
    }

    tsk_id_t node_limit
        = (tsk_id_t) (num_nodes < MAX_BENCHMARK_NODES ? num_nodes : MAX_BENCHMARK_NODES);

    ret = tsk_haplotype_init(&haplotype_decoder, &treeseq, 0, (tsk_id_t) num_sites);
    CHECK_TSK(ret);

    haplotype = malloc(num_sites * sizeof(*haplotype));
    if (haplotype == NULL) {
        fprintf(stderr, "Failed to allocate haplotype buffer\n");
        exit(EXIT_FAILURE);
    }

    start_clock = clock();
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        for (tsk_id_t node = 0; node < node_limit; node++) {
            ret = tsk_haplotype_decode(&haplotype_decoder, node, haplotype);
            CHECK_TSK(ret);
            for (tsk_id_t site = 0; site < (tsk_id_t) num_sites; site++) {
                checksum += (uint64_t) haplotype[site];
            }
        }
    }
    end_clock = clock();

    elapsed_seconds = (double) (end_clock - start_clock) / CLOCKS_PER_SEC;

    printf("Loaded tree sequence from %s\n", filename);
    printf("Decoded %d iterations over %lld nodes × %lld sites in %.3f seconds\n",
        NUM_ITERATIONS, (long long) node_limit, (long long) num_sites, elapsed_seconds);
    printf("Checksummed haplotypes: %llu\n", (unsigned long long) checksum);

    free(haplotype);
    tsk_haplotype_free(&haplotype_decoder);
    tsk_treeseq_free(&treeseq);
    tsk_table_collection_free(&tables);
    return EXIT_SUCCESS;
}
