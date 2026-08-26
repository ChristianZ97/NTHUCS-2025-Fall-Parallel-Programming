/**
 * Parallel odd-even sort using MPI and MPI-IO.
 *
 * Each rank sorts a contiguous partition, then exchanges boundary values with
 * alternating neighbors. An ordered boundary skips the full merge-split. When
 * an exchange is required, a partial merge retains the lower or upper partition
 * in an alternate buffer and swaps the buffer pointers. Local sorting selects
 * insertion sort, std::sort, or spreadsort according to partition size. A
 * periodic collective check permits early termination.
 *
 * Define PROFILING to enable timing and NVTX instrumentation.
 */


/* Headers */
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mpi.h>
#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <errno.h> // For strtol error checking


#ifdef PROFILING
#include <nvtx3/nvToolsExt.h>
#include <string.h>

// ============================================================================
// NVTX Color Definitions for Performance Profiling
// ============================================================================
#define COLOR_IO           0xFF5C9FFF  // Azure Blue
#define COLOR_BOUNDARY     0xFFFF6B81  // Rose Red (Boundary Check)
#define COLOR_DATA_EXC     0xFFFFA500  // Orange (Data Exchange)
#define COLOR_LOCAL_SORT   0xFF5FD068  // Spring Green (Local Sort)
#define COLOR_MERGE_SPLIT  0xFF32CD32  // Lime Green (Merge-Split)
#define COLOR_SORTED_CHECK 0xFF9B59B6  // Amethyst Purple (Sorted Check)
#define COLOR_SETUP        0xFFFFBE3D  // Sunflower Yellow
#define COLOR_DEFAULT      0xFFCBD5E0  // Soft Gray

// ============================================================================
// Smart NVTX Macro: Auto-selects color based on range name pattern
// ============================================================================
#define NVTX_PUSH(name) \
    do { \
        uint32_t color = COLOR_DEFAULT; \
        if (strstr(name, "IO_") != NULL) { \
            color = COLOR_IO; \
        } else if (strcmp(name, "Boundary_Check") == 0) { \
            color = COLOR_BOUNDARY; \
        } else if (strcmp(name, "Data_Exchange") == 0) { \
            color = COLOR_DATA_EXC; \
        } else if (strcmp(name, "Sorted_Check") == 0) { \
            color = COLOR_SORTED_CHECK; \
        } else if (strcmp(name, "Local_Sort") == 0) { \
            color = COLOR_LOCAL_SORT; \
        } else if (strcmp(name, "Merge_Split") == 0) { \
            color = COLOR_MERGE_SPLIT; \
        } else if (strcmp(name, "MPI_Setup") == 0 || strcmp(name, "Mem_Alloc") == 0) { \
            color = COLOR_SETUP; \
        } else { \
            color = COLOR_DEFAULT; \
        } \
        nvtxEventAttributes_t eventAttrib = {0}; \
        eventAttrib.version = NVTX_VERSION; \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        eventAttrib.colorType = NVTX_COLOR_ARGB; \
        eventAttrib.color = color; \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name; \
        nvtxRangePushEx(&eventAttrib); \
    } while(0)


#define NVTX_POP() nvtxRangePop()
#endif


/* Function Prototypes */
void local_sort(float local_data[], const int my_count);
void merge_sort_split(float *&local_data, const int my_count, float *recv_data, const int recv_count, float *&temp, const int is_left);
int sorted_check(float *local_data, const int my_count, const int my_rank, const int numtasks, const int phase, MPI_Comm comm);

/* Main Function */
int main(int argc, char *argv[]) {

    // ========================================================================
    // Block 1: MPI Initialization and Environment Setup
    // ========================================================================
    MPI_Init(&argc, &argv);

    #ifdef PROFILING
    double total_start_time, temp_start, io_time = 0.0, comm_time = 0.0;
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before starting the main timer
    total_start_time = MPI_Wtime();
    NVTX_PUSH("MPI_Setup");
    #endif

    // Use strtol for safer conversion from command-line argument.
    char *endptr;
    long val = strtol(argv[1], &endptr, 10);
    const int N = (int)val;
    
    /* MPI Init and Grouping for active processes */
    MPI_Group orig_group, active_group;
    MPI_Comm active_comm;
    int world_numtasks;

    MPI_Comm_size(MPI_COMM_WORLD, &world_numtasks);
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

    // Only create a new communicator for processes that might receive data.
    const int active_numtasks = std::min(world_numtasks, N);
    int *active_ranks = (int *)malloc(active_numtasks * sizeof(int));
    if (!active_ranks) MPI_Abort(MPI_COMM_WORLD, 1);
    for (int i = 0; i < active_numtasks; i++) active_ranks[i] = i;

    MPI_Group_incl(orig_group, active_numtasks, active_ranks, &active_group);
    MPI_Comm_create(MPI_COMM_WORLD, active_group, &active_comm);
    
    int numtasks = 0, my_rank = -1;
    if (active_comm != MPI_COMM_NULL) {
        MPI_Comm_size(active_comm, &numtasks);
        MPI_Comm_rank(active_comm, &my_rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef PROFILING
    NVTX_POP(); // end MPI_Setup
    #endif

    // ========================================================================
    // Block 2: Data Distribution and Memory Allocation
    // ========================================================================
    /* Data distribution and I/O preparation */
    const int valid_tasks = (numtasks > 0) ? 1 : 0;
    const int valid_rank = (my_rank != -1) ? 1 : 0;
    const int base_chunk_size = (valid_tasks) ? N / numtasks : 0;
    const int remainder = (valid_tasks) ? N % numtasks : 0;
    const int my_count = (valid_rank && my_rank < remainder) ? base_chunk_size + 1 : base_chunk_size;
    const int my_start_index = (valid_rank) ? (my_rank * base_chunk_size + std::min(my_rank, remainder)) : 0;
    const int is_active = ((active_comm != MPI_COMM_NULL) && (my_count > 0)) ? 1 : 0;

    // Robustness check - prevent integer overflow before memory allocation.
    const size_t max_elements = (size_t)(base_chunk_size + 1);
    if (max_elements > SIZE_MAX / sizeof(float)) MPI_Abort(MPI_COMM_WORLD, 1);
    const size_t unaligned_bytes = max_elements * sizeof(float);
    if (unaligned_bytes > SIZE_MAX - 31) MPI_Abort(MPI_COMM_WORLD, 1);
    const size_t chunk_size_bytes = ((unaligned_bytes + 31) / 32) * 32;

    MPI_File input_file, output_file;
    const char *const input_filename = argv[2];
    const char *const output_filename = argv[3];
    float *temp = NULL;
    float *local_data = NULL;
    float *recv_data = NULL;

    if (is_active) {

        #ifdef PROFILING
        NVTX_PUSH("Mem_Alloc");
        #endif
        // Aligned Memory Allocation.
        temp = (float *)aligned_alloc(32, chunk_size_bytes);
        local_data = (float *)aligned_alloc(32, chunk_size_bytes);
        recv_data = (float *)aligned_alloc(32, chunk_size_bytes);

        // Fallback to malloc for robustness if aligned_alloc fails.
        if (!temp || !local_data || !recv_data) {
            if (temp) free(temp);
            if (local_data) free(local_data);
            if (recv_data) free(recv_data);
            
            temp = (float *)malloc(chunk_size_bytes);
            local_data = (float *)malloc(chunk_size_bytes);
            recv_data = (float *)malloc(chunk_size_bytes);
            
            if (!temp || !local_data || !recv_data) MPI_Abort(active_comm, 1);
        }
        #ifdef PROFILING
        NVTX_POP(); // end Mem_Alloc
        #endif
    } // end is_active

    if (is_active) {

        // ========================================================================
        // Block 3: Initial I/O (Read)
        // ========================================================================
        #ifdef PROFILING
        temp_start = MPI_Wtime();
        NVTX_PUSH("IO_Read");
        #endif
        // Initial data read and sort.
        MPI_File_open(active_comm, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
        MPI_File_read_at(input_file, my_start_index * sizeof(float), local_data, my_count, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&input_file);
        #ifdef PROFILING
        io_time += MPI_Wtime() - temp_start; // end io_time
        NVTX_POP(); // end IO_Read
        #endif

        // ========================================================================
        // Block 4: Initial Computation (Local Sort)
        // ========================================================================
        // Adaptive Local Sort is called here.
        #ifdef PROFILING
        NVTX_PUSH("Local_Sort");
        #endif
        local_sort(local_data, my_count);
        #ifdef PROFILING
        NVTX_POP(); // end Local_Sort
        #endif
    } // end is_active

    if (is_active) {
        // ========================================================================
        // Block 5: Main Communication & Computation Loop
        // ========================================================================
        #ifdef PROFILING
        NVTX_PUSH("Main_Loop");
        #endif
        /* Main sorting loop */
        const int max_phases = numtasks + (numtasks / 2);
        const int is_odd_rank = (my_rank % 2) ? 1 : 0;
        float partner_boundary = -FLT_MAX;

        for (int phase = 0; phase < max_phases; phase++) {
            const int phase_odd = (phase % 2) ? 1 : 0;
            int partner = -1;

            if (phase_odd) partner = (is_odd_rank) ? my_rank + 1 : my_rank - 1;
            else partner = (is_odd_rank) ? my_rank - 1 : my_rank + 1;
            if ((partner < 0) || (partner >= numtasks)) partner = MPI_PROC_NULL;
            const int valid_comm = (partner != MPI_PROC_NULL) ? 1 : 0;
            
            if (valid_comm) {

                const int recv_count = (partner < remainder) ? base_chunk_size + 1 : base_chunk_size;
                // Unique MPI tags ensure message correctness.
                const int mpi_boundary_tag = 2 * phase;
                const int mpi_data_tag = 2 * phase + 1;
                const int is_left = (my_rank < partner) ? 1 : 0;
                
                // Conditional Merge-Split logic begins here.
                if (is_left) {
                    const float my_last = local_data[my_count - 1];

                    #ifdef PROFILING
                    temp_start = MPI_Wtime();
                    NVTX_PUSH("Boundary_Check");
                    #endif
                    MPI_Sendrecv(&my_last, 1, MPI_FLOAT, partner, mpi_boundary_tag,
                                 &partner_boundary, 1, MPI_FLOAT, partner, mpi_boundary_tag,
                                 active_comm, MPI_STATUS_IGNORE);
                    #ifdef PROFILING
                    comm_time += MPI_Wtime() - temp_start; // end comm_time
                    NVTX_POP(); // end Boundary_Check
                    #endif

                    if (my_last > partner_boundary) {
                        
                        #ifdef PROFILING
                        temp_start = MPI_Wtime();
                        NVTX_PUSH("Data_Exchange");
                        #endif
                        MPI_Sendrecv(local_data, my_count, MPI_FLOAT, partner, mpi_data_tag,
                                     recv_data, recv_count, MPI_FLOAT, partner, mpi_data_tag,
                                     active_comm, MPI_STATUS_IGNORE);
                        #ifdef PROFILING
                        comm_time += MPI_Wtime() - temp_start; // end comm_time
                        NVTX_POP(); // end Data_Exchange
                        #endif
                        
                        #ifdef PROFILING
                        NVTX_PUSH("Merge_Split");
                        #endif
                        // Zero-Copy Merge-Split is called here.
                        merge_sort_split(local_data, my_count, recv_data, recv_count, temp, is_left);
                        #ifdef PROFILING
                        NVTX_POP(); // end Merge_Split
                        #endif

                    }
                
                } else { // I am the right process.
                    const float my_first = local_data[0];

                    #ifdef PROFILING
                    temp_start = MPI_Wtime();
                    NVTX_PUSH("Boundary_Check");
                    #endif
                    MPI_Sendrecv(&my_first, 1, MPI_FLOAT, partner, mpi_boundary_tag,
                                 &partner_boundary, 1, MPI_FLOAT, partner, mpi_boundary_tag,
                                 active_comm, MPI_STATUS_IGNORE);
                    #ifdef PROFILING
                    comm_time += MPI_Wtime() - temp_start; // end comm_time
                    NVTX_POP(); // end Boundary_Check
                    #endif

                    if (my_first < partner_boundary) {

                        #ifdef PROFILING
                        temp_start = MPI_Wtime();
                        NVTX_PUSH("Data_Exchange");
                        #endif
                        MPI_Sendrecv(local_data, my_count, MPI_FLOAT, partner, mpi_data_tag,
                                     recv_data, recv_count, MPI_FLOAT, partner, mpi_data_tag,
                                     active_comm, MPI_STATUS_IGNORE);
                        #ifdef PROFILING
                        comm_time += MPI_Wtime() - temp_start; // end comm_time
                        NVTX_POP(); // end Data_Exchange
                        #endif
                        
                        #ifdef PROFILING
                        NVTX_PUSH("Merge_Split");
                        #endif
                        // Zero-Copy Merge-Split is called here.
                        merge_sort_split(local_data, my_count, recv_data, recv_count, temp, is_left);
                        #ifdef PROFILING
                        NVTX_POP(); // end Merge_Split
                        #endif
                    }
                }
            } // end valid_comm

            // Efficient Early Exit check.
            if (phase >= numtasks / 2 && !phase_odd) {

                #ifdef PROFILING
                temp_start = MPI_Wtime();
                NVTX_PUSH("Sorted_Check");
                #endif
                const int done = sorted_check(local_data, my_count, my_rank, numtasks, phase, active_comm);
                #ifdef PROFILING
                comm_time += MPI_Wtime() - temp_start; // end comm_time
                NVTX_POP(); // end Sorted_Check
                #endif

                if (done) break;
            }

        } // end for loop
        #ifdef PROFILING
        NVTX_POP(); // end Main_Loop
        #endif
    } // end is_active

    if (is_active) {

        // ========================================================================
        // Block 6: Final I/O (Write)
        // ========================================================================
        #ifdef PROFILING
        temp_start = MPI_Wtime();
        NVTX_PUSH("IO_Write");
        #endif
        // Final write to output file.
        MPI_File_open(active_comm, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
        MPI_File_write_at(output_file, my_start_index * sizeof(float), local_data, my_count, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&output_file);
        #ifdef PROFILING
        io_time += MPI_Wtime() - temp_start; // end io_time
        NVTX_POP(); // end IO_Write
        #endif

        free(local_data); free(recv_data); free(temp);
    } // end is_active

    // ========================================================================
    // Block 7: Finalization and Cleanup
    // ========================================================================
    #ifdef PROFILING
    MPI_Barrier(MPI_COMM_WORLD);
    double total_time = MPI_Wtime() - total_start_time;
    double cpu_time = total_time - io_time - comm_time;

    // All processes participate, inactive ones contribute 0
    double avg_io_time = 0.0, avg_comm_time = 0.0, avg_cpu_time = 0.0, avg_total_time = 0.0;
    double max_total_time = 0.0, min_total_time = total_time;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Reduce(&io_time, &avg_io_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &avg_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&cpu_time, &avg_cpu_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &avg_total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &min_total_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (world_rank == 0 && world_numtasks > 0) {
        avg_io_time /= world_numtasks;
        avg_comm_time /= world_numtasks;
        avg_cpu_time /= world_numtasks;
        avg_total_time /= world_numtasks;
        
        printf("N=%d, Procs=%d, Avg Total Time: %.6f s\n", N, world_numtasks, avg_total_time);
        printf("  Avg IO Time:   %.6f s (%.2f%%)\n", avg_io_time, (avg_io_time / avg_total_time) * 100);
        printf("  Avg Comm Time: %.6f s (%.2f%%)\n", avg_comm_time, (avg_comm_time / avg_total_time) * 100);
        printf("  Avg CPU Time:  %.6f s (%.2f%%)\n", avg_cpu_time, (avg_cpu_time / avg_total_time) * 100);
        printf("  Max Total Time: %.6f s, Min Total Time: %.6f s\n", max_total_time, min_total_time);
    }
    #endif

    #ifdef PROFILING
    NVTX_PUSH("MPI_Finalize");
    #endif
    /* Cleanup MPI resources */
    free(active_ranks);
    MPI_Group_free(&orig_group);
    if (active_comm != MPI_COMM_NULL) MPI_Group_free(&active_group);
    if (active_comm != MPI_COMM_NULL) MPI_Comm_free(&active_comm);
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef PROFILING
    NVTX_POP(); // end MPI_Finalize
    #endif

    MPI_Finalize();
    return 0;
}


/**
 * @brief Sorts the local data array using an adaptive strategy.
 * This function implements the adaptive local sort. It chooses
 * an appropriate algorithm based on the array size to balance overhead and
 * raw performance.
 */
void local_sort(float local_data[], const int my_count) {
    if (my_count < 2) return;
    
    // For tiny arrays, insertion sort has the lowest overhead.
    if (my_count < 33) {
        for (int i = 1; i < my_count; i++) {
            float temp_val = local_data[i];
            int j = i - 1;
            while (j >= 0 && temp_val < local_data[j]) { 
                local_data[j + 1] = local_data[j];
                j--; 
            }
            local_data[j + 1] = temp_val;
        }
    }
    // For large arrays, radix-based spreadsort is extremely fast for floats.
    else boost::sort::spreadsort::float_sort(local_data, local_data + my_count);
    // O(N)
}


/**
 * @brief Merges local data with received data from a partner process.
 * This function implements the Zero-Copy merge-split.
 * It merges into a temporary buffer and then uses std::swap to switch pointers,
 * avoiding a costly memory copy.
 */
void merge_sort_split(float *&local_data, const int my_count, float *recv_data, const int recv_count, float *&temp, const int is_left) {
    // Guard clause for cases with no data to merge.
    if (my_count < 1) return;
    
    if (is_left) { // I am the left process, I keep the smaller elements.
        int i = 0, j = 0, k = 0;
        while (k < my_count && i < my_count && j < recv_count) {
            temp[k++] = (local_data[i] <= recv_data[j]) ? local_data[i++] : recv_data[j++];
        }
        while (k < my_count && i < my_count) temp[k++] = local_data[i++];
        while (k < my_count && j < recv_count) temp[k++] = recv_data[j++];

    } else { // I am the right process, I keep the larger elements.
        int i = my_count - 1, j = recv_count - 1, k = my_count - 1;
        while (k >= 0 && i >= 0 && j >= 0) {
            temp[k--] = (local_data[i] >= recv_data[j]) ? local_data[i--] : recv_data[j--];
        }
        while (k >= 0 && i >= 0) temp[k--] = local_data[i--];
        while (k >= 0 && j >= 0) temp[k--] = recv_data[j--];
    }
    
    // The O(1) pointer swap, the core of the Zero-Copy strategy.
    std::swap(local_data, temp);
    //memcpy(local_data, temp, my_count * sizeof(float));
}


/**
 * @brief Checks if the global array is sorted by verifying boundary elements.
 * This function implements the efficient early exit check.
 */
int sorted_check(float *local_data, const int my_count, const int my_rank, const int numtasks, const int phase, MPI_Comm comm) {
    const int prev_rank = (my_rank > 0) ? my_rank - 1 : MPI_PROC_NULL;
    const int next_rank = (my_rank < numtasks - 1) ? my_rank + 1 : MPI_PROC_NULL;
    
    // Robustly handle cases where my_count is 0.
    const float my_first = (my_count > 0) ? local_data[0] : FLT_MAX;
    const float my_last = (my_count > 0) ? local_data[my_count - 1] : -FLT_MAX;
    
    int boundary_sorted = 1, global_sorted = 0;
    float prev_last = -FLT_MAX;

    // Each process sends its last element to its right neighbor and receives
    // the last element from its left neighbor.
    MPI_Sendrecv(&my_last, 1, MPI_FLOAT, next_rank, phase,
                 &prev_last, 1, MPI_FLOAT, prev_rank, phase,
                 comm, MPI_STATUS_IGNORE);

    // If my left neighbor's last element is greater than my first, we are not sorted.
    if (my_rank > 0 && my_count > 0 && prev_last > my_first) boundary_sorted = 0;
    
    // Perform a global AND operation. If any process found an unsorted boundary,
    // the global result will be 0.
    MPI_Allreduce(&boundary_sorted, &global_sorted, 1, MPI_INT, MPI_LAND, comm);
    return global_sorted;
}
