/**
 * Distributed odd-even sort for binary arrays of 32-bit floats.
 *
 * Each active MPI rank sorts one contiguous partition. Alternating neighbor
 * pairs first exchange their boundary values and skip the full exchange when
 * the two partitions are already ordered. Otherwise, each rank keeps the
 * appropriate half of a partial merge and swaps its working-buffer pointer.
 */

#include <algorithm>
#include <cerrno>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <boost/sort/spreadsort/spreadsort.hpp>
#include <mpi.h>

void local_sort(float local_data[], int count);
void merge_sort_split(float *&local_data,
                      int local_count,
                      float *received_data,
                      int received_count,
                      float *&temporary,
                      bool keep_lower);
int sorted_check(const float *local_data,
                 int local_count,
                 int rank,
                 int rank_count,
                 int phase,
                 MPI_Comm communicator);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    if (argc != 4) {
        int error_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &error_rank);
        if (error_rank == 0) {
            std::fprintf(stderr,
                         "Usage: %s <element-count> <input.bin> <output.bin>\n",
                         argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    errno = 0;
    char *end = nullptr;
    const long parsed_count = std::strtol(argv[1], &end, 10);
    if (errno == ERANGE || end == argv[1] || *end != '\0' ||
        parsed_count < 1 || parsed_count > INT_MAX) {
        int error_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &error_rank);
        if (error_rank == 0) {
            std::fprintf(stderr,
                         "element-count must be an integer in [1, %d]\n",
                         INT_MAX);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    const int element_count = static_cast<int>(parsed_count);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Group world_group;
    MPI_Group active_group;
    MPI_Comm active_communicator;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    const int active_rank_count = std::min(world_size, element_count);
    int *active_ranks =
        static_cast<int *>(std::malloc(active_rank_count * sizeof(int)));
    if (active_ranks == nullptr) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    for (int rank = 0; rank < active_rank_count; ++rank) {
        active_ranks[rank] = rank;
    }

    MPI_Group_incl(
        world_group, active_rank_count, active_ranks, &active_group);
    MPI_Comm_create(
        MPI_COMM_WORLD, active_group, &active_communicator);

    int rank_count = 0;
    int rank = -1;
    if (active_communicator != MPI_COMM_NULL) {
        MPI_Comm_size(active_communicator, &rank_count);
        MPI_Comm_rank(active_communicator, &rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const int base_partition_size =
        rank_count > 0 ? element_count / rank_count : 0;
    const int remainder =
        rank_count > 0 ? element_count % rank_count : 0;
    const int local_count =
        rank >= 0 && rank < remainder
            ? base_partition_size + 1
            : base_partition_size;
    const int local_offset =
        rank >= 0
            ? rank * base_partition_size + std::min(rank, remainder)
            : 0;
    const bool is_active =
        active_communicator != MPI_COMM_NULL && local_count > 0;

    const std::size_t max_elements =
        static_cast<std::size_t>(base_partition_size) + 1;
    if (max_elements > SIZE_MAX / sizeof(float)) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const std::size_t unaligned_bytes = max_elements * sizeof(float);
    if (unaligned_bytes > SIZE_MAX - 31) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const std::size_t buffer_bytes =
        ((unaligned_bytes + 31) / 32) * 32;

    float *temporary = nullptr;
    float *local_data = nullptr;
    float *received_data = nullptr;

    if (is_active) {
        temporary = static_cast<float *>(std::aligned_alloc(32, buffer_bytes));
        local_data = static_cast<float *>(std::aligned_alloc(32, buffer_bytes));
        received_data =
            static_cast<float *>(std::aligned_alloc(32, buffer_bytes));

        if (temporary == nullptr || local_data == nullptr ||
            received_data == nullptr) {
            std::free(temporary);
            std::free(local_data);
            std::free(received_data);

            temporary = static_cast<float *>(std::malloc(buffer_bytes));
            local_data = static_cast<float *>(std::malloc(buffer_bytes));
            received_data = static_cast<float *>(std::malloc(buffer_bytes));

            if (temporary == nullptr || local_data == nullptr ||
                received_data == nullptr) {
                MPI_Abort(active_communicator, EXIT_FAILURE);
            }
        }
    }

    const char *input_filename = argv[2];
    const char *output_filename = argv[3];
    const MPI_Offset file_offset =
        static_cast<MPI_Offset>(local_offset) * sizeof(float);
    MPI_File input_file;
    MPI_File output_file;

    if (is_active) {
        if (MPI_File_open(active_communicator,
                          input_filename,
                          MPI_MODE_RDONLY,
                          MPI_INFO_NULL,
                          &input_file) != MPI_SUCCESS) {
            MPI_Abort(active_communicator, EXIT_FAILURE);
        }
        if (MPI_File_read_at(input_file,
                             file_offset,
                             local_data,
                             local_count,
                             MPI_FLOAT,
                             MPI_STATUS_IGNORE) != MPI_SUCCESS ||
            MPI_File_close(&input_file) != MPI_SUCCESS) {
            MPI_Abort(active_communicator, EXIT_FAILURE);
        }

        local_sort(local_data, local_count);
    }

    if (is_active) {
        const int max_phases = rank_count + rank_count / 2;
        const bool odd_rank = rank % 2 != 0;
        float partner_boundary = -FLT_MAX;

        for (int phase = 0; phase < max_phases; ++phase) {
            const bool odd_phase = phase % 2 != 0;
            int partner = -1;

            if (odd_phase) {
                partner = odd_rank ? rank + 1 : rank - 1;
            } else {
                partner = odd_rank ? rank - 1 : rank + 1;
            }
            if (partner < 0 || partner >= rank_count) {
                partner = MPI_PROC_NULL;
            }

            if (partner != MPI_PROC_NULL) {
                const int received_count =
                    partner < remainder
                        ? base_partition_size + 1
                        : base_partition_size;
                const int boundary_tag = 2 * phase;
                const int data_tag = boundary_tag + 1;
                const bool keep_lower = rank < partner;

                if (keep_lower) {
                    const float local_boundary =
                        local_data[local_count - 1];
                    MPI_Sendrecv(&local_boundary,
                                 1,
                                 MPI_FLOAT,
                                 partner,
                                 boundary_tag,
                                 &partner_boundary,
                                 1,
                                 MPI_FLOAT,
                                 partner,
                                 boundary_tag,
                                 active_communicator,
                                 MPI_STATUS_IGNORE);

                    if (local_boundary > partner_boundary) {
                        MPI_Sendrecv(local_data,
                                     local_count,
                                     MPI_FLOAT,
                                     partner,
                                     data_tag,
                                     received_data,
                                     received_count,
                                     MPI_FLOAT,
                                     partner,
                                     data_tag,
                                     active_communicator,
                                     MPI_STATUS_IGNORE);
                        merge_sort_split(local_data,
                                         local_count,
                                         received_data,
                                         received_count,
                                         temporary,
                                         keep_lower);
                    }
                } else {
                    const float local_boundary = local_data[0];
                    MPI_Sendrecv(&local_boundary,
                                 1,
                                 MPI_FLOAT,
                                 partner,
                                 boundary_tag,
                                 &partner_boundary,
                                 1,
                                 MPI_FLOAT,
                                 partner,
                                 boundary_tag,
                                 active_communicator,
                                 MPI_STATUS_IGNORE);

                    if (local_boundary < partner_boundary) {
                        MPI_Sendrecv(local_data,
                                     local_count,
                                     MPI_FLOAT,
                                     partner,
                                     data_tag,
                                     received_data,
                                     received_count,
                                     MPI_FLOAT,
                                     partner,
                                     data_tag,
                                     active_communicator,
                                     MPI_STATUS_IGNORE);
                        merge_sort_split(local_data,
                                         local_count,
                                         received_data,
                                         received_count,
                                         temporary,
                                         keep_lower);
                    }
                }
            }

            if (phase >= rank_count / 2 && !odd_phase) {
                const int done = sorted_check(local_data,
                                              local_count,
                                              rank,
                                              rank_count,
                                              phase,
                                              active_communicator);
                if (done) {
                    break;
                }
            }
        }
    }

    if (is_active) {
        if (MPI_File_open(active_communicator,
                          output_filename,
                          MPI_MODE_CREATE | MPI_MODE_WRONLY,
                          MPI_INFO_NULL,
                          &output_file) != MPI_SUCCESS) {
            MPI_Abort(active_communicator, EXIT_FAILURE);
        }
        if (MPI_File_set_size(
                output_file,
                static_cast<MPI_Offset>(element_count) * sizeof(float)) !=
                MPI_SUCCESS ||
            MPI_File_write_at(output_file,
                              file_offset,
                              local_data,
                              local_count,
                              MPI_FLOAT,
                              MPI_STATUS_IGNORE) != MPI_SUCCESS ||
            MPI_File_close(&output_file) != MPI_SUCCESS) {
            MPI_Abort(active_communicator, EXIT_FAILURE);
        }

        std::free(local_data);
        std::free(received_data);
        std::free(temporary);
    }

    std::free(active_ranks);
    MPI_Group_free(&world_group);
    MPI_Group_free(&active_group);
    if (active_communicator != MPI_COMM_NULL) {
        MPI_Comm_free(&active_communicator);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return EXIT_SUCCESS;
}

void local_sort(float local_data[], const int count) {
    if (count < 2) {
        return;
    }

    if (count < 33) {
        for (int i = 1; i < count; ++i) {
            const float value = local_data[i];
            int j = i - 1;
            while (j >= 0 && value < local_data[j]) {
                local_data[j + 1] = local_data[j];
                --j;
            }
            local_data[j + 1] = value;
        }
    } else {
        boost::sort::spreadsort::float_sort(
            local_data, local_data + count);
    }
}

void merge_sort_split(float *&local_data,
                      const int local_count,
                      float *received_data,
                      const int received_count,
                      float *&temporary,
                      const bool keep_lower) {
    if (local_count < 1) {
        return;
    }

    if (keep_lower) {
        int local_index = 0;
        int received_index = 0;
        int output_index = 0;

        while (output_index < local_count &&
               local_index < local_count &&
               received_index < received_count) {
            temporary[output_index++] =
                local_data[local_index] <= received_data[received_index]
                    ? local_data[local_index++]
                    : received_data[received_index++];
        }
        while (output_index < local_count && local_index < local_count) {
            temporary[output_index++] = local_data[local_index++];
        }
        while (output_index < local_count &&
               received_index < received_count) {
            temporary[output_index++] = received_data[received_index++];
        }
    } else {
        int local_index = local_count - 1;
        int received_index = received_count - 1;
        int output_index = local_count - 1;

        while (output_index >= 0 &&
               local_index >= 0 &&
               received_index >= 0) {
            temporary[output_index--] =
                local_data[local_index] >= received_data[received_index]
                    ? local_data[local_index--]
                    : received_data[received_index--];
        }
        while (output_index >= 0 && local_index >= 0) {
            temporary[output_index--] = local_data[local_index--];
        }
        while (output_index >= 0 && received_index >= 0) {
            temporary[output_index--] = received_data[received_index--];
        }
    }

    std::swap(local_data, temporary);
}

int sorted_check(const float *local_data,
                 const int local_count,
                 const int rank,
                 const int rank_count,
                 const int phase,
                 MPI_Comm communicator) {
    const int previous_rank = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next_rank =
        rank < rank_count - 1 ? rank + 1 : MPI_PROC_NULL;
    const float local_first =
        local_count > 0 ? local_data[0] : FLT_MAX;
    const float local_last =
        local_count > 0 ? local_data[local_count - 1] : -FLT_MAX;

    float previous_last = -FLT_MAX;
    MPI_Sendrecv(&local_last,
                 1,
                 MPI_FLOAT,
                 next_rank,
                 phase,
                 &previous_last,
                 1,
                 MPI_FLOAT,
                 previous_rank,
                 phase,
                 communicator,
                 MPI_STATUS_IGNORE);

    int boundary_sorted = 1;
    if (rank > 0 && local_count > 0 && previous_last > local_first) {
        boundary_sorted = 0;
    }

    int globally_sorted = 0;
    MPI_Allreduce(&boundary_sorted,
                  &globally_sorted,
                  1,
                  MPI_INT,
                  MPI_LAND,
                  communicator);
    return globally_sorted;
}
