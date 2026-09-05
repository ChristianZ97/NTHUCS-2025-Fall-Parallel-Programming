---
title: "CS542200 Parallel Programming Homework 5: Observe the behavior of UCX yourself"

---

# PP HW 5 Report Template
> - Please include both brief and detailed answers.
> - The report should be based on the UCX code.
> - Describe the code using the 'permalink' from [GitHub repository](https://github.com/NTHU-LSALAB/UCX-lsalab).

## 1. Overview
> In conjunction with the UCP architecture mentioned in the lecture, please read [ucp_hello_world.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/examples/ucp_hello_world.c)
1. Identify how UCP Objects (`ucp_context`, `ucp_worker`, `ucp_ep`) interact through the API, including at least the following functions:
    - `ucp_init`
    - `ucp_worker_create`
    - `ucp_ep_create`
2. UCX abstracts communication into three layers as below. Please provide a diagram illustrating the architectural design of UCX.
    - `ucp_context`
    - `ucp_worker`
    - `ucp_ep`
> Please provide detailed example information in the diagram corresponding to the execution of the command `srun -N 2 ./send_recv.out` or `mpiucx --host HostA:1,HostB:1 ./send_recv.out`
3. Based on the description in HW5, where do you think the following information is loaded/created?
    - `UCX_TLS`
    - TLS selected by UCX

Context represents the global scope and resource management for the UCP library. In the code, `status = ucp_init(&ucp_params, config, &ucp_context);` initializes the UCP context. This function uses the configuration read earlier via `ucp_config_read` and the parameters specified in `ucp_params` (which includes features like OOB communication support) to set up the shared resources and capabilities available for the application. Following context initialization, `status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);` is called to create a Worker. The Worker is created within the scope of a specific Context and utilizes the resources managed by that Context. It serves as the engine for progress, responsible for processing communication requests, managing operational queues, and polling hardware resources for event completions. Finally, Endpoints are created to establish connections. The calls `status = ucp_ep_create(ucp_worker, &ep_params, &server_ep);` and `status = ucp_ep_create(ucp_worker, &ep_params, &client_ep);` instantiate Endpoints attached to the specific `ucp_worker`. An Endpoint represents a logical, persistent connection to a specific remote peer (whose details are provided in `ep_params`). An Endpoint cannot exist independently of a Worker, as it relies on the Worker to drive the underlying transport operations for all point-to-point communications.

In summary, the application first creates a shared communication context, then one or more workers bound to that context, and finally multiple endpoints attached to each worker to talk to remote peers.

<img width="2160" height="1215" alt="ucx_arch" src="https://github.com/user-attachments/assets/562b726b-e5a2-4d9c-be1e-a07fe8815282" />

The diagram illustrates the UCX architecture during the execution of `srun -N 2 ./send_recv.out`, where two MPI processes (Rank 0 and Rank 1) communicate across two separate nodes (Host A and Host B). Host A (Rank 0) and Host B (Rank 1) each initialize a `ucp_context`. Both contexts detect and utilize the InfiniBand device `ibp3s0:1` as the underlying resource, configured to use the `ud_verbs` transport layer as specified by the environment (`UCX_TLS=ud_verbs`). A `ucp_worker` is created on each host (Addresses: `0x557...` on Host A, `0x556...` on Host B). These workers manage the progress of communication and interface with the network hardware. On Host A, the worker manages two endpoints: 1) `ucp_ep (self cfg#0)`: A loopback connection for intra-process communication; 2) `ucp_ep (inter-node cfg#1)`: The critical connection established to transmit data to the remote peer (Host B). On Host B, the worker manages its own `ucp_ep (self cfg#0)` for local operations. The dotted arrow represents the transmission of the message "Hello from rank 0". The message originates from Host A, travels through the `inter-node` endpoint using the `ud_verbs` protocol over the physical network, and is received by Host B's worker.

`UCX_TLS` is loaded during the `ucp_config_read` phase, which occurs before `ucp_init`. In [ucp_hello_world.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/examples/ucp_hello_world.c), `ucp_config_read(NULL, NULL, &config)` is explicitly called. This function, defined in [ucp_context.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/src/ucp/core/ucp_context.c), internally invokes `ucs_config_parser_fill_opts` (defined in [parser.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/src/ucs/config/parser.c)) to read environment variables, including `UCX_TLS`. This populated configuration is then passed to `ucp_init` to create the UCP context. The TLS selected by UCX is determined during the `ucp_worker_create` phase. While the UCP Context is responsible for discovering all available hardware resources (Devices) during initialization, the specific selection of transports—matching these resources against `UCX_TLS` constraints—happens when the Worker is created. The Worker calculates reachability and selects the optimal transports (e.g., `ud_verbs` for inter-node communication, `shm` for intra-node), storing these selections within its internal structures.

Aligning with the UCX Framework architecture (Slide 37), this flow confirms that the **UCS (Services)** layer handles the configuration parsing and infrastructure duties (loading `UCX_TLS`), while the **UCP (Protocols)** layer executes the high-level logic for transport selection and resource binding during worker and endpoint creation.

## 2. Implementation
> Please complete the implementation according to the [spec](https://docs.google.com/document/d/1fmm0TFpLxbDP7neNcbLDn8nhZpqUBi9NGRzWjgxZaPE/edit?usp=sharing)
> Describe how you implemented the two special features of HW5.
1. Which files did you modify, and where did you choose to print Line 1 and Line 2?
2. How do the functions in these files call each other? Why is it designed this way?
3. Observe when Line 1 and 2 are printed during the call of which UCP API?
4. Does it match your expectations for questions **1-3**? Why?
5. In implementing the features, we see variables like lanes, tl_rsc, tl_name, tl_device, bitmap, iface, etc., used to store different Layer's protocol information. Please explain what information each of them stores.

We modified two files: [parser.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/src/ucs/config/parser.c) and [ucp_worker.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/src/ucp/core/ucp_worker.c). In [parser.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/src/ucs/config/parser.c), We defined a new flag `UCS_CONFIG_PRINT_TLS` (mapped to `UCS_BIT(8)`) to specifically trigger the printing of the configured UCX TLS environment variable. Inside the `ucs_config_parser_print_opts` function, we added a conditional block that checks for this flag. If the flag is set, the function retrieves the value of the `UCX_TLS` environment variable using `getenv("UCX_TLS")` and prints "Line 1" in the format `UCX_TLS=<value>`. In `src/ucp/core/ucp_worker.c`, we also defined the same `UCS_CONFIG_PRINT_TLS` flag to ensure consistency. We modified the `ucp_worker_print_used_tls` function to serve as the main entry point for printing the required information. First, we invoked `ucp_config_print` with the custom `UCS_CONFIG_PRINT_TLS` flag. This call triggers the logic added in [parser.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/src/ucs/config/parser.c) to print "Line 1". Second, to print "Line 2", we modified the final output of the function. Instead of using the default `ucs_info` logging, we used `printf` to output the formatted string buffer `strb` directly to stdout. This string buffer contains the address and transport details (e.g., `self cfg#0 tag(...)`) constructed by previous calls like `ucp_ep_config_name`.

The design relies on a hierarchical call flow where runtime configuration discovery triggers information display. Specifically, `ucp_worker_print_used_tls` in [ucp_worker.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/src/ucp/core/ucp_worker.c) is the core function for displaying transport details. Based on the code trace, this function is called by `ucp_worker_get_ep_config` also defined in [ucp_worker.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/src/ucp/core/ucp_worker.c). This happens when the worker initializes a new endpoint configuration (typically during connection establishment or worker creation when self-endpoints are set up). When a new unique configuration key is encountered and added to the `ep_config` array, and assuming the experimental protocol v2 features are not enabled (`!context->config.ext.proto_enable`), `ucp_worker_get_ep_config` explicitly calls `ucp_worker_print_used_tls` to log the transport selection for that specific configuration index (e.g., `cfg#0`). Inside `ucp_worker_print_used_tls`, we injected a call to `ucp_config_print` (defined in [ucp_context.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/src/ucp/core/ucp_context.c)). This function acts as a bridge, delegating the task of printing the global TLS configuration to `ucs_config_parser_print_opts` in [parser.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/src/ucs/config/parser.c), where our logic for printing "Line 1" (`UCX_TLS=...`) resides. After this delegation returns, `ucp_worker_print_used_tls` proceeds to construct and print the runtime transport details ("Line 2") using the worker's internal state. This design ensures that transport information is printed exactly when a new communication path configuration is instantiated.

The implementation largely matches our expectations, with some insightful nuances revealed by the experiments. First, regarding Line 1 (`UCX_TLS=...`), the output consistently reflects the user's input (e.g., `all`, `ud_verbs`), confirming that our modification in [parser.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/src/ucs/config/parser.c) correctly captures the environment variable before UCX internal processing. Second, regarding "Line 2" (Transport Details), the results confirm that UCX's internal selection logic is dynamic and resource-aware. When `UCX_TLS=all` was used, "Line 2" showed `tag(self/memory cma/memory)` and `tag(sysv/memory cma/memory)`, confirming that UCX prioritized Shared Memory (CMA/SysV) for intra-node communication as hypothesized. When `UCX_TLS=ud_verbs` was forced, "Line 2" showed `tag(ud_verbs/ibp3s0:1)`, confirming that UCX respected the constraint and fell back to the network hardware even for local communication. The failure cases (e.g., `UCX_TLS=shm` or `rc_verbs`) further validate that our print logic resides in the correct place: the program aborted *before* or *during* the connection establishment phase where our "Line 2" print would occur, or printed error messages indicating unreachable destinations. This confirms that transport selection is a critical, early-stage runtime decision. Finally, the call flow is verified. The `ucp_worker_print_used_tls` function is indeed invoked during the initialization of endpoint configurations (`cfg#0`, `cfg#1`), matching the architectural understanding that resources are bound when connections (endpoints) are established.

- lanes: In UCP, an endpoint is composed of multiple "lanes". Each lane represents a logical channel mapped to a specific underlying transport resource capability. For example, our `UCX_TLS=all` output showed `cma/memory` and `sysv/memory` potentially occupying different lanes or being selected for different message sizes (e.g., one for small messages, one for large). Lanes allow UCP to stripe data or separate control/data paths.

- tl_rsc (Transport Layer Resource): This structure stores metadata about a specific physical or logical transport resource available to the context. It includes the resource's index and performance characteristics. In our experiments, `ibp3s0:1` (InfiniBand) and `memory` (Shared Memory) are distinct resources identified by `tl_rsc`.

- tl_name: This string identifies the name of the specific transport protocol or module being used. Our logs explicitly showed names like `ud_verbs` (Unreliable Datagram), `rc_verbs` (Reliable Connected), `cma` (Cross Memory Attach), and `sysv` (System V Shared Memory). It defines "how" the data is transmitted.

- tl_device: This refers to the name of the network interface or device associated with the transport resource. In our output, `ibp3s0:1` identifies the physical IB device, while `memory` identifies the system RAM used for shared memory transports.

- bitmap: This is a bitmask used to track usage or availability of resources/lanes. For instance, when we tried `UCX_TLS=rc_verbs`, the failure `no auxiliary transport` implies that the bitmap for available transports to establish connections (like UD for wireup) was empty or didn't match the requirements, preventing the connection.

- iface (Interface): This represents an instantiated communication interface on a worker. It holds the operational state (queues, credits) for a specific transport. When `UCX_TLS=tcp` failed with `no usable transports`, it implies that UCP could not create a valid `iface` for the TCP transport on the given network device.

## 3. Optimize System 
1. Below are the current configurations for OpenMPI and UCX in the system. Based on your learning, what methods can you use to optimize single-node performance by setting UCX environment variables?

```
-------------------------------------------------------------------
/opt/modulefiles/openmpi/ucx-pp:


module load     openmpi/5.0.8
prepend-path    PATH $ucx_prefix/bin
prepend-path    LD_LIBRARY_PATH $ucx_prefix/lib
setenv          UCX_TLS ud_verbs
setenv          UCX_NET_DEVICES ibp3s0:1
-------------------------------------------------------------------
```

1. Please use the following commands to test different data sizes for latency and bandwidth, to verify your ideas:
```bash
module load openmpi/ucx-pp
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_latency
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_bw
```
2. Please create a chart to illustrate the impact of different parameter options on various data sizes and the effects of different testsuite.
3. Based on the chart, explain the impact of different TLS implementations and hypothesize the possible reasons (references required).

The current configuration (`setenv UCX_TLS ud_verbs`) forces UCX to use the InfiniBand network transport (`ud_verbs`) even for intra-node communication. This is inefficient because data must travel through the PCIe bus to the Network Interface Card (NIC) and back (loopback), incurring significant latency and limited bandwidth. To optimize single-node performance, we should enable **Shared Memory** transport mechanisms. This allows processes on the same node to exchange data directly through system RAM (using technologies like Cross Memory Attach or SysV shared memory), bypassing the hardware network stack entirely. We can achieve this by setting the `UCX_TLS` environment variable to: 1) **`all`**: This allows UCX to automatically select the best available transport. For intra-node communication, it prioritizes shared memory (e.g., `shm`, `cma`, `self`) while falling back to network transports for inter-node communication; 2) **`shm,ud_verbs`**: Explicitly enabling shared memory transports alongside network verbs.

<img width="3600" height="2100" alt="rma_put_latency" src="https://github.com/user-attachments/assets/6389a2cf-7987-4485-9d06-624ff3fd6aa9" />
<img width="3600" height="2100" alt="rma_put_bandwidth" src="https://github.com/user-attachments/assets/0049a58a-7a6d-4ad3-bee1-2740f58c5aeb" />
<img width="3600" height="2100" alt="rma_get_latency" src="https://github.com/user-attachments/assets/d29ff9b0-2269-4832-96d8-d6838d4abf06" />
<img width="3600" height="2100" alt="pt2pt_latency" src="https://github.com/user-attachments/assets/faf76a01-d9c4-41d0-a06e-fb3be63db2c9" />
<img width="3600" height="2100" alt="pt2pt_bibandwidth" src="https://github.com/user-attachments/assets/bc74fde6-10a7-4a58-b746-bf380a26d74a" />
<img width="3600" height="2100" alt="pt2pt_bandwidth" src="https://github.com/user-attachments/assets/f0d38255-5ce9-49e1-902b-ba3b639297c0" />

**Analysis of Latency:**
Based on the figure "Point-to-Point Latency Comparison", we observe a dramatic difference in performance for small messages.
- **Shared Memory (Blue Line)**: Achieves extremely low latency. This confirms that bypassing the PCIe/NIC path significantly reduces overhead.
- **UD Verbs (Red Line)**: Shows much higher latency. This latency is consistent whether running on a single node or across multiple nodes (Green Dotted Line), confirming that `ud_verbs` treats local loopback almost identically to remote transmission—both involve the HCA hardware.
- **Hypothesis**: The latency reduction with Shared Memory is due to the elimination of hardware transaction overhead. `ud_verbs` requires generating a Work Queue Element (WQE), posting it to the HCA, and waiting for a completion event, whereas `shm` uses CPU instructions (`memcpy` or similar) to move data directly in memory.

**Analysis of Bandwidth:**
Based on figure "Bandwidth Comparison", Shared Memory outperforms `ud_verbs` significantly for large messages.
- **Shared Memory**: Peaks at over 10,000 MB/s.
- **UD Verbs**: Saturates around 2,500 MB/s.
- **Hypothesis**: The bandwidth of `ud_verbs` is limited by the physical link speed of the InfiniBand interface (e.g., QDR/FDR) or the PCIe bandwidth allocated to the HCA. Shared Memory bandwidth, conversely, is bounded by the system's memory bandwidth (DRAM speed), which is typically much higher than the external network interface speed.
- **Supplement (Bidirectional)**: The "Bidirectional Bandwidth Comparison" figure further amplifies this gap. While `ud_verbs` bidirectional throughput (~2,500 MB/s) barely exceeds its unidirectional performance (indicating a bottleneck at the PCIe/HCA link), Shared Memory scales impressively to nearly 17,500 MB/s. This demonstrates the full-duplex efficiency of memory buses compared to the hardware limitations of the network loopback.

**Effect of Different Test Suites (Pt2Pt vs. RMA):**
Comparing "Point-to-Point Latency Comparison" with "RMA Put/Get Latency Comparison" reveals a fascinating distinction between two-sided and one-sided communication.
- **In Two-sided Communication (Pt2Pt)**: There is a massive performance gap between Shared Memory and UD Verbs for small messages. This is because standard MPI Send/Recv involves "matching" overhead (software processing at the receiver to match tag/source) and, in the case of UD Verbs, a full round-trip through the HCA hardware loopback.
- **In One-sided Communication (RMA)**: Surprisingly, the latency of `ud_verbs` (Red Line) drops significantly, becoming almost indistinguishable from Shared Memory (Blue Line) for small messages. This suggests that for RMA Put/Get operations, UCX can offload the operation entirely to the hardware (RDMA) or optimize the loopback path to bypass the expensive software matching and interrupt overhead required for two-sided messaging. The HCA can directly write/read memory without involving the remote CPU, making `ud_verbs` extremely efficient even in a loopback scenario.

Therefore, while Shared Memory is consistently superior for standard MPI messaging, the advantage diminishes for RMA operations on small data because modern interconnect hardware (and UCX's optimization of it) handles one-sided loopback extremely efficiently.

### Advanced Challenge: Multi-Node Testing

This challenge involves testing the performance across multiple nodes. You can accomplish this by utilizing the sbatch script provided below. The task includes creating tables and providing explanations based on your findings. Notably, Writing a comprehensive report on this exercise can earn you up to 5 additional points.

- For information on sbatch, refer to the documentation at [Slurm's sbatch page](https://slurm.schedmd.com/sbatch.html).
- To conduct multi-node testing, use the following command:
```
cd ~/UCX-lsalab/test/
sbatch run.batch
```

As observed in the "Point-to-Point Latency Comparison", the latency for `ud_verbs` is nearly identical whether communicating within a single node (Red Line) or across multiple nodes (Green Line). This occurs because `ud_verbs` relies on Hardware Offload. Even for a local loopback, the data path involves: CPU -> PCIe Bus -> HCA (Network Card) -> Internal Loopback -> PCIe Bus -> CPU. The primary latency cost comes from the PCIe traversal and HCA processing, not the physical wire transmission. Since both single-node loopback and multi-node transmission incur these same hardware overheads, their performance is nearly indistinguishable for small messages. In the RMA (One-sided) charts, we observe an anomaly: Single-node `ud_verbs` (Red Line) becomes extremely fast, matching Shared Memory performance, while Multi-node `ud_verbs` (Green Line) remains slow. This indicates a specific optimization for **Intra-node RMA**. When UCX or the driver detects that the target memory address for an RMA operation resides on the local host, it "short-circuits" the operation. Instead of sending the request through the expensive PCIe/HCA path described above, it effectively performs a direct memory copy (similar to `memcpy`). For cross-node RMA, this short-circuit is impossible. The request must physically traverse the network to reach the remote memory, thus incurring the full hardware latency. This explains the large gap between the Red and Green lines in RMA charts, which is absent in P2P charts. Across all test suites (P2P and RMA), the Shared Memory transport (Blue Line) consistently delivers the lowest latency. SHM bypasses the network hardware stack entirely. It uses CPU instructions or kernel mechanisms to copy data directly between process memory spaces within the main RAM. By avoiding the high-latency PCIe bus and HCA command processing required by `ud_verbs`, Shared Memory represents the theoretical performance limit for intra-node communication.

## 4. Experience & Conclusion
1. What have you learned from this homework?
2. How long did you spend on the assignment?
3. Feedback (optional)

In this assignment, we gained deep insights into the architecture of high-performance communication frameworks. We learned how strongly software performance is coupled with underlying hardware mechanisms—specifically, how **Shared Memory** bypasses the PCIe bottleneck for intra-node speed, and how **Hardware Offload (Verbs)** incurs fixed overheads regardless of physical distance. Tracing the UCX codebase revealed the immense complexity behind its "simple" APIs; we witnessed how `ucp_worker` dynamically discovers resources and selects the optimal transport path (e.g., `shm` vs `ud_verbs`) during endpoint creation to maximize hardware utilization. The entire weekend is spent to be familiarized with the codebase and debugging the multi-node environment. This was arguably the most challenging assignment in the course. The UCX source code is vast and intricate, making navigation difficult without prior experience. The specification for the implementation part was somewhat ambiguous, and the hints provided were minimal relative to the scale of the codebase. A more guided tour of the critical data structures (`ucp_ep`, `ucp_worker`) would have been very helpful.

