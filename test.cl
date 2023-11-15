typedef struct TestResults {
  atomic_uint seq0;
  atomic_uint seq1;
  atomic_uint interleaved0;
  atomic_uint interleaved1;
  atomic_uint racy0;
  atomic_uint not_bound0;
  atomic_uint not_bound1;
  atomic_uint not_bound2;
  atomic_uint other;
} TestResults;

typedef struct Params {
  uint testing_workgroups;
  uint permute_id;
  uint mem_stride;
} Params;

static uint permute_id(uint id, uint factor, uint mask) {
  return (id * factor) % mask;
}

static uint stripe_workgroup(uint workgroup_id, uint local_id, uint testing_workgroups) {
  return (workgroup_id + 1 + local_id % (testing_workgroups - 1)) % testing_workgroups;
}

__kernel void run_test (
  __global uint* non_atomic_test_locations,
  __global atomic_uint* atomic_test_locations,
  __global TestResults* test_results,
  __global uint* shuffled_workgroups,
  __global Params* params) {
  uint shuffled_workgroup = shuffled_workgroups[get_group_id(0)];
  if(shuffled_workgroup < params->testing_workgroups) {
    uint total_ids = get_local_size(0) * params->testing_workgroups;
    uint id_0 = shuffled_workgroup * get_local_size(0) + get_local_id(0);
    uint new_workgroup = stripe_workgroup(shuffled_workgroup, get_local_id(0), params->testing_workgroups);
    uint id_1 = new_workgroup * get_local_size(0) + permute_id(get_local_id(0), params->permute_id, get_local_size(0));
    uint x_0 = (id_0) * params->mem_stride; // used to write to the racy location and write the flag (thread 0)
    uint x_1 = (id_1) * params->mem_stride; // used to write to the racy location, read the flag, first read of racy location (thread 1)
    uint y_1 = (permute_id(id_1, 1, total_ids)) * params->mem_stride; // aliased second read of racy location (thread 1)
    // Thread 0
    non_atomic_test_locations[x_0] = 1;
    atomic_store_explicit(&atomic_test_locations[x_0], 1, memory_order_release);

    // Thread 1
    uint flag = atomic_load_explicit(&atomic_test_locations[x_1], memory_order_acquire);
    while (flag == 0) {
      flag =  atomic_load_explicit(&atomic_test_locations[x_1], memory_order_acquire);
    }

    uint r0 = non_atomic_test_locations[x_1];
    uint r1 = non_atomic_test_locations[y_1]; 

    if (flag == 1 && r0 == 1 && r1 == 1) {
      atomic_fetch_add(&test_results->seq0, 1);
    } else if (flag == 0 && r0 == 0 && r1 == 0) {
      atomic_fetch_add(&test_results->seq1, 1);
    } else if (flag == 0 && r0 == 1 & r1 == 1) {
      atomic_fetch_add(&test_results->interleaved0, 1);
    } else if (flag == 0 && r0 == 0 & r1 == 1) {
      atomic_fetch_add(&test_results->interleaved1, 1);
    } else if (flag == 0 && r0 == 1 & r1 == 0) {
      atomic_fetch_add(&test_results->racy0, 1);
    } else if (flag == 1 && r0 == 0 & r1 == 0) {
      atomic_fetch_add(&test_results->not_bound0, 1);
    } else if (flag == 1 && r0 == 0 & r1 == 1) {
      atomic_fetch_add(&test_results->not_bound1, 1);
    } else if (flag == 1 && r0 == 1 & r1 == 0) {
      atomic_fetch_add(&test_results->not_bound2, 1);
    } else {
      atomic_fetch_add(&test_results->other, 1);
    }
  }
}
