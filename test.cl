static uint permute_id(uint id, uint factor, uint mask) {
  return (id * factor) % mask;
}

static uint stripe_workgroup(uint workgroup_id, uint local_id, uint testing_workgroups) {
  return (workgroup_id + 1 + local_id % (testing_workgroups - 1)) % testing_workgroups;
}

static uint get_new_id(uint id) {
    return id;
}

__kernel void run_test (
  __global uint* non_atomic_test_locations,
  __global atomic_uint* atomic_test_locations,
  __global uint* shuffled_locations,
  __global atomic_uint* read_results) {

  uint id_0 = get_group_id(0) * get_local_size(0) + get_local_id(0);
  uint id_1 = shuffled_locations[id_0];
  uint x_0 = id_0 * 4;
  uint x_1 = id_1 * 4;
  uint y_1 = get_new_id(id_1) * 4;

  // Thread 0
  non_atomic_test_locations[x_0] = 1;
  atomic_store_explicit(&atomic_test_locations[x_0], 1, memory_order_release);

  // Thread 1
  non_atomic_test_locations[x_1] = 2;
  uint flag = atomic_load_explicit(&atomic_test_locations[x_1], memory_order_acquire);
  while(flag == 0) {
    flag = atomic_load_explicit(&atomic_test_locations[x_1], memory_order_acquire);
  }

  uint r0 = non_atomic_test_locations[x_1];
  uint r1 = non_atomic_test_locations[y_1]; 

  // Store back results for analysis
  atomic_store(&read_results[id_1 * 3], flag);
  atomic_store(&read_results[id_1 * 3 + 1], r0);
  atomic_store(&read_results[id_1 * 3 + 2], r1);
}
