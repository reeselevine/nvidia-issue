#include <map>
#include <set>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>
#include <easyvk.h>
#include <unistd.h>

using namespace std;
using namespace easyvk;

const int TEST_ITERATIONS = 100;
const int TESTING_WORKGROUPS = 164;
const int MAX_WORKGROUPS = 988;
const int WORKGROUP_SIZE = 185;
const int SHUFFLE_PCT = 82;
const int BARRIER_PCT = 89;
const int MEM_STRIDE = 7;
const int MEM_STRESS_PCT = 83;
const int MEM_STRESS_ITERATIONS = 800;
const int MEM_STRESS_PATTERN = 1;
const int PRE_STRESS_PCT = 86;
const int PRE_STRESS_ITERATIONS = 112;
const int PRE_STRESS_PATTERN = 3;
const int STRESS_ASSIGNMENT_STRATEGY = 0;
const int PERMUTE_THREAD = 419;

const int NUM_OUTPUTS = 3;
const int NUM_RESULTS = 9;
const int PERMUTE_LOCATION = 1;

/** Returns the GPU to use for this test run. Users can specify the specific GPU to use
 *  with the a device index parameter. If the index is too large, an error is returned.
 */
Device getDevice(Instance &instance, int device_idx) {
  Device device = Device(instance, instance.physicalDevices().at(device_idx));
  cout << "Using device " << device.properties.deviceName << "\n";
  return device;
}

void listDevices() {
  auto instance = Instance(false);
  int i = 0;
  for (auto physicalDevice : instance.physicalDevices()) {
    Device device = Device(instance, physicalDevice);
    cout << "Device: " << device.properties.deviceName << " ID: " << device.properties.deviceID << " Index: " << i << "\n";
    i++;
  }
}

/** Zeroes out the specified buffer. */
void clearMemory(Buffer &gpuMem, int size) {
  for (int i = 0; i < size; i++) {
    gpuMem.store<uint32_t>(i, 0);
  }
}

/** Checks whether a random value is less than a given percentage. Used for parameters like memory stress that should only
 *  apply some percentage of iterations.
 */
bool percentageCheck(int percentage) {
  return rand() % 100 < percentage;
}

/** Assigns shuffled workgroup ids, using the shufflePct to determine whether the ids should be shuffled this iteration. */
void setShuffledWorkgroups(Buffer &shuffledWorkgroups, int numWorkgroups, int shufflePct) {
  for (int i = 0; i < numWorkgroups; i++) {
    shuffledWorkgroups.store<uint32_t>(i, i);
  }
  if (percentageCheck(shufflePct)) {
    for (int i = numWorkgroups - 1; i > 0; i--) {
      int swap = rand() % (i + 1);
      int temp = shuffledWorkgroups.load<uint32_t>(i);
      shuffledWorkgroups.store<uint32_t>(i, shuffledWorkgroups.load<uint32_t>(swap));
      shuffledWorkgroups.store<uint32_t>(swap, temp);
    }
  }
}

/** These parameters vary per iteration, based on a given percentage. */
void setDynamicStressParams(Buffer &stressParams) {
  if (percentageCheck(BARRIER_PCT)) {
    stressParams.store<uint32_t>(0, 1);
  } else {
    stressParams.store<uint32_t>(0, 0);
  }  
  if (percentageCheck(MEM_STRESS_PCT)) {
    stressParams.store<uint32_t>(1, 1);
  } else {
    stressParams.store<uint32_t>(1, 0);
  }  
  if (percentageCheck(PRE_STRESS_PCT)) {
    stressParams.store<uint32_t>(4, 1);
  } else {
    stressParams.store<uint32_t>(4, 0);
  }
}

/** These parameters are static for all iterations of the test. Aliased memory is used for coherence tests. */
void setStaticStressParams(Buffer &stressParams) {
  stressParams.store<uint32_t>(2, MEM_STRESS_ITERATIONS);
  stressParams.store<uint32_t>(3, MEM_STRESS_PATTERN);
  stressParams.store<uint32_t>(5, PRE_STRESS_ITERATIONS);
  stressParams.store<uint32_t>(6, PRE_STRESS_PATTERN);
  stressParams.store<uint32_t>(7, PERMUTE_THREAD);
  stressParams.store<uint32_t>(8, PERMUTE_LOCATION);
  stressParams.store<uint32_t>(9, TESTING_WORKGROUPS);
  stressParams.store<uint32_t>(10, MEM_STRIDE);
}

/** Returns a value between the min and max. */
int setBetween(int min, int max) {
  if (min == max) {
    return min;
  } else {
    int size = rand() % (max - min);
    return min + size;
  }
}

/** A test consists of N iterations of a shader. */
void run(int device_id, bool enable_validation_layers)
{
  // initialize settings
  auto instance = Instance(enable_validation_layers);
  auto device = getDevice(instance, device_id);
  int testingThreads = WORKGROUP_SIZE * TESTING_WORKGROUPS;
  int testLocSize = testingThreads * MEM_STRIDE;

  // set up buffers
  vector<Buffer> buffers;
  auto nonAtomicTestLocations = Buffer(device, testLocSize, sizeof(uint32_t));
  buffers.push_back(nonAtomicTestLocations);
  auto atomicTestLocations = Buffer(device, testLocSize, sizeof(uint32_t));
  buffers.push_back(atomicTestLocations);
  auto testResults = Buffer(device, NUM_RESULTS, sizeof(uint32_t));
  buffers.push_back(testResults);
  auto shuffledWorkgroups = Buffer(device, MAX_WORKGROUPS, sizeof(uint32_t));
  buffers.push_back(shuffledWorkgroups);
  auto stressParams = Buffer(device, 11, sizeof(uint32_t));
  setStaticStressParams(stressParams);
  buffers.push_back(stressParams);


  // run iterations
  chrono::time_point<std::chrono::system_clock> start, end;
  start = chrono::system_clock::now();
  int numViolations = 0;
  for (int i = 0; i < TEST_ITERATIONS; i++) {
    auto program = Program(device, "test.spv", buffers);

    int numWorkgroups = setBetween(TESTING_WORKGROUPS, MAX_WORKGROUPS);
    clearMemory(nonAtomicTestLocations, testLocSize);
    clearMemory(atomicTestLocations, testLocSize);
    clearMemory(testResults, NUM_RESULTS);
    setShuffledWorkgroups(shuffledWorkgroups, numWorkgroups, SHUFFLE_PCT);
    setDynamicStressParams(stressParams);

    program.setWorkgroups(numWorkgroups);
    program.setWorkgroupSize(WORKGROUP_SIZE);

    program.initialize("run_test");
    program.run();

    cout << "Iteration " << i << "\n";
    vector<uint32_t> results;
    for (int i = 0; i < NUM_RESULTS; i++) {
      results.push_back(testResults.load<uint32_t>(i));
    }
    cout << "flag=1, r0=2, r1=2 (seq): " << results[0] << "\n";
    cout << "flag=0, r0=2, r1=2 (seq): " << results[1] << "\n";
    cout << "flag=1, r0=1, r1=1 (interleaved): " << results[2] << "\n";
    cout << "flag=0, r0=1, r1=1 (interleaved): " << results[3] << "\n";
    cout << "flag=0, r0=2, r1=1 (racy): " << results[4] << "\n";
    cout << "flag=0, r0=1, r1=2 (racy): " << results[5] << "\n";
    cout << "flag=1, r0=2, r1=1 (not bound): " << results[6] << "\n";
    cout << "flag=1, r0=1, r1=2 (not bound): " << results[7] << "\n";
    cout << "Other/error: " << results[8] << "\n\n";
    numViolations += results[6] + results[7] + results[8];

    program.teardown();
  }

  cout << "Number of violations: " << numViolations << "\n";

  for (Buffer buffer : buffers) {
    buffer.teardown();
  }
  device.teardown();
  instance.teardown();
}

int main(int argc, char *argv[])
{

  int deviceIndex = 0;
  bool enableValidationLayers = false;
  bool list_devices = false;

  int c;
  while ((c = getopt(argc, argv, "vld:")) != -1)
    switch (c)
    {
    case 'v':
      enableValidationLayers = true;
      break;
    case 'l':
      list_devices = true;
      break;
    case 'd':
      deviceIndex = atoi(optarg);
      break;
    case '?':
      std::cerr << "Unknown option" << optopt << "\n";
      return 1;
    default:
      abort();
    }

  if (list_devices) {
    listDevices();
    return 0;
  }

  srand(time(NULL));
  run(deviceIndex, enableValidationLayers);
  return 0;
}
