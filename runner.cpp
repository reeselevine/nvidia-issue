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
const int WORKGROUP_SIZE = 185;
const int TESTING_WORKGROUPS = 164;
const int MAX_WORKGROUPS = 988;
const int MEM_STRIDE = 7;
const int PERMUTE_ID = 419;

const int NUM_OUTPUTS = 3;
const int NUM_RESULTS = 9;

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

/** Assigns shuffled workgroup ids, using the shufflePct to determine whether the ids should be shuffled this iteration. */
void setShuffledWorkgroups(Buffer &shuffledWorkgroups, int numWorkgroups) {
  for (int i = 0; i < numWorkgroups; i++) {
    shuffledWorkgroups.store<uint32_t>(i, i);
  }
  for (int i = numWorkgroups - 1; i > 0; i--) {
    int swap = rand() % (i + 1);
    int temp = shuffledWorkgroups.load<uint32_t>(i);
    shuffledWorkgroups.store<uint32_t>(i, shuffledWorkgroups.load<uint32_t>(swap));
    shuffledWorkgroups.store<uint32_t>(swap, temp);
  }
}

/** These parameters are static for all iterations of the test. */
void setStressParams(Buffer &stressParams) {
  stressParams.store<uint32_t>(0, TESTING_WORKGROUPS);
  stressParams.store<uint32_t>(1, PERMUTE_ID);
  stressParams.store<uint32_t>(2, MEM_STRIDE);
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
  auto stressParams = Buffer(device, 3, sizeof(uint32_t));
  setStressParams(stressParams);
  buffers.push_back(stressParams);

  // run iterations
  int numViolations = 0;
  for (int i = 0; i < TEST_ITERATIONS; i++) {
    auto program = Program(device, "test.spv", buffers);

    clearMemory(nonAtomicTestLocations, testLocSize);
    clearMemory(atomicTestLocations, testLocSize);
    clearMemory(testResults, NUM_RESULTS);
    setShuffledWorkgroups(shuffledWorkgroups, MAX_WORKGROUPS);

    program.setWorkgroups(MAX_WORKGROUPS);
    program.setWorkgroupSize(WORKGROUP_SIZE);

    program.initialize("run_test");
    program.run();

    cout << "Iteration " << i << "\n";
    cout << "flag=1, r0=2, r1=2 (seq): " << testResults.load<uint32_t>(0) << "\n";
    cout << "flag=0, r0=2, r1=2 (seq): " << testResults.load<uint32_t>(1) << "\n";
    cout << "flag=1, r0=1, r1=1 (interleaved): " << testResults.load<uint32_t>(2) << "\n";
    cout << "flag=0, r0=1, r1=1 (interleaved): " << testResults.load<uint32_t>(3) << "\n";
    cout << "flag=0, r0=2, r1=1 (racy): " << testResults.load<uint32_t>(4) << "\n";
    cout << "flag=0, r0=1, r1=2 (racy): " << testResults.load<uint32_t>(5) << "\n";
    cout << "flag=1, r0=2, r1=1 (not bound): " << testResults.load<uint32_t>(6) << "\n";
    cout << "flag=1, r0=1, r1=2 (not bound): " << testResults.load<uint32_t>(7) << "\n";
    cout << "Other/error: " << testResults.load<uint32_t>(8) << "\n\n";
    numViolations += testResults.load<uint32_t>(6) + testResults.load<uint32_t>(7) + testResults.load<uint32_t>(8);

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
