#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <random>
#include <chrono>
#include <easyvk.h>
#include <unistd.h>
#include <algorithm>

using namespace std;
using namespace easyvk;

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

void setShuffledLocations(Buffer &shuffledLocations, int testingThreads) {
  vector<uint> locs;
  for (int i = 0; i < testingThreads; i++) {
    locs.push_back(i);
  }
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  shuffle(locs.begin(), locs.end(), default_random_engine(seed));
  for (int i = 0; i < testingThreads; i++) {
      shuffledLocations.store(i, locs[i]);
  }
}

/** A test consists of N iterations of a shader and its corresponding result shader. */
void run(int device_id, bool enable_validation_layers)
{
  // initialize settings
  auto instance = Instance(enable_validation_layers);
  auto device = getDevice(instance, device_id);
  int workgroups = 512;
  int workgroupSize = 192;
  int testingThreads = workgroups * workgroupSize;
  int testLocSize = testingThreads * 7;

  // set up buffers
  auto nonAtomicTestLocations = Buffer(device, testLocSize, sizeof(uint32_t));
  auto atomicTestLocations = Buffer(device, testLocSize, sizeof(uint32_t));
  auto shuffledLocations = Buffer(device, testingThreads, sizeof(uint32_t));
  auto readResults = Buffer(device, 3 * testingThreads, sizeof(uint32_t));
  auto testResults = Buffer(device, 9, sizeof(uint32_t));

  vector<Buffer> buffers = {nonAtomicTestLocations, atomicTestLocations, shuffledLocations, readResults};
  vector<Buffer> resultBuffers = {readResults, testResults};

  int numViolations = 0;
  for (int i = 0; i < 1000; i++) {
    auto program = Program(device, "test.spv", buffers);
    auto resultProgram = Program(device, "check_results.spv", resultBuffers);

    clearMemory(nonAtomicTestLocations, testLocSize);
    clearMemory(atomicTestLocations, testLocSize);
    setShuffledLocations(shuffledLocations, testingThreads);
    clearMemory(testResults, 9);

    program.setWorkgroups(workgroups);
    resultProgram.setWorkgroups(workgroups);
    program.setWorkgroupSize(workgroupSize);
    resultProgram.setWorkgroupSize(workgroupSize);

    program.initialize("run_test");
//    cout << "Running program\n";
    program.run();
//    cout << "Done running program\n";
    resultProgram.initialize("check_results");
//    cout << "Running results\n";
    resultProgram.run();
//    cout << "Done running results\n";


    cout << "Iteration " << i << "\n";
    vector<uint32_t> results;
    for (int i = 0; i < 9; i++) {
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
    resultProgram.teardown();
  }

  cout << "Number of violations: " << numViolations << "\n";

  nonAtomicTestLocations.teardown();
  atomicTestLocations.teardown();
  readResults.teardown();
  testResults.teardown();
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
      if (optopt == 'd')
        std::cerr << "Option -" << optopt << "requires an argument\n";
      else
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
