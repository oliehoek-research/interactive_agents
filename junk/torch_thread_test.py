import numpy as np
import traceback

import torch

from torch.multiprocessing import Manager, Pool  # NOTE: Seems that using the pool context manager fixed things, weird
# from multiprocessing import Manager, Pool

NUM_PROCESSES = 2
MAX_THREADS = 1

def print_error(error):
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)


def test_get_threads(pid, lock):
    # with lock:
    print(f"\nProcess {pid}:")
    print(f"    current intra-op threads: {torch.get_num_threads()}")
    print(f"    current inter-op threads: {torch.get_num_interop_threads()}\n")


def global_test():
    torch.set_num_threads(MAX_THREADS)
    torch.set_num_interop_threads(MAX_THREADS)

    # with Manager() as manager:
    #     lock = manager.Lock()

    with Pool(NUM_PROCESSES) as pool:
        processes = []
        for pid in range(NUM_PROCESSES):
            processes.append(pool.apply_async(test_get_threads, 
                (pid, None), error_callback=print_error))

        for process in processes:
            process.wait()


def test_set_threads(pid, lock, barrier):
    np.random.seed(pid + 1000)
    num_threads = np.random.randint(1, MAX_THREADS + 1)

    # Attempt to limit CPU parallelism within current process
    with lock:
        print(f"Process {pid} - setting limit to {num_threads} threads")
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)

    # Wait for all threads to write
    barrier.wait()

    # Print current number of threads
    with lock:
        print(f"\nProcess {pid} (thread limit {num_threads})")
        print(f"    current intra-op threads: {torch.get_num_threads()}")
        print(f"    current inter-op threads: {torch.get_num_interop_threads()}\n")


def local_test():
    with Manager() as manager:
        lock = manager.Lock()
        barrier = manager.Barrier(NUM_PROCESSES)

        with Pool(NUM_PROCESSES) as pool:
            processes = []
            for pid in range(NUM_PROCESSES):
                processes.append(pool.apply_async(test_set_threads, 
                    (pid, lock, barrier), error_callback=print_error))

            for process in processes:
                process.wait()


if __name__ == '__main__':
    global_test()
    # local_test()

