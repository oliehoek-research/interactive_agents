'''Testing of Ray capabilities'''
import ray
from ray.exceptions import TaskCancelledError
import time


@ray.remote
def test_fn_1():
    print("Test Function")


@ray.remote
def test_fn_2(value):
    print(f"Test Function {value}")


@ray.remote
def test_fn_3(value):
    return value + 1


@ray.remote
def test_fn_4():
    time.sleep(1e12)
    return True


@ray.remote
class TestClass:

    def __init__(self):
        self._counter = 0

    def increment(self):
        self._counter += 1
        return self._counter


if __name__ == "__main__":
    ray.init(num_cpus=2, num_gpus=0)

    # value = 0
    # for i in range(4):
    #     test_fn_2.remote(i)
    #     # value = test_fn_3.remote(value)

    # print(f"Final value: {ray.get(value)}")

    # output = test_fn_4.remote()
    # ray.cancel(output)

    # try:
    #     print(f"output: {ray.get(output)}")
    # except TaskCancelledError:
    #     print("ERROR: Task cancelled before it completed")

    counter = TestClass.remote()

    values = []
    for _ in range(10):
        values.append(counter.increment.remote())

    for value in values:
        print(f"count: {ray.get(value)}")
