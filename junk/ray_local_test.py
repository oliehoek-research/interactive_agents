"""Tests ability to use ray actors synchronously"""
import ray


class TestActor:

    def test(self):
        print("hello world")


if __name__ == "__main__":
    local_actor = TestActor()
    local_actor.test()

    ray.init()

    remote_actor = ray.remote(num_cpus=1)(TestActor).remote()
    remote_actor.test.remote()
