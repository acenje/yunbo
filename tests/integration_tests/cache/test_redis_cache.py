"""Test Redis cache functionality."""
import redis

import langchain
from langchain.cache import RedisCache, RedisSemanticCache
from langchain.schema import Generation, LLMResult
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings
from tests.unit_tests.llms.fake_llm import FakeLLM

REDIS_TEST_URL = "redis://localhost:6379"


def test_redis_cache() -> None:
    langchain.llm_cache = RedisCache(redis_=redis.Redis.from_url(REDIS_TEST_URL))
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    langchain.llm_cache.update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["foo"])
    print(output)
    expected_output = LLMResult(
        generations=[[Generation(text="fizz")]],
        llm_output={},
    )
    print(expected_output)
    assert output == expected_output
    langchain.llm_cache.redis.flushall()


def test_redis_semantic_cache() -> None:
    langchain.llm_cache = RedisSemanticCache(
        embedding=FakeEmbeddings(), redis_url=REDIS_TEST_URL, score_threshold=0.1
    )
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    langchain.llm_cache.update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(
        ["bar"]
    )  # foo and bar will have the same embedding produced by FakeEmbeddings
    expected_output = LLMResult(
        generations=[[Generation(text="fizz")]],
        llm_output={},
    )
    assert output == expected_output
    # clear the cache
    langchain.llm_cache.clear(llm_string=llm_string)
    output = llm.generate(
        ["bar"]
    )  # foo and bar will have the same embedding produced by FakeEmbeddings
    # expect different output now without cached result
    assert output != expected_output
    langchain.llm_cache.clear(llm_string=llm_string)
def _concat_same_type(cls, to_concat, axis=0):
        # NB: We are assuming at this point that dtypes all match
        new_values = [obj._ndarray for obj in to_concat]
        new_arr = cnp.PyArray_Concatenate(new_values, axis)
        return to_concat[0]._from_backing_data(new_arr)
