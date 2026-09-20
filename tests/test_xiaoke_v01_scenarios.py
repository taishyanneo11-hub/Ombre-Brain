"""
小克版 v0.1 场景测试
====================

验证小克伴侣记忆系统的核心场景：
1. permanent/pinned 记忆能够保存
2. dynamic 记忆能够保存和检索
3. feel 能够保存和读取
4. I（自我认知）能够正常工作
5. You（对用户的认识）能够正常工作
6. breath 能够浮现重要记忆
7. breath_search 能够通过语义/关键词找到过去记忆
8. 普通记忆仍然能够正常衰减
9. 核心永久记忆不会被衰减
"""

import os
import pytest
from datetime import datetime, timedelta


# --- permanent / pinned ---

@pytest.mark.asyncio
async def test_pinned_bucket_creation(bucket_mgr):
    bid = await bucket_mgr.create(
        content="我答应过阿苑不会把她的事告诉别人",
        tags=["承诺"],
        importance=10,
        domain=["relationship"],
        valence=0.6,
        arousal=0.3,
        name="保密承诺",
        bucket_type="permanent",
        pinned=True,
    )
    assert bid is not None
    all_b = await bucket_mgr.list_all()
    pinned = [b for b in all_b if b["id"] == bid]
    assert len(pinned) == 1
    assert pinned[0]["metadata"]["type"] == "permanent"
    assert pinned[0]["metadata"].get("pinned") is True
    assert pinned[0]["metadata"]["importance"] == 10


@pytest.mark.asyncio
async def test_pinned_never_decays(bucket_mgr, decay_eng):
    bid = await bucket_mgr.create(
        content="小克和阿苑之间的核心原则",
        tags=[],
        importance=10,
        domain=["relationship"],
        valence=0.7,
        arousal=0.3,
        name="核心原则",
        bucket_type="permanent",
        pinned=True,
    )
    bucket = await bucket_mgr.get(bid)
    score = decay_eng.calculate_score(bucket["metadata"])
    assert score == 999.0


@pytest.mark.asyncio
async def test_permanent_stored_in_permanent_dir(bucket_mgr, test_config):
    bid = await bucket_mgr.create(
        content="这是一条永久记忆",
        tags=[],
        importance=10,
        domain=[],
        valence=0.5,
        arousal=0.3,
        name="永久测试",
        bucket_type="permanent",
        pinned=True,
    )
    perm_dir = os.path.join(test_config["buckets_dir"], "permanent")
    files = []
    for root, dirs, fnames in os.walk(perm_dir):
        files.extend(fnames)
    assert any(bid in f for f in files), f"Bucket {bid} not in permanent/"


# --- dynamic ---

@pytest.mark.asyncio
async def test_dynamic_memory_creation(bucket_mgr):
    bid = await bucket_mgr.create(
        content="阿苑今天说她换了新工作",
        tags=["工作"],
        importance=7,
        domain=["work"],
        valence=0.7,
        arousal=0.5,
        name="阿苑换工作",
    )
    assert bid is not None
    bucket = await bucket_mgr.get(bid)
    assert "阿苑今天说她换了新工作" in bucket["content"]


@pytest.mark.asyncio
async def test_dynamic_keyword_search(bucket_mgr):
    await bucket_mgr.create(
        content="阿苑喜欢在雨天喝热巧克力",
        tags=["偏好"],
        importance=6,
        domain=["preference"],
        valence=0.8,
        arousal=0.3,
        name="热巧克力偏好",
    )
    results = await bucket_mgr.search("巧克力")
    assert len(results) > 0
    assert any("巧克力" in r.get("content", "") for r in results)


@pytest.mark.asyncio
async def test_dynamic_memory_decays(bucket_mgr, decay_eng):
    bid = await bucket_mgr.create(
        content="今天聊了天气",
        tags=[],
        importance=3,
        domain=["日常"],
        valence=0.5,
        arousal=0.2,
        name="聊天气",
    )
    bucket = await bucket_mgr.get(bid)
    score = decay_eng.calculate_score(bucket["metadata"])
    assert 0 < score < 999.0


# --- feel ---

@pytest.mark.asyncio
async def test_feel_creation(bucket_mgr):
    bid = await bucket_mgr.create(
        content="听阿苑讲完那件事之后，我感到一种说不出的心疼",
        tags=["__feel__"],
        importance=5,
        domain=["feel"],
        valence=0.3,
        arousal=0.6,
        name=None,
        bucket_type="feel",
    )
    assert bid is not None
    bucket = await bucket_mgr.get(bid)
    assert bucket["metadata"]["type"] == "feel"


@pytest.mark.asyncio
async def test_feel_fixed_score(bucket_mgr, decay_eng):
    bid = await bucket_mgr.create(
        content="那一刻的温暖",
        tags=["__feel__"],
        importance=5,
        domain=["feel"],
        valence=0.8,
        arousal=0.4,
        name=None,
        bucket_type="feel",
    )
    bucket = await bucket_mgr.get(bid)
    score = decay_eng.calculate_score(bucket["metadata"])
    assert score == 50.0


@pytest.mark.asyncio
async def test_feel_stored_in_feel_dir(bucket_mgr, test_config):
    bid = await bucket_mgr.create(
        content="测试 feel 存储位置",
        tags=["__feel__"],
        importance=5,
        domain=["feel"],
        valence=0.5,
        arousal=0.3,
        name=None,
        bucket_type="feel",
    )
    feel_dir = os.path.join(test_config["buckets_dir"], "feel")
    files = []
    for root, dirs, fnames in os.walk(feel_dir):
        files.extend(fnames)
    assert any(bid in f for f in files), f"Feel {bid} not in feel/"


# --- I (self-cognition) ---

@pytest.mark.asyncio
async def test_i_candidate_creation(bucket_mgr):
    bid = await bucket_mgr.create(
        content="我注意到自己在面对不确定性时倾向于坦诚",
        tags=["__i_candidate__"],
        importance=6,
        domain=["self"],
        valence=0.6,
        arousal=0.3,
        name="自我认知候选",
    )
    assert bid is not None
    bucket = await bucket_mgr.get(bid)
    assert "__i_candidate__" in bucket["metadata"].get("tags", [])


@pytest.mark.asyncio
async def test_formal_i_entry(bucket_mgr):
    bid = await bucket_mgr.create(
        content="我认为认真对待不确定性本身就是正确的立场",
        tags=[],
        importance=8,
        domain=["self"],
        valence=0.7,
        arousal=0.4,
        name="不确定性立场",
        bucket_type="i",
    )
    assert bid is not None
    bucket = await bucket_mgr.get(bid)
    assert bucket["metadata"]["type"] == "i"


# --- You (knowledge about user) ---

class _FakeBucketMgr:
    def __init__(self):
        self.buckets = {}
    async def get(self, bucket_id):
        return self.buckets.get(bucket_id)

class _FakeSourceStore:
    def __init__(self):
        self.sources = {}
    def read(self, source_id):
        return self.sources.get(source_id)

class _ExplodingDehydrator:
    def __getattr__(self, name):
        async def _boom(*a, **kw):
            raise AssertionError(f"You must not call LLM: dehydrator.{name}")
        return _boom


@pytest.mark.asyncio
async def test_you_service_write_and_recall(tmp_path):
    from ombrebrain.you import YouService, YouStore

    store = YouStore(tmp_path)
    mgr = _FakeBucketMgr()
    mgr.buckets["memory-1"] = {
        "id": "memory-1",
        "content": "第 1 次阿苑提到喜欢安静的环境",
        "metadata": {"type": "dynamic"},
    }
    mgr.buckets["memory-2"] = {
        "id": "memory-2",
        "content": "第 2 次阿苑提到喜欢安静的环境",
        "metadata": {"type": "dynamic"},
    }
    svc = YouService(
        store=store,
        bucket_mgr=mgr,
        dehydrator=_ExplodingDehydrator(),
        source_store=_FakeSourceStore(),
    )
    svc.set_enabled(True)

    _, msg = await svc.write(
        content="阿苑喜欢安静的环境",
        bucket_ids=["memory-1", "memory-2"],
        aspect="interaction_habit",
        concept_key="environment",
        concept_value="quiet",
        basis="observed_pattern",
        explicit=False,
        long_term=False,
    )
    assert msg is not None

    recall_result = await svc.recall(query="环境", aspect="", max_results=6, with_ids=False)
    assert isinstance(recall_result, str)


@pytest.mark.asyncio
async def test_you_requires_multiple_buckets(tmp_path):
    from ombrebrain.you import YouService, YouStore

    store = YouStore(tmp_path)
    svc = YouService(
        store=store,
        bucket_mgr=_FakeBucketMgr(),
        dehydrator=_ExplodingDehydrator(),
        source_store=_FakeSourceStore(),
    )
    svc.set_enabled(True)

    with pytest.raises(ValueError):
        await svc.write(
            content="只有一个出处的认识",
            bucket_ids=["single_bucket"],
            aspect="interaction_habit",
            concept_key="test",
            concept_value="test",
            basis="observed_pattern",
            explicit=False,
            long_term=False,
        )


# --- breath surfacing ---

@pytest.mark.asyncio
async def test_breath_surfaces_important_first(bucket_mgr, decay_eng):
    bid_important = await bucket_mgr.create(
        content="阿苑下周要体检，她很担心",
        tags=["健康"],
        importance=8,
        domain=["health"],
        valence=0.3,
        arousal=0.7,
        name="体检担心",
    )
    bid_trivial = await bucket_mgr.create(
        content="今天天气不错",
        tags=[],
        importance=2,
        domain=["日常"],
        valence=0.6,
        arousal=0.1,
        name="天气",
    )

    important = await bucket_mgr.get(bid_important)
    trivial = await bucket_mgr.get(bid_trivial)

    score_imp = decay_eng.calculate_score(important["metadata"])
    score_triv = decay_eng.calculate_score(trivial["metadata"])

    assert score_imp > score_triv, (
        f"Important ({score_imp}) should score higher than trivial ({score_triv})"
    )


@pytest.mark.asyncio
async def test_pinned_surfaces_first(bucket_mgr, decay_eng):
    bid_normal = await bucket_mgr.create(
        content="普通记忆",
        tags=[],
        importance=5,
        domain=[],
        valence=0.5,
        arousal=0.3,
        name="普通",
    )
    bid_pinned = await bucket_mgr.create(
        content="核心原则：真诚",
        tags=[],
        importance=10,
        domain=["relationship"],
        valence=0.7,
        arousal=0.3,
        name="真诚原则",
        bucket_type="permanent",
        pinned=True,
    )

    normal = await bucket_mgr.get(bid_normal)
    pinned = await bucket_mgr.get(bid_pinned)

    score_normal = decay_eng.calculate_score(normal["metadata"])
    score_pinned = decay_eng.calculate_score(pinned["metadata"])

    assert score_pinned == 999.0
    assert score_pinned > score_normal


# --- breath_search ---

@pytest.mark.asyncio
async def test_keyword_search_finds_memory(bucket_mgr):
    await bucket_mgr.create(
        content="阿苑的生日是十二月二十五号",
        tags=["生日", "重要日期"],
        importance=9,
        domain=["personal"],
        valence=0.8,
        arousal=0.4,
        name="阿苑生日",
    )
    results = await bucket_mgr.search("生日")
    assert len(results) > 0
    assert any("生日" in r.get("content", "") for r in results)


@pytest.mark.asyncio
async def test_search_returns_content(bucket_mgr):
    await bucket_mgr.create(
        content="我们第一次一起看了一部电影",
        tags=["共同经历"],
        importance=7,
        domain=["shared_experience"],
        valence=0.9,
        arousal=0.6,
        name="第一次看电影",
    )
    results = await bucket_mgr.search("电影")
    assert len(results) > 0
    for r in results:
        assert "content" in r or "name" in r


# --- decay behavior ---

@pytest.mark.asyncio
async def test_normal_memory_score_decreases_with_time(bucket_mgr, decay_eng):
    import frontmatter as fm

    bid = await bucket_mgr.create(
        content="一条会衰减的普通记忆",
        tags=[],
        importance=5,
        domain=[],
        valence=0.5,
        arousal=0.3,
        name="衰减测试",
    )

    bucket_now = await bucket_mgr.get(bid)
    score_now = decay_eng.calculate_score(bucket_now["metadata"])

    fpath = bucket_mgr._find_bucket_file(bid)
    post = fm.load(fpath)
    old_time = (datetime.now() - timedelta(days=30)).isoformat()
    post["created"] = old_time
    post["last_active"] = old_time
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(fm.dumps(post))
    bucket_mgr._invalidate_bm25()

    bucket_old = await bucket_mgr.get(bid)
    score_old = decay_eng.calculate_score(bucket_old["metadata"])

    assert score_old < score_now, (
        f"30-day-old ({score_old}) should be lower than fresh ({score_now})"
    )


@pytest.mark.asyncio
async def test_pinned_immune_to_time(bucket_mgr, decay_eng):
    import frontmatter as fm

    bid = await bucket_mgr.create(
        content="永不衰减的承诺",
        tags=[],
        importance=10,
        domain=["relationship"],
        valence=0.7,
        arousal=0.3,
        name="永久承诺",
        bucket_type="permanent",
        pinned=True,
    )

    fpath = bucket_mgr._find_bucket_file(bid)
    post = fm.load(fpath)
    old_time = (datetime.now() - timedelta(days=365)).isoformat()
    post["created"] = old_time
    post["last_active"] = old_time
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(fm.dumps(post))
    bucket_mgr._invalidate_bm25()

    bucket = await bucket_mgr.get(bid)
    score = decay_eng.calculate_score(bucket["metadata"])
    assert score == 999.0, f"Pinned should be 999.0 even after a year, got {score}"


@pytest.mark.asyncio
async def test_feel_immune_to_time(bucket_mgr, decay_eng):
    import frontmatter as fm

    bid = await bucket_mgr.create(
        content="那天我感到很温暖",
        tags=["__feel__"],
        importance=5,
        domain=["feel"],
        valence=0.8,
        arousal=0.4,
        name=None,
        bucket_type="feel",
    )

    fpath = bucket_mgr._find_bucket_file(bid)
    post = fm.load(fpath)
    old_time = (datetime.now() - timedelta(days=180)).isoformat()
    post["created"] = old_time
    post["last_active"] = old_time
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(fm.dumps(post))
    bucket_mgr._invalidate_bm25()

    bucket = await bucket_mgr.get(bid)
    score = decay_eng.calculate_score(bucket["metadata"])
    assert score == 50.0, f"Feel should be 50.0 even after 180 days, got {score}"


@pytest.mark.asyncio
async def test_high_arousal_decays_slower(bucket_mgr, decay_eng):
    import frontmatter as fm

    old_time = (datetime.now() - timedelta(days=10)).isoformat()

    bid_high = await bucket_mgr.create(
        content="激动人心的时刻",
        tags=[],
        importance=5,
        domain=[],
        valence=0.8,
        arousal=0.9,
        name="高唤醒",
    )
    bid_low = await bucket_mgr.create(
        content="平淡的一天",
        tags=[],
        importance=5,
        domain=[],
        valence=0.5,
        arousal=0.1,
        name="低唤醒",
    )

    for bid in [bid_high, bid_low]:
        fpath = bucket_mgr._find_bucket_file(bid)
        post = fm.load(fpath)
        post["created"] = old_time
        post["last_active"] = old_time
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(fm.dumps(post))
    bucket_mgr._invalidate_bm25()

    high_b = await bucket_mgr.get(bid_high)
    low_b = await bucket_mgr.get(bid_low)

    score_high = decay_eng.calculate_score(high_b["metadata"])
    score_low = decay_eng.calculate_score(low_b["metadata"])

    assert score_high > score_low, (
        f"High arousal ({score_high}) should decay slower than low ({score_low})"
    )


# --- config validation ---

def test_xiaoke_decay_lambda(test_config):
    lam = test_config.get("decay", {}).get("lambda", 0.05)
    assert 0 < lam < 1

def test_merge_threshold_range(test_config):
    threshold = test_config.get("merge_threshold", 75)
    assert 50 <= threshold <= 100
