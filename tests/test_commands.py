from deepresearch.commands.registry import dispatch


def test_passthrough_for_plain_text():
    r = dispatch("what is the capital of france")
    assert r.kind == "passthrough"
    assert r.command_used == "free_text"
    assert r.query == "what is the capital of france"


def test_help_is_pure():
    r = dispatch("/help")
    assert r.kind == "pure"
    assert r.command_used == "/help"
    assert "/tools" in r.output


def test_tools_lists_catalog():
    r = dispatch("/tools")
    assert r.kind == "pure"
    assert "web_search" in r.output
    assert "render_qa" in r.output


def test_why_requires_arg():
    r = dispatch("/why")
    assert r.kind == "pure"
    assert "usage" in r.output.lower()


def test_why_returns_ranking():
    r = dispatch("/why arxiv quantum computing papers")
    assert r.kind == "pure"
    assert "arxiv_search" in r.output


def test_research_expands_query():
    r = dispatch("/research transformer architectures")
    assert r.kind == "expand"
    assert r.command_used == "/research"
    assert "transformer architectures" in r.query
    assert "render_qa" in r.query


def test_compare_expands_query():
    r = dispatch("/compare NVDA vs AMD")
    assert r.kind == "expand"
    assert r.command_used == "/compare"
    assert "NVDA vs AMD" in r.query
    assert "render_table" in r.query


def test_summarize_requires_arg():
    r = dispatch("/summarize")
    assert r.kind == "pure"
    assert "usage" in r.output.lower()


def test_summarize_expands_query():
    r = dispatch("/summarize https://example.com/article")
    assert r.kind == "expand"
    assert "https://example.com/article" in r.query
    assert "render_card" in r.query


def test_unknown_command_is_pure():
    r = dispatch("/nope")
    assert r.kind == "pure"
    assert "unknown" in r.output.lower()
    assert r.command_used == "/nope"


def test_command_is_case_insensitive():
    r = dispatch("/HELP")
    assert r.kind == "pure"
    assert r.command_used == "/help"
