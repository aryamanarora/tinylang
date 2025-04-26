import pytest
import numpy as np
import torch
from tinylang.language.pcfg import PCFG, QueryType, Node
import random
import time

# set seeds
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

@pytest.fixture
def basic_pcfg():
    """Create a basic PCFG instance for testing."""
    return PCFG(
        num_terminals=20,
        num_nonterminals=20,
        max_rhs_len=10,
        max_rules_per_nt=5,
        max_depth=10,
        head_position="left",
        mask_nonquery=False,
        no_sibling_queries=False,
        no_child_queries=False,
        max_length=1024,
        train_test_split=0.0,
        transparent_nonterminals=False,
        unambiguous_queries=False,
        sample_first="target",
    )

def test_pcfg_initialization(basic_pcfg):
    """Test that PCFG initializes with correct parameters."""
    assert basic_pcfg.num_terminals == 20
    assert basic_pcfg.num_nonterminals == 20
    assert basic_pcfg.max_rhs_len == 10
    assert basic_pcfg.max_rules_per_nt == 5
    assert basic_pcfg.max_depth == 10
    assert basic_pcfg.head_position == "left"
    assert not basic_pcfg.mask_nonquery

def test_pcfg_vocab_size(basic_pcfg):
    """Test that vocabulary size is calculated correctly."""
    # TERMINAL_START (3) + num_terminals (5) + QueryType values
    expected_size = 3 + 20 + max(QueryType).value + 1
    assert basic_pcfg.vocab_size == expected_size

def test_pcfg_sample(basic_pcfg):
    """Test that PCFG can generate samples."""
    tokens, probing_schema = basic_pcfg.sample()
    assert isinstance(tokens, list)
    assert isinstance(probing_schema, dict)
    assert len(tokens) > 0
    assert "queries" in probing_schema
    assert "keys" in probing_schema

def test_pcfg_batchify(basic_pcfg):
    """Test that batchify works correctly."""
    # Generate some samples
    toks = []
    probing_schemas = []
    for _ in range(3):
        tok, schema = basic_pcfg.sample()
        toks.append(tok)
        probing_schemas.append(schema)
    
    # Test batchify
    batch = basic_pcfg.batchify(toks, probing_schemas)
    assert "input_ids" in batch
    assert "labels" in batch
    assert "strs" in batch
    assert "probing_schemas" in batch
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert batch["input_ids"].shape[0] == 3  # batch size

def test_pcfg_get_eligible_pairs(basic_pcfg: PCFG):
    """Test that get_eligible_pairs works correctly."""
    sentence = basic_pcfg.sample(return_stats=True, return_sentence=True)[2]["sentence"]
    ambiguous_pairs, ambiguous_queries, ambiguous_targets = basic_pcfg.get_eligible_pairs(sentence)

    basic_pcfg.unambiguous_queries = True
    unambiguous_pairs, unambiguous_queries, unambiguous_targets = basic_pcfg.get_eligible_pairs(sentence)

    rightmost_types = dict()
    for i, node in enumerate(sentence):
        rightmost_types[node.label] = max(rightmost_types.get(node.label, 0), i)
    for query_type in ambiguous_pairs.keys():
        ambiguous_pairs[query_type] = [
            (query, target) for query, target in ambiguous_pairs[query_type] if rightmost_types[sentence[query].label] == query
        ]
    assert ambiguous_pairs == unambiguous_pairs

def test_pcfg_sample_methods_consistency(basic_pcfg):
    """Test that both sample methods produce valid sentences following PCFG rules."""
    # Generate sentences using both methods
    sentence1 = basic_pcfg._sample()
    sentence2 = basic_pcfg._sample2()
    
    # Both should produce non-empty sentences
    assert len(sentence1) > 0
    assert len(sentence2) > 0
    
    # All nodes should be terminals (since both methods fully expand)
    assert all(node.label in range(basic_pcfg.num_terminals) for node in sentence1)
    assert all(node.label in range(basic_pcfg.num_terminals) for node in sentence2)
    
    # All nodes should have valid head_ids (except root)
    assert all(node.head_id is not None or node.id == 1 for node in sentence1)
    assert all(node.head_id is not None or node.id == 1 for node in sentence2)
    
    # All head_ids should point to valid nodes
    head_ids1 = {node.id for node in sentence1}
    head_ids2 = {node.id for node in sentence2}
    assert all(node.head_id in head_ids1 or node.head_id is None for node in sentence1)
    assert all(node.head_id in head_ids2 or node.head_id is None for node in sentence2)

def test_pcfg_sample_methods_depth(basic_pcfg):
    """Test that both sample methods respect depth constraints."""
    # Generate multiple sentences to get good coverage
    for _ in range(10):
        sentence1 = basic_pcfg._sample()
        sentence2 = basic_pcfg._sample2()
        
        # Check that depths are reasonable
        max_depth1 = max(node.depth for node in sentence1)
        max_depth2 = max(node.depth for node in sentence2)
        assert max_depth1 <= basic_pcfg.max_depth
        assert max_depth2 <= basic_pcfg.max_depth
        
        # Check that depth increases by at most 1 from parent to child
        id_to_depth = {node.id: node.depth for node in sentence1}
        assert all(id_to_depth[node.head_id] + 1 >= node.depth for node in sentence1 if node.head_id is not None)
        
        id_to_depth = {node.id: node.depth for node in sentence2}
        assert all(id_to_depth[node.head_id] + 1 >= node.depth for node in sentence2 if node.head_id is not None)

def test_pcfg_sample_methods_head_position(basic_pcfg):
    """Test that both sample methods respect head position setting."""
    # Test with left head position
    basic_pcfg.head_position = "left"
    sentence1 = basic_pcfg._sample()
    sentence2 = basic_pcfg._sample2()
    
    # For left head position, the head should be the first child of its parent
    id_to_pos = {node.id: i for i, node in enumerate(sentence1)}
    assert all(id_to_pos[node.head_id] < id_to_pos[node.id] for node in sentence1 if node.head_id is not None)
    
    id_to_pos = {node.id: i for i, node in enumerate(sentence2)}
    assert all(id_to_pos[node.head_id] < id_to_pos[node.id] for node in sentence2 if node.head_id is not None)
    
    # Test with right head position
    basic_pcfg.head_position = "right"
    sentence1 = basic_pcfg._sample()
    sentence2 = basic_pcfg._sample2()
    
    # For right head position, the head should be the last child of its parent
    id_to_pos = {node.id: i for i, node in enumerate(sentence1)}
    assert all(id_to_pos[node.head_id] > id_to_pos[node.id] for node in sentence1 if node.head_id is not None)
    
    id_to_pos = {node.id: i for i, node in enumerate(sentence2)}
    assert all(id_to_pos[node.head_id] > id_to_pos[node.id] for node in sentence2 if node.head_id is not None)


# def test_pcfg_sample_methods_time(basic_pcfg):
#     """Test that sample is faster than sample2."""
#     time_sample1 = 0.0
#     time_sample2 = 0.0
#     start_time = time.time()
#     for _ in range(5000):
#         basic_pcfg._sample()
#     end_time = time.time()
#     time_sample1 += end_time - start_time
#     start_time = time.time()
#     for _ in range(5000):
#         basic_pcfg._sample2()
#     end_time = time.time()
#     time_sample2 += end_time - start_time
#     print(time_sample1, time_sample2)
#     assert time_sample1 < time_sample2
    
def test_pcfg_sample_methods_nonprobabilistic():
    """If the PCFG is transparent and there is only 1 rule per nonterminal and start symbol is a nonterminal, the sample methods should be deterministic."""
    for _ in range(10):
        np.random.seed(42 + _)
        torch.manual_seed(42 + _)
        random.seed(42 + _)
        basic_pcfg = PCFG(num_terminals=20,
            num_nonterminals=20,
            max_rhs_len=10,
            max_rules_per_nt=1,
            max_depth=10,
            head_position="left",
            mask_nonquery=False,
            no_sibling_queries=False,
            no_child_queries=False,
            max_length=1024,
            train_test_split=0.0,
            transparent_nonterminals=True,
            unambiguous_queries=False,
            sample_first="target",
        )
        for i in range(basic_pcfg.num_nonterminals):
            basic_pcfg.start_probs = np.zeros(basic_pcfg.num_nonterminals)
            basic_pcfg.start_probs[0] = 1
            sentence1 = basic_pcfg._sample()
            sentence2 = basic_pcfg._sample2()
            sentence1_labels = [(node.label, node.depth) for node in sentence1]
            sentence2_labels = [(node.label, node.depth) for node in sentence2]
            assert sentence1_labels == sentence2_labels
