from vision_engine.solver.cache_r37 import RadixTreeCache, SharedScoreTable
from vision_engine.scoring import score_mask


def test_radix_trie_parity_and_overwrite():
    tree = RadixTreeCache()
    entries = {0: score_mask(0).total, (1<<25)-1:score_mask((1<<25)-1).total,
               (1<<24):score_mask(1<<24).total, 1234567:score_mask(1234567).total}
    for mask, score in entries.items():tree.put(mask,score)
    assert tree.size==len(entries)
    for mask, score in entries.items():assert tree.get(mask)==score
    assert tree.get(42) is None
    tree.put(1234567,125)
    assert tree.get(1234567)==125 and tree.size==len(entries)


def test_shared_binary_table_parity_and_missing():
    entries={m:score_mask(m).total for m in (0,1,2,3,7,127,255,12345,1<<24,(1<<25)-1)}
    owner=SharedScoreTable.build(entries)
    reader=SharedScoreTable.attach(owner.shm.name,owner.capacity)
    try:
        for mask,score in entries.items():assert reader.get(mask)==score
        assert reader.get(777777) is None
    finally:
        reader.close();owner.close();owner.unlink()


def test_invalid_mask_rejected():
    tree=RadixTreeCache()
    import pytest
    with pytest.raises(ValueError):tree.put(1<<25,10)


def test_numba_batch_matches_existing_solver_scoring():
    import numpy as np
    from vision_engine.solver.r37_numba_kernel import LUCKY,NORMAL,compiled_kernel
    masks=np.array([0, 1, 3, 7, (1<<25)-1, 0b1010101010101010101010101],dtype=np.int64)
    try:
        kernel=compiled_kernel()
    except ImportError:
        import pytest
        pytest.skip('Numba is an optional benchmark dependency')
    scores=kernel(masks,LUCKY,NORMAL)
    assert list(scores)==[score_mask(int(m)).total for m in masks]
