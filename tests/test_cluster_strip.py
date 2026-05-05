# tests/test_cluster_strip.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cluster_match import strip_cluster_noise

def test_strip_shn_single_suffix():
    assert strip_cluster_noise("SHN Cluster Süderdonn T411") == "Süderdonn"

def test_strip_shn_multi_suffix():
    assert strip_cluster_noise("SHN Cluster Süderdonn T412 T413") == "Süderdonn"

def test_strip_ava():
    assert strip_cluster_noise("AVA Cluster Alfstedt") == "Alfstedt"

def test_strip_bag_nwak_with_number():
    assert strip_cluster_noise("BAG NWAK-Cluster 17 Altheim") == "Altheim"

def test_strip_ttg_with_ee():
    assert strip_cluster_noise("TTG Cluster EE Bechterdissen T411") == "Bechterdissen"

def test_strip_croc():
    assert strip_cluster_noise("CROC Cluster Görries") == "Görries"

def test_strip_no_suffix():
    assert strip_cluster_noise("AVA Cluster Helmstedt") == "Helmstedt"

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
