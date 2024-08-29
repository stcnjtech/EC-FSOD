import os
from detectron2.data import MetadataCatalog
from .builtin_meta import _get_builtin_metadata
from .meta_voc import register_meta_nwpu, register_meta_dior

def register_all_npwu(root="datasets"):
    METASPLITS = [
        ("nwpu_trainval_base1", "nwpu", "trainval", "base1", 1),
        ("nwpu_trainval_base2", "nwpu", "trainval", "base2", 2),

        ("nwpu_trainval_all1", "nwpu", "trainval", "base_novel_1", 1),
        ("nwpu_trainval_all2", "nwpu", "trainval", "base_novel_2", 2),

        ("nwpu_test_base1", "nwpu", "test", "base1", 1),
        ("nwpu_test_base2", "nwpu", "test", "base2", 2),
        
        ("nwpu_test_novel1", "nwpu", "test", "novel1", 1),
        ("nwpu_test_novel2", "nwpu", "test", "novel2", 2),
        
        ("nwpu_test_all1", "nwpu", "test", "base_novel_1", 1),
        ("nwpu_test_all2", "nwpu", "test", "base_novel_2", 2),
    ]

    for prefix in ["all", "novel"]:
        for sid in [1, 2]:
            for shot in [3, 5, 10, 20]:
                for seed in range(30):
                    seed = "_seed{}".format(seed)
                    name = "nwpu_trainval_{}{}_{}shot{}".format(
                        prefix, sid, shot, seed
                    )
                    dirname = "nwpu"
                    img_file = "{}_{}shot_split_{}_trainval".format(
                        prefix, shot, sid
                    )
                    keepclasses = (
                        "base_novel_{}".format(sid)
                        if prefix == "all"
                        else "novel{}".format(sid)
                    )
                    METASPLITS.append(
                        (name, dirname, img_file, keepclasses, sid)
                    )
                    
    for name, dirname, split, keepclasses, sid in METASPLITS:
        register_meta_nwpu(
            name,
            _get_builtin_metadata("nwpu_fewshot"),
            os.path.join(root, dirname),
            split,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "nwpu"

def register_all_dior(root="datasets"):
    METASPLITS = [
        ("dior_trainval_base1", "dior", "trainval", "base1", 1),
        ("dior_trainval_base2", "dior", "trainval", "base2", 2),
        ("dior_trainval_base3", "dior", "trainval", "base3", 3),
        ("dior_trainval_base4", "dior", "trainval", "base4", 4),

        ("dior_trainval_all1", "dior", "trainval", "base_novel_1", 1),
        ("dior_trainval_all2", "dior", "trainval", "base_novel_2", 2),
        ("dior_trainval_all3", "dior", "trainval", "base_novel_3", 3),
        ("dior_trainval_all4", "dior", "trainval", "base_novel_4", 4),

        ("dior_test_base1", "dior", "test", "base1", 1),
        ("dior_test_base2", "dior", "test", "base2", 2),
        ("dior_test_base3", "dior", "test", "base3", 3),
        ("dior_test_base4", "dior", "test", "base4", 4),
        
        ("dior_test_novel1", "dior", "test", "novel1", 1),
        ("dior_test_novel2", "dior", "test", "novel2", 2),
        ("dior_test_novel3", "dior", "test", "novel3", 3),
        ("dior_test_novel4", "dior", "test", "novel4", 4),
        
        ("dior_test_all1", "dior", "test", "base_novel_1", 1),
        ("dior_test_all2", "dior", "test", "base_novel_2", 2),
        ("dior_test_all3", "dior", "test", "base_novel_3", 3),
        ("dior_test_all4", "dior", "test", "base_novel_4", 4),
    ]

    for prefix in ["all", "novel"]:
        for sid in [1, 2, 3, 4]:
            for shot in [3, 5, 10, 20]:
                for seed in range(30):
                    seed = "_seed{}".format(seed)
                    name = "dior_trainval_{}{}_{}shot{}".format(
                        prefix, sid, shot, seed
                    )
                    dirname = "dior"
                    img_file = "{}_{}shot_split_{}_trainval".format(
                        prefix, shot, sid
                    )
                    keepclasses = (
                        "base_novel_{}".format(sid)
                        if prefix == "all"
                        else "novel{}".format(sid)
                    )
                    METASPLITS.append(
                        (name, dirname, img_file, keepclasses, sid)
                    )
    
    for name, dirname, split, keepclasses, sid in METASPLITS:
        register_meta_dior(
            name,
            _get_builtin_metadata("dior_fewshot"),
            os.path.join(root, dirname),
            split,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "dior"

register_all_npwu()
register_all_dior()