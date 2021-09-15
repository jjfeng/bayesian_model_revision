CYCLE_LEN = 55
SIM_SETTINGS = {
    "iid": {
            "change_portion": 0.0,
            "all_batch_meta": [
            {
                "size": 500,
                "do_change": False,
                "copy_old_beta": None,
            }
        ]},
    "iid_deteriorate": {
            "change_portion": 0.02,
            "all_batch_meta": [
            {
                "size": 500,
                "do_change": False,
                "copy_old_beta": None,
            }
        ]},
    "late_deteriorate": {
            "change_portion": 0.18,
            "all_batch_meta": [
            {
                "size": 200,
                "do_change": False,
                "copy_old_beta": None,
            },
            {
                "size": 300,
                "do_change": True,
                "copy_old_beta": None,
            }
        ]},
    "mid_deteriorate": {
            "change_portion": 0.12,
            "all_batch_meta": [
            {
                "size": 100,
                "do_change": False,
                "copy_old_beta": None,
            },
            {
                "size": 200,
                "do_change": True,
                "copy_old_beta": None,
            },
            {
                "size": 200,
                "do_change": True,
                "copy_old_beta": None,
            }
        ]},
    "iid_big_deteriorate": {
            "change_portion": 0.15,
            "all_batch_meta": [
            {
                "size": 500,
                "do_change": True,
                "copy_old_beta": None,
            }
        ]},
    "new_iid": {
            "change_portion": 0.05,
            "all_batch_meta": [
            {
                "size": 500,
                "do_change": True,
                "copy_old_beta": None,
            }
        ]},
    "good_then_deteriorate": {
        "change_portion": 0.03,
        "all_batch_meta": [
            {
                "size": 100,
                "do_change": False,
                "copy_old_beta": None,
            },
            {
                "size": 100,
                "do_change": True,
                "copy_old_beta": None,
            },
            {
                "size": 100,
                "do_change": True,
                "copy_old_beta": None,
            },
            {
                "size": 200,
                "do_change": True,
                "copy_old_beta": None,
            },
        ]},
    "very_deteriorate": {
        "change_portion": 0.03,
        "all_batch_meta": [
            {
                "size": 20,
                "do_change": False,
                "copy_old_beta": None,
            },
            {
                "size": 30,
                "do_change": True,
                "copy_old_beta": None,
            },
            {
                "size": 60,
                "do_change": True,
                "copy_old_beta": None,
            },
            {
                "size": 100,
                "do_change": True,
                "copy_old_beta": None,
            },
            {
                "size": 120,
                "do_change": True,
                "copy_old_beta": None,
            },
            {
                "size": 170,
                "do_change": True,
                "copy_old_beta": None,
            },
        ]},
    "cyclical_deteriorate":{
        "change_portion": 0.03,
        "all_batch_meta": [
            {
                "size": CYCLE_LEN,
                "do_change": True,
                "copy_old_beta": None,
            },
            {
                "size": CYCLE_LEN,
                "do_change": True,
                "copy_old_beta": None,
            },
            {
                "size": CYCLE_LEN,
                "do_change": True,
                "copy_old_beta": None,
            },
        ] + [
            {
                "size": CYCLE_LEN,
                "do_change": True,
                "copy_old_beta": 0,
            },
            {
                "size": CYCLE_LEN,
                "do_change": True,
                "copy_old_beta": 1,
            },
            {
                "size": CYCLE_LEN,
                "do_change": True,
                "copy_old_beta": 2,
            },
        ] * 2},
    "tiny":{
        "change_portion": 0.05,
        "all_batch_meta": [
            {
                "size": 30,
                "do_change": True,
                "copy_old_beta": None,
            },
            {
                "size": 30,
                "do_change": True,
                "copy_old_beta": None,
            },
        ]}
}
