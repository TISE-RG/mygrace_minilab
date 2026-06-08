EVENTS = [
    {
        "name": "Hari biasa tanpa kejadian khusus",
        "type": "ordinary",
        "effect": {"N": -1, "R": -1, "C": -1, "E": -1},
        "recommended": "a_nar"
    },
    {
        "name": "Anak menelepon dengan hangat",
        "type": "relational_positive",
        "effect": {"N": 3, "S": 1, "R": 8, "A": 3},
        "recommended": "a_rel"
    },
    {
        "name": "Anak tidak menelepon hari ini",
        "type": "relational_negative",
        "effect": {"N": -2, "R": -6, "A": -3},
        "recommended": "a_rel"
    },
    {
        "name": "Cucu mengirim foto kegiatan sekolah",
        "type": "legacy_positive",
        "effect": {"N": 4, "R": 6, "C": 2, "A": 2},
        "recommended": "a_nar"
    },
    {
        "name": "Badan terasa lemah",
        "type": "health_minor",
        "effect": {"N": -2, "E": -2, "A": -6},
        "recommended": "a_prac"
    },
    {
        "name": "Menemukan foto lama saat masih bekerja",
        "type": "memory_positive",
        "effect": {"N": 6, "C": 4, "A": 2},
        "recommended": "a_nar"
    },
    {
        "name": "Diminta memberi nasihat oleh cucu",
        "type": "contribution_positive",
        "effect": {"N": 3, "R": 4, "C": 8, "A": 2},
        "recommended": "a_con"
    },
    {
        "name": "Mendapat undangan kegiatan komunitas",
        "type": "community_opportunity",
        "effect": {"R": 2, "E": 6, "C": 2},
        "recommended": "a_com"
    },
    {
        "name": "Konflik kecil dengan anak",
        "type": "family_conflict",
        "effect": {"N": -3, "S": -2, "R": -7, "A": -4},
        "recommended": "a_rel"
    },
    {
        "name": "Mengenang pasangan yang telah meninggal",
        "type": "grief_trigger",
        "effect": {"N": -3, "S": -4, "R": -4, "A": -6},
        "recommended": "a_spi"
    },
]