# These scores are taken from the rliable library. It originaly comes from the MuZero paper.
ALL_HUMAN_SCORES = {
    "Alien": 7127.7,
    "Amidar": 1719.5,
    "Assault": 742.0,
    "Asterix": 8503.3,
    "Asteroids": 47388.7,
    "Atlantis": 29028.1,
    "BankHeist": 753.1,
    "BattleZone": 37187.5,
    "BeamRider": 16926.5,
    "Berzerk": 2630.4,
    "Bowling": 160.7,
    "Boxing": 12.1,
    "Breakout": 30.5,
    "Centipede": 12017.0,
    "ChopperCommand": 7387.8,
    "CrazyClimber": 35829.4,
    "DemonAttack": 1971.0,
    "DoubleDunk": -16.4,
    "Enduro": 860.5,
    "FishingDerby": -38.7,
    "Freeway": 29.6,
    "Frostbite": 4334.7,
    "Gopher": 2412.5,
    "Gravitar": 3351.4,
    "Hero": 30826.4,
    "IceHockey": 0.9,
    "Jamesbond": 302.8,
    "Kangaroo": 3035.0,
    "Krull": 2665.5,
    "KungFuMaster": 22736.3,
    "MontezumaRevenge": 4753.3,
    "MsPacman": 6951.6,
    "NameThisGame": 8049.0,
    "Phoenix": 7242.6,
    "Pitfall": 6463.7,
    "Pong": 14.6,
    "PrivateEye": 69571.3,
    "Qbert": 13455.0,
    "Riverraid": 17118.0,
    "RoadRunner": 7845.0,
    "Robotank": 11.9,
    "Seaquest": 42054.7,
    "Skiing": -4336.9,
    "Solaris": 12326.7,
    "SpaceInvaders": 1668.7,
    "StarGunner": 10250.0,
    "Tennis": -8.3,
    "TimePilot": 5229.2,
    "Tutankham": 167.6,
    "UpNDown": 11693.2,
    "Venture": 1187.5,
    "VideoPinball": 17667.9,
    "WizardOfWor": 4756.5,
    "YarsRevenge": 54576.9,
    "Zaxxon": 9173.3,
}

ALL_RANDOM_SCORES = {
    "Alien": 227.8,
    "Amidar": 5.8,
    "Assault": 222.4,
    "Asterix": 210.0,
    "Asteroids": 719.1,
    "Atlantis": 12850.0,
    "BankHeist": 14.2,
    "BattleZone": 2360.0,
    "BeamRider": 363.9,
    "Berzerk": 123.7,
    "Bowling": 23.1,
    "Boxing": 0.1,
    "Breakout": 1.7,
    "Centipede": 2090.9,
    "ChopperCommand": 811.0,
    "CrazyClimber": 10780.5,
    "Defender": 2874.5,
    "DemonAttack": 152.1,
    "DoubleDunk": -18.6,
    "Enduro": 0.0,
    "FishingDerby": -91.7,
    "Freeway": 0.0,
    "Frostbite": 65.2,
    "Gopher": 257.6,
    "Gravitar": 173.0,
    "Hero": 1027.0,
    "IceHockey": -11.2,
    "Jamesbond": 29.0,
    "Kangaroo": 52.0,
    "Krull": 1598.0,
    "KungFuMaster": 258.5,
    "MontezumaRevenge": 0.0,
    "MsPacman": 307.3,
    "NameThisGame": 2292.3,
    "Phoenix": 761.4,
    "Pitfall": -229.4,
    "Pong": -20.7,
    "PrivateEye": 24.9,
    "Qbert": 163.9,
    "Riverraid": 1338.5,
    "RoadRunner": 11.5,
    "Robotank": 2.2,
    "Seaquest": 68.4,
    "Skiing": -17098.1,
    "Solaris": 1236.3,
    "SpaceInvaders": 148.0,
    "StarGunner": 664.0,
    "Surround": -10.0,
    "Tennis": -23.8,
    "TimePilot": 3568.0,
    "Tutankham": 11.4,
    "UpNDown": 533.4,
    "Venture": 0.0,
    "VideoPinball": 0.0,
    "WizardOfWor": 563.5,
    "YarsRevenge": 3092.9,
    "Zaxxon": 32.5,
}


EXPERIMENTED_GAME = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis",
    "BankHeist",
    "BattleZone",
    "BeamRider",
    "Berzerk",
    "Bowling",
    "Boxing",
    "Breakout",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "DemonAttack",
    "DoubleDunk",
    "Enduro",
    "FishingDerby",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "IceHockey",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MontezumaRevenge",
    "MsPacman",
    "NameThisGame",
    "Phoenix",
    "Pitfall",
    "Pong",
    "PrivateEye",
    "Qbert",
    "Riverraid",
    "RoadRunner",
    "Robotank",
    "Seaquest",
    "Skiing",
    "Solaris",
    "SpaceInvaders",
    "StarGunner",
    # "Tennis",
    "TimePilot",
    "Tutankham",
    "UpNDown",
    "Venture",
    "VideoPinball",
    "WizardOfWor",
    "YarsRevenge",
    "Zaxxon",
]

EXPERIMENTED_GAME_SHORT = [
    "Alien",
    "BankHeist",
    "ChopperCommand",
    "Enduro",
    "Frostbite",
    "Jamesbond",
    "KungFuMaster",
    "Seaquest",
    "Skiing",
    "StarGunner",
]

EXPERIMENTED_GAME_MEDIUM = [
    "Alien",
    "Assault",
    "BankHeist",
    "Berzerk",
    "Breakout",
    "Centipede",
    "ChopperCommand",
    "DemonAttack",
    "Enduro",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "IceHockey",
    "Jamesbond",
    "Krull",
    "KungFuMaster",
    "Riverraid",
    "Seaquest",
    "Skiing",
    "StarGunner",
]

AVAILABLE_COLORS = {
    "black": "#000000",
    "blue": "#1F77B4",
    "light_blue": "#AEC7E8",
    "orange": "#FF7F0E",
    "light_orange": "#FFBB78",
    "green": "#2CA02C",
    "light_green": "#98DF8A",
    "red": "#D1797A",
    "light_red": "#FF9896",
    "purple": "#9467BD",
    "light_purple": "#C5B0D5",
    "brown": "#8C564B",
    "light_brown": "#C49C94",
    "pink": "#E377C2",
    "light_pink": "#F7B6D2",
    "grey": "#7F7F7F",
    "light_grey": "#C7C7C7",
    "yellow": "#DEDE00",
    "light_yellow": "#F0E886",
    "cyan": "#17BECF",
    "light_cyan": "#9EDAE5",
}

COLORS = {
    # iDQN
    "DQN_sanity_check": AVAILABLE_COLORS["brown"],
    "iDQN_sanity_check_1": AVAILABLE_COLORS["green"],
    "iDQN_ut30_uh6000_5": AVAILABLE_COLORS["green"],
    # iIQN
    "IQN_sanity_check": AVAILABLE_COLORS["orange"],
    "iIQN_sanity_check_1": AVAILABLE_COLORS["cyan"],
    "iIQN_ut30_uh6000_3": AVAILABLE_COLORS["green"],
    "iIQN_ut30_uh6000_5": AVAILABLE_COLORS["green"],
    "iIQN_weak_ut30_uh6000_3": AVAILABLE_COLORS["green"],
    # iREM
    "REM_sanity_check": AVAILABLE_COLORS["yellow"],
    "iREM_sanity_check_1": AVAILABLE_COLORS["black"],
    # iDQN + 3-step return
    "DQN_sanity_check_3_steps": AVAILABLE_COLORS["brown"],
    "DQN_sanity_check_3_steps_old": AVAILABLE_COLORS["yellow"],
    "iDQN_sanity_check_3_steps_1": AVAILABLE_COLORS["cyan"],
    "iDQN_ut30_uh6000_3_steps_5": AVAILABLE_COLORS["pink"],
    # iIQN + 3-step return
    "IQN_sanity_check_3_steps": AVAILABLE_COLORS["orange"],
    "iIQN_sanity_check_3_steps_1": AVAILABLE_COLORS["light_cyan"],
    "iIQN_ut30_uh6000_3_steps_3": AVAILABLE_COLORS["yellow"],
    "iIQN_ut30_uh6000_3_steps_old_5": AVAILABLE_COLORS["light_cyan"],
    # Ablations K
    "iDQN_ablation_study_K_5": AVAILABLE_COLORS["green"],
    "iDQN_ablation_study_K_10": AVAILABLE_COLORS["orange"],
    # Ablations behavioral policy
    "iDQN_ut30_uh6000_k1_5_5": AVAILABLE_COLORS["green"],
    "iDQN_ut30_uh6000_k1_5": AVAILABLE_COLORS["blue"],
    "iDQN_ut30_uh6000_k5_5": AVAILABLE_COLORS["pink"],
    # Ablations head std
    "iDQN_head_std_5": AVAILABLE_COLORS["brown"],
    # Ablations Bellman iterations
    "DQN_ut30_uh2000": AVAILABLE_COLORS["light_pink"],
    "iDQN_ut30_uh2000_4": AVAILABLE_COLORS["green"],
    # Ablations overfit
    "DQN_one_step_one_grad": AVAILABLE_COLORS["light_pink"],
    "DQN_ut30_uh8000": AVAILABLE_COLORS["orange"],
    "iDQN_ut30_uh8000_2": AVAILABLE_COLORS["cyan"],
    "iDQN_ut30_uh8000_4": AVAILABLE_COLORS["green"],
    # Ablations R
    "iDQN_ut30_uh100_5": AVAILABLE_COLORS["orange"],
    "iDQN_ut30_uh500_5": AVAILABLE_COLORS["orange"],
    # Ablations T
    "iDQN_ut1_uh6000_5": AVAILABLE_COLORS["red"],
    # Ablations independent
    "iDQN_ut30_uh6000_indep_5": AVAILABLE_COLORS["cyan"],
    # Baselines from "Deep Reinforcement Learning at the Edge of the Statistical Precipice"
    "DQN (Nature)": AVAILABLE_COLORS["grey"],
    "Quantile (JAX)_dopamine": AVAILABLE_COLORS["light_blue"],
    "DQN (Adam)": AVAILABLE_COLORS["green"],
    "DQN + n-step return": AVAILABLE_COLORS["pink"],  # Only on a few games
    "C51": AVAILABLE_COLORS["purple"],
    "REM": AVAILABLE_COLORS["brown"],
    "Rainbow": AVAILABLE_COLORS["purple"],
    "IQN_pure": AVAILABLE_COLORS["cyan"],  # Only on a few games
    "IQN": AVAILABLE_COLORS["yellow"],
    "IQN + n-step return (dopamine)": AVAILABLE_COLORS["light_green"],
    "M-IQN": AVAILABLE_COLORS["light_yellow"],
    # Baselines from dopamine
    # "DQN_dopamine": AVAILABLE_COLORS["grey"],
    # "DQN (Adam + MSE in JAX)_dopamine": AVAILABLE_COLORS["light_pink"],
    # "C51_dopamine": AVAILABLE_COLORS["light_purple"],
    # "Rainbow_dopamine": AVAILABLE_COLORS["purple"],
    # "IQN_dopamine": AVAILABLE_COLORS["orange"],
}

LABEL = {
    # iDQN
    "DQN_sanity_check": "DQN (our implementation)",
    "iDQN_sanity_check_1": "iDQN K=1",
    "iDQN_ut30_uh6000_5": "iDQN, T=6000",  # iDQN K=5, T=30, R=6000 for ablations on R and T | iDQN K=5, shared convolutions for indep | iDQN K=5 normal
    # iIQN
    "IQN_sanity_check": "IQN w/o 3-step return (our implementation)",
    "iIQN_sanity_check_1": "iIQN K=1",
    "iIQN_ut30_uh6000_3": "iIQN K=3 (iDQN + IQN)",
    "iIQN_ut30_uh6000_5": "iIQN K=5 (iDQN + IQN)",
    "iIQN_weak_ut30_uh6000_3": "iIQN weak K=3 (iDQN + IQN weak)",
    # iREM
    "REM_sanity_check": "REM (our implementation)",
    "iREM_sanity_check_1": "iREM K=1",
    # iDQN + 3-step return
    "DQN_sanity_check_3_steps": "DQN + 3-step return (our implementation)",
    "DQN_sanity_check_3_steps_old": "DQN + 3-step return (our implementation, gamma=1)",
    "iDQN_sanity_check_3_steps_1": "iDQN K=1 + 3-step return",
    "iDQN_ut30_uh6000_3_steps_5": "iDQN + 3-step return",
    # iIQN + 3-step return
    "IQN_sanity_check_3_steps": "IQN (our implementation)",
    "iIQN_sanity_check_3_steps_1": "iIQN K=1 + 3-step return",
    "iIQN_ut30_uh6000_3_steps_3": "iIQN",
    "iIQN_ut30_uh6000_3_steps_old_5": "iIQN K=5 (iDQN + IQN) + 3-step return (gamma=1)",
    # Ablations K
    "iDQN_ablation_study_K_5": "iDQN K=5",
    "iDQN_ablation_study_K_10": "iDQN K=10",
    # Ablations behavioral policy
    "iDQN_ut30_uh6000_k1_5_5": "uniform sampling",
    "iDQN_ut30_uh6000_k1_5": "first online Q sampling",
    "iDQN_ut30_uh6000_k5_5": "last online Q sampling",
    # Ablations head std
    "iDQN_head_std_5": "inter-head standard deviation",
    # Ablations Bellman iterations
    "DQN_ut30_uh2000": "DQN, T=2000",
    "iDQN_ut30_uh2000_4": "iDQN K=4, T=2000",
    # Ablations overfit
    "DQN_one_step_one_grad": "DQN, G=1",
    "DQN_ut30_uh8000": "DQN, G=1",
    "iDQN_ut30_uh8000_2": "iDQN K=2",
    "iDQN_ut30_uh8000_4": "iDQN K=4, G=4",
    # Ablations R
    "iDQN_ut30_uh100_5": "iDQN, T=100",
    "iDQN_ut30_uh500_5": "iDQN K=5, T=30, R=500",
    # Ablations T
    "iDQN_ut1_uh6000_5": "iDQN w/o delayed params",
    # Ablations independent
    "iDQN_ut30_uh6000_indep_5": "iDQN (independent networks)",
    # Baselines from "Deep Reinforcement Learning at the Edge of the Statistical Precipice"
    "DQN (Nature)": "DQN (Nature)",
    "Quantile (JAX)_dopamine": "QR-DQN + 3-step return",
    "DQN (Adam)": "DQN (dopamine)",  # DQN, T=8000, G=4 for overfit | DQN, T=8000 for Bellman | DQN (Adam) normal
    "DQN + n-step return": "DQN + 3-step return (dopamine)",  # Only on a few games
    "C51": "C51",
    "REM": "REM",
    "Rainbow": "Rainbow",  # Rainbow (C51 + 3-step return + PER)
    "IQN_pure": "IQN w/o 3-step return (dopamine)",  # Only on a few games
    "IQN": "IQN (dopamine)",  # IQN + 3-step return
    "IQN + n-step return (dopamine)": "IQN + 3-step return (dopamine)",
    "M-IQN": "Munchausen + IQN + 3-step return",
    # Baselines from dopamine
    # "DQN_dopamine": "DQN (Nature) dopamine",
    # "DQN (Adam + MSE in JAX)_dopamine": "DQN (Adam) dopamine",
    # "C51_dopamine": "C51 dopamine",
    # "Rainbow_dopamine": "Rainbow dopamine",
    # "IQN_dopamine": "IQN dopamine",
}

ORDER = {
    # iDQN
    "DQN_sanity_check": 5,
    "iDQN_sanity_check_1": 6,
    "iDQN_ut30_uh6000_5": 11,
    # iIQN
    "IQN_sanity_check": 9,
    "iIQN_sanity_check_1": 10,
    "iIQN_ut30_uh6000_3": 12,
    "iIQN_ut30_uh6000_5": 14,
    "iIQN_weak_ut30_uh6000_3": 13,
    # iREM
    "REM_sanity_check": 7,
    "iREM_sanity_check_1": 8,
    # iDQN + 3-step return
    "DQN_sanity_check_3_steps": 4,
    "DQN_sanity_check_3_steps_old": 5,
    "iDQN_sanity_check_3_steps_1": 5,
    "iDQN_ut30_uh6000_3_steps_5": 11,
    # iIQN + 3-step return
    "IQN_sanity_check_3_steps": 9,
    "iIQN_sanity_check_3_steps_1": 10,
    "iIQN_ut30_uh6000_3_steps_3": 11,
    "iIQN_ut30_uh6000_3_steps_old_5": 11,
    # Ablations K
    "iDQN_ablation_study_K_5": 11,
    "iDQN_ablation_study_K_10": 12,
    # Ablations behavioral policy
    "iDQN_ut30_uh6000_k1_5_5": 5,
    "iDQN_ut30_uh6000_k1_5": 2,
    "iDQN_ut30_uh6000_k5_5": 3,
    # Ablations head std
    "iDQN_head_std_5": 2,
    # Ablations Bellman iterations
    "DQN_ut30_uh2000": 4,
    "iDQN_ut30_uh2000_4": 5,
    # Ablations overfit
    "DQN_one_step_one_grad": 3,
    "DQN_ut30_uh8000": 3,
    "iDQN_ut30_uh8000_2": 5,
    "iDQN_ut30_uh8000_4": 5,
    # Ablations R
    "iDQN_ut30_uh100_5": 10,
    "iDQN_ut30_uh500_5": 13,
    # Ablations T
    "iDQN_ut1_uh6000_5": 10,
    # Ablations independent
    "iDQN_ut30_uh6000_indep_5": 12,
    # Baselines from "Deep Reinforcement Learning at the Edge of the Statistical Precipice"
    "DQN (Nature)": 2,
    "Quantile (JAX)_dopamine": 3,
    "DQN (Adam)": 4,
    "DQN + n-step return": 3,  # Only on a few games
    "C51": 5,
    "REM": 6,
    "Rainbow": 7,
    "IQN_pure": 8,  # Only on a few games
    "IQN": 8,
    "IQN + n-step return (dopamine)": 8,
    "M-IQN": 9,
    # Baselines from dopamine
    # "DQN_dopamine": 2,
    # "DQN (Adam + MSE in JAX)_dopamine": 4,
    # "C51_dopamine": 5,
    # "Rainbow_dopamine": 7,
    # "IQN_dopamine": 8,
}
