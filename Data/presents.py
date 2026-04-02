from environment_classes import Radar, Drone

λ = 0.02998
default_radar = Radar(
    λ = λ,
    f_c = cs.c/λ
)


djimavicair2 = Drone(
    name="DJI_Mavic_Air_2",
    N=2,
    L_1=0.005,
    L_2=0.07,
    f_rot=91.66,
)

djimavicmini = Drone(
    name="DJI_Mavic_Mini",
    N=2,
    L_1=0.005,
    L_2=0.035,
    f_rot=160,
)

djimatrice300rtk = Drone(
    name="DJI_Matrice_300_RTK",
    N=2,
    L_1=0.05,
    L_2=0.2665,
    f_rot=70,
)

parrotdisco = Drone(
    name="Parrot_Disco",
    N=2,
    L_1=0.01,
    L_2=0.104,
    f_rot=40,
)

djiphantom4 = Drone(
    name="DJI_Phantom_4",
    N=2,
    L_1=0.006,
    L_2=0.05,
    f_rot=116,
)