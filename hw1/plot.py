import numpy as np
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  
from matplotlib.ticker import MaxNLocator



def plot_bc():
    num_steps = list(range(0, 3001, 300))
    bc_mean = [-493.51, -8.66, 702.88, 1605.65, 3336.61, 3813.11, 3970.26, 4492.45, 4422.06, 4574.97, 4665.38]
    bc_std = [1172.55, 177.84, 613.82, 1180.15, 979.11, 995.86, 917.67, 110.70, 166.05, 158.21, 110.87]
    expert_mean = [4713.65] * len(num_steps)
    expert_std = [12.20] * len(num_steps)
    plt.figure()
    plt.errorbar(num_steps, bc_mean, bc_std, marker='o', capsize=4, linestyle='--', label='Behavior Cloning')
    plt.errorbar(num_steps, expert_mean, expert_std, marker='o', capsize=4, linestyle='--', label='Expert')
    plt.xlabel('Training steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('1.png')
    plt.show()


def plot_dagger():
    num_steps = list(range(0, 50))
    dagger_mean = [
        232.18380737304688,
        308.1683349609375,
        239.3844451904297,
        302.87371826171875,
        324.6551513671875,
        357.31719970703125,
        356.6280822753906,
        458.8662414550781,
        455.4006042480469,
        470.15020751953125,
        557.0930786132812,
        617.7924194335938,
        747.564697265625,
        868.0506591796875,
        1006.0827026367188,
        1100.6466064453125,
        1014.6576538085938,
        1083.1766357421875,
        1183.3812255859375,
        1530.71728515625,
        1512.5357666015625,
        1669.6158447265625,
        2213.501708984375,
        1707.0062255859375,
        2602.097900390625,
        2421.791748046875,
        2766.146484375,
        2156.802490234375,
        3720.918212890625,
        4438.18017578125,
        2334.154296875,
        3764.64794921875,
        3665.724853515625,
        3498.26904296875,
        3303.936279296875,
        3694.117431640625,
        4505.74951171875,
        4548.97607421875,
        5401.30029296875,
        3018.60009765625,
        4460.07666015625,
        5436.408203125,
        5342.14599609375,
        4867.4990234375,
        8092.26953125,
        5533.7333984375,
        7260.12744140625,
        3134.082275390625,
        7332.23388671875,
        5361.11376953125,
    ]

    dagger_std = [
        52.732154846191406,
        41.42679977416992,
        36.1939697265625,
        32.57678985595703,
        62.16191864013672,
        77.10896301269531,
        105.12774658203125,
        142.6112518310547,
        172.1103057861328,
        156.5653533935547,
        223.5443572998047,
        210.44618225097656,
        298.62261962890625,
        360.3747863769531,
        330.9210510253906,
        407.444580078125,
        469.24029541015625,
        620.4979858398438,
        565.2058715820312,
        877.3981323242188,
        743.0598754882812,
        925.4865112304688,
        995.1302490234375,
        862.0228881835938,
        1479.339599609375,
        930.5311889648438,
        1681.20849609375,
        1274.4154052734375,
        1999.11376953125,
        2251.276611328125,
        904.8419189453125,
        1970.3011474609375,
        1709.5633544921875,
        1221.2064208984375,
        1949.9425048828125,
        2050.328125,
        2439.839599609375,
        2410.05908203125,
        1592.962158203125,
        1245.8651123046875,
        2377.04736328125,
        2428.5517578125,
        2635.788818359375,
        2530.608154296875,
        2734.177490234375,
        2252.7197265625,
        2112.251708984375,
        1350.0015869140625,
        2737.692626953125,
        2998.627685546875,
    ]

    expert_mean = [10344.52] * len(num_steps)
    expert_std = [20.98] * len(num_steps)

    bc_mean = [232.18380737304688] * len(num_steps)
    bc_std = [52.732154846191406] * len(num_steps)

    # ax = plt.figure().gca()
    plt.errorbar(num_steps, dagger_mean, dagger_std, marker='o', capsize=4, linestyle='--', label='DAgger')
    plt.errorbar(num_steps, expert_mean, expert_std, marker='o', capsize=4, linestyle='--', label='Expert')
    plt.errorbar(num_steps, bc_mean, bc_std, marker='o', capsize=4, linestyle='--', label='Behavior Cloning')
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.legend()

    plt.savefig('2.png')
    plt.show()


plot_bc()
plot_dagger()
