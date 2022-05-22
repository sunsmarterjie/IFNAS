from collections import namedtuple
from random import sample
import random
import copy

Genotype = namedtuple('Genotype', 'stage1 stage2 stage3')

PRIMITIVES = [
    'skip',
    'conv3',
    'none',
    'avg_pool_3x3',
    'max_pool_3x3',
    # 'sconv3',
]

'''
each stage is one dict(), the keys are "node0, node1, ...'
the values are list(), the elements are inputs' nodes and operators
'''


def supernet_genotype(s1, s2, s3, s4=None, length=0):
    '''
    input:
        s1, s2, s3, s4 are number of nodes in stage1, 2, 3, 4
        length: the biggest depth between two nodes; if 0, no limination on this
    output:
        the genotype of supernet
    '''
    geno = Genotype(
        stage1=dict(),
        stage2=dict(),
        stage3=dict(),
        # stage4=dict(),
    )

    for i in range(1, s1 + 1):
        # nodes of stage1
        g = []
        if length == 0:
            k = 0
        else:
            k = max(0, i - length)
        for j in range(k, i):
            for k in range(len(PRIMITIVES)):
                g.append((PRIMITIVES[k], j))
        geno[0]['node%d' % i] = g
    for i in range(1, s2 + 1):
        # nodes of stage2
        g = []
        if length == 0:
            k = 0
        else:
            k = max(0, i - length)
        for j in range(k, i):
            for k in range(len(PRIMITIVES)):
                g.append((PRIMITIVES[k], j))
        geno[1]['node%d' % i] = g
    for i in range(1, s3 + 1):
        # nodes of stage3
        g = []
        if length == 0:
            k = 0
        else:
            k = max(0, i - length)
        for j in range(k, i):
            for k in range(len(PRIMITIVES)):
                g.append((PRIMITIVES[k], j))
        geno[2]['node%d' % i] = g
    # for i in range(1, s4 + 1):
    #     # nodes of stage4
    #     g = []
    #     if length == 0:
    #         k = 0
    #     else:
    #         k = max(0, i - length)
    #     for j in range(k, i):
    #         for k in range(len(PRIMITIVES)):
    #             g.append((PRIMITIVES[k], j))
    #     geno[3]['node%d' % i] = g
    return geno


def show_genotype(geno):
    for i in range(3):
        # four stages
        print('stage%d:' % (i + 1))
        for j, n in enumerate(geno[i].items()):
            print("     ", n)


def filter_genotype(geno):
    geno1 = filter(geno)
    while len(geno1[0]) != len(geno[0]) or len(geno1[1]) != len(geno[1]) or len(geno1[2]) != len(geno[2]):# or len(
            # geno1[3]) != len(geno[3]):
        geno = geno1
        # print('filter while')
        geno1 = filter(geno)
    return geno


def filter(geno):
    '''
        To filter some nodes which are not inputs
    '''
    S = set()
    S_todelete = list()
    geno_ = copy.deepcopy(geno)
    # filter stage1
    for v in geno_[0].values():
        for _, e in enumerate(v):
            S.add(e[1])
    keys = geno_[0].keys()
    for k in keys:
        if int(k[4:]) not in S:
            S_todelete.append(k)
    for n in S_todelete[:-1]:
        geno_[0].pop(n)
    # filter stage1 end
    S.clear()
    S_todelete.clear()

    # filter stage2
    for v in geno_[1].values():
        for _, e in enumerate(v):
            S.add(e[1])
    keys = geno_[1].keys()
    for k in keys:
        if int(k[4:]) not in S:
            S_todelete.append(k)
    for n in S_todelete[:-1]:
        geno_[1].pop(n)
    # filter stage2 end
    S.clear()
    S_todelete.clear()

    # filter stage3
    for v in geno_[2].values():
        for _, e in enumerate(v):
            S.add(e[1])
    keys = geno_[2].keys()
    for k in keys:
        if int(k[4:]) not in S:
            S_todelete.append(k)
    for n in S_todelete[:-1]:
        geno_[2].pop(n)
    # filter stage3 end
    # print('S:', S)
    # print('S_todelete:', S_todelete)
    S.clear()
    S_todelete.clear()

    # filter stage4
    # for v in geno_[3].values():
    #     for _, e in enumerate(v):
    #         S.add(e[1])
    # keys = geno_[3].keys()
    # for k in keys:
    #     if int(k[4:]) not in S:
    #         S_todelete.append(k)
    # for n in S_todelete[:-1]:
    #     geno_[3].pop(n)
    # filter stage4 end

    return geno_


def sample_geno(num_per_node, super_geno):
    '''
    sample genotype within the allowed set (existing genotype of supernet)
    '''
    geno = Genotype(
        stage1=dict(),
        stage2=dict(),
        stage3=dict(),
        stage4=dict()
    )
    super_geno = filter_genotype(super_geno)
    # sample stage1
    for k, v in super_geno[0].items():
        # print('len:', len(v))
        # print('id:', id)
        id = random.randint(0, max(0, len(v) - num_per_node[0]))
        geno[0][k] = v[id:id + min(num_per_node[0], len(v) - id)]
        # geno[0][k] = sample(v, min(num_per_node[0], len(v)))
    # sample stage1 end

    # sample stage2
    for k, v in super_geno[1].items():
        # geno[1][k] = sample(v, min(num_per_node[1], len(v)))
        id = random.randint(0, max(0, len(v) - num_per_node[1]))
        geno[1][k] = v[id:id + min(num_per_node[1], len(v) - id)]
    # sample stage2 end

    # sample stage3
    for k, v in super_geno[2].items():
        # geno[2][k] = sample(v, min(num_per_node[2], len(v)))
        id = random.randint(0, max(0, len(v) - num_per_node[2]))
        geno[2][k] = v[id:id + min(num_per_node[2], len(v) - id)]
    # sample stage3 end

    # sample stage4
    for k, v in super_geno[3].items():
        # geno[3][k] = sample(v, min(num_per_node[3], len(v)))
        id = random.randint(0, max(0, len(v) - num_per_node[3]))
        geno[3][k] = v[id:id + min(num_per_node[3], len(v) - id)]
    # sample stage4 end

    return filter_genotype(geno)


def tiny_enough(geno, weights, num_retain, prune=0.1):
    '''
        If supernet is tiny enough for GOLD-NAS then stop to crop supernet
    '''
    link1 = 2
    for k, n in enumerate(geno[0]):
        if k > 0:
            link1 += max(num_retain, sum(weights[0][int(n[4:]) - 1][:len(geno[0][n])] > prune))
    link2 = 2
    for k, n in enumerate(geno[1]):
        if k > 0:
            link2 += max(num_retain, sum(weights[1][int(n[4:]) - 1][:len(geno[1][n])] > prune))
    link3 = 2
    for k, n in enumerate(geno[2]):
        if k > 0:
            link3 += max(num_retain, sum(weights[2][int(n[4:]) - 1][:len(geno[2][n])] > prune))
    link4 = 2
    for k, n in enumerate(geno[3]):
        if k > 0:
            link4 += max(num_retain, sum(weights[3][int(n[4:]) - 1][:len(geno[3][n])] > prune))
    nodes = len(geno[0]) + len(geno[1]) + len(geno[2]) + len(geno[3])
    retain = nodes * num_retain
    links = link1 + link2 + link3 + link4
    if links > retain - 2:
        return False
    else:
        return True


def sample_random(supergeno, max_inp):
    '''
        random sample geno from a supergeno with the max_num of inputs: max_inp
    '''
    geno = Genotype(
        stage1={},
        stage2={},
        stage3={},
        stage4={}
    )
    for i in range(4):
        for k, n in enumerate(supergeno[i]):
            geno[i][n] = random.sample(supergeno[i][n], min(len(supergeno[i][n]), random.randint(1, max_inp)))
    return geno


def TAAS_sample(super_geno, turns=1, randomSample=True, length=4):
    '''
        super_geno: supernet until now
        turns: if 1, search node1\node5\node9; if 2, search node2\node6\node10, etc.
        randomSample: whether random sample child net genotype, pick one of two between turns and randomSample
        length: same to genotype
        sample TAAS genotype within the allowed set (existing genotype of supernet)
    '''
    geno = Genotype(
        stage1=dict(),
        stage2=dict(),
        stage3=dict(),
        # stage4=dict(),
    )
    super_geno = filter_genotype(super_geno)
    # sample stage1
    for k, v in super_geno[0].items():
        if (int(k[4:]) - turns) % (length + 1) == 0:
            geno[0][k] = v
        else:
            geno[0][k] = [v[-1]]
    # sample stage1 end

    # sample stage2
    for k, v in super_geno[1].items():
        if (int(k[4:]) - turns) % (length + 1) == 0:
            geno[1][k] = v
        else:
            geno[1][k] = [v[-1]]
    # sample stage2 end

    # sample stage3
    for k, v in super_geno[2].items():
        if (int(k[4:]) - turns) % (length + 1) == 0:
            geno[2][k] = v
        else:
            geno[2][k] = [v[-1]]
    # sample stage3 end

    # # sample stage4
    # for k, v in super_geno[3].items():
    #     if (int(k[4:]) - turns) % (length + 1) == 0:
    #         geno[3][k] = v
    #     else:
    #         geno[3][k] = [v[-1]]
    # sample stage4 end
    return filter_genotype(geno)

geno_cifar = Genotype(
    stage1={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            'node17': [('conv3', 16)],
            'node18': [('conv3', 17), ('skip', 16)],
            'node19': [('conv3', 18)],
            'node20': [('conv3', 19), ('skip', 18)],
            'node21': [('conv3', 20)],
            'node22': [('conv3', 21), ('skip', 20)],
            'node23': [('conv3', 22)],
            'node24': [('conv3', 23), ('skip', 22)]},
    stage2={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            'node17': [('conv3', 16)],
            'node18': [('conv3', 17), ('skip', 16)],
            'node19': [('conv3', 18)],
            'node20': [('conv3', 19), ('skip', 18)],
            'node21': [('conv3', 20)],
            'node22': [('conv3', 21), ('skip', 20)],
            'node23': [('conv3', 22)],
            'node24': [('conv3', 23), ('skip', 22)],
            'node25': [('conv3', 24)],
            'node26': [('conv3', 25), ('skip', 24)],
            'node27': [('conv3', 26)],
            'node28': [('conv3', 27), ('skip', 26)]
            },
    stage3={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            'node17': [('conv3', 16)],
            'node18': [('conv3', 17), ('skip', 16)],
            'node19': [('conv3', 18)],
            'node20': [('conv3', 19), ('skip', 18)],
            'node21': [('conv3', 20)],
            'node22': [('conv3', 21), ('skip', 20)],
            'node23': [('conv3', 22)],
            'node24': [('conv3', 23), ('skip', 22)]},
)

geno_image = Genotype(
    stage1={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            'node17': [('conv3', 16)],
            'node18': [('conv3', 17), ('skip', 16)],
            'node19': [('conv3', 18)],
            'node20': [('conv3', 19), ('skip', 18)]},
    stage2={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            'node17': [('conv3', 16)],
            'node18': [('conv3', 17), ('skip', 16)],
            'node19': [('conv3', 18)],
            'node20': [('conv3', 19), ('skip', 18)],
            'node21': [('conv3', 20)],
            'node22': [('conv3', 21), ('skip', 20)],
            },
    stage3={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            'node17': [('conv3', 16)],
            'node18': [('conv3', 17), ('skip', 16)],
            'node19': [('conv3', 18)],
            'node20': [('conv3', 19), ('skip', 18)]},
)

geno_image_short = Genotype(
    stage1={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)]},
    # 'node17': [('conv3', 16)],
    # 'node18': [('conv3', 17), ('skip', 16)],
    # 'node19': [('conv3', 18)],
    # 'node20': [('conv3', 19), ('skip', 18)]},
    stage2={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            'node17': [('conv3', 16)],
            'node18': [('conv3', 17), ('skip', 16)],
            # 'node19': [('conv3', 18)],
            # 'node20': [('conv3', 19), ('skip', 18)],
            # 'node21': [('conv3', 20)],
            # 'node22': [('conv3', 21), ('skip', 20)],
            },
    stage3={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            # 'node17': [('conv3', 16)],
            # 'node18': [('conv3', 17), ('skip', 16)],
            # 'node19': [('conv3', 18)],
            # 'node20': [('conv3', 19), ('skip', 18)]
            },
)

geno_image_short2 = Genotype(
    stage1={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 1), ('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 2), ('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 5), ('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 8), ('conv3', 10)],
            'node12': [('skip', 8), ('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 10), ('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)]},
    # 'node17': [('conv3', 16)],
    # 'node18': [('conv3', 17), ('skip', 16)],
    # 'node19': [('conv3', 18)],
    # 'node20': [('conv3', 19), ('skip', 18)]},
    stage2={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 1), ('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 2), ('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 5), ('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 8), ('conv3', 10)],
            'node12': [('skip', 8), ('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 10), ('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('skip', 13), ('conv3', 15), ('skip', 14)],
            'node17': [('skip', 13), ('conv3', 16)],
            'node18': [('conv3', 15), ('skip', 17)]
            # 'node19': [('conv3', 18)],
            # 'node20': [('conv3', 19), ('skip', 18)],
            # 'node21': [('conv3', 20)],
            # 'node22': [('conv3', 21), ('skip', 20)],
            },
    stage3={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 1), ('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 2), ('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 5), ('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 8), ('conv3', 10)],
            'node12': [('skip', 8), ('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 10), ('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)]
            # 'node17': [('conv3', 16)],
            # 'node18': [('conv3', 17), ('skip', 16)],
            # 'node19': [('conv3', 18)],
            # 'node20': [('conv3', 19), ('skip', 18)]
            },
)

geno_image_long = Genotype(
    stage1={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            'node17': [('conv3', 16)],
            'node18': [('conv3', 17), ('skip', 16)],
            'node19': [('conv3', 18)],
            'node20': [('conv3', 19), ('skip', 18)],
            'node21': [('conv3', 20)],
            'node22': [('conv3', 21), ('skip', 20)],
            'node23': [('conv3', 22)],
            'node24': [('conv3', 23), ('skip', 22)]},
    stage2={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            'node17': [('conv3', 16)],
            'node18': [('conv3', 17), ('skip', 16)],
            'node19': [('conv3', 18)],
            'node20': [('conv3', 19), ('skip', 18)],
            'node21': [('conv3', 20)],
            'node22': [('conv3', 21), ('skip', 20)],
            'node23': [('conv3', 22)],
            'node24': [('conv3', 23), ('skip', 22)],
            },
    stage3={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            'node17': [('conv3', 16)],
            'node18': [('conv3', 17), ('skip', 16)],
            'node19': [('conv3', 18)],
            'node20': [('conv3', 19), ('skip', 18)],
            'node21': [('conv3', 20)],
            'node22': [('conv3', 21), ('skip', 20)],
            'node23': [('conv3', 22)],
            'node24': [('conv3', 23), ('skip', 22)],
            },
)

geno_image_short4 = Genotype(
    stage1={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            # 'node17': [('conv3', 16)],
            # 'node18': [('conv3', 17), ('skip', 16)],
            # 'node19': [('conv3', 18)],
            # 'node20': [('conv3', 19), ('skip', 18)],
            # 'node21': [('conv3', 20)],
            # 'node22': [('conv3', 21), ('skip', 20)],
            # 'node23': [('conv3', 22)],
            # 'node24': [('conv3', 23), ('skip', 22)]
            },
    stage2={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            # 'node17': [('conv3', 16)],
            # 'node18': [('conv3', 17), ('skip', 16)],
            # 'node19': [('conv3', 18)],
            # 'node20': [('conv3', 19), ('skip', 18)],
            # 'node21': [('conv3', 20)],
            # 'node22': [('conv3', 21), ('skip', 20)],
            # 'node23': [('conv3', 22)],
            # 'node24': [('conv3', 23), ('skip', 22)],
            },
    stage3={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 0)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 2)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 4)],
            'node7': [('conv3', 6)],
            'node8': [('conv3', 7), ('skip', 6)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 8)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 10)],
            'node13': [('conv3', 12)],
            'node14': [('conv3', 13), ('skip', 12)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 15), ('skip', 14)],
            # 'node17': [('conv3', 16)],
            # 'node18': [('conv3', 17), ('skip', 16)],
            # 'node19': [('conv3', 18)],
            # 'node20': [('conv3', 19), ('skip', 18)],
            # 'node21': [('conv3', 20)],
            # 'node22': [('conv3', 21), ('skip', 20)],
            # 'node23': [('conv3', 22)],
            # 'node24': [('conv3', 23), ('skip', 22)],
            },
)

geno_image_normal = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            'node19': [('conv3', 18), ('skip', 18)],
            'node20': [('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
)

geno_image_normal_short = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            #'node14': [('conv3', 13), ('skip', 13)],
            # 'node15': [('conv3', 14), ('skip', 14)],
            # 'node16': [('conv3', 15), ('skip', 15)],
            # 'node17': [('conv3', 16), ('skip', 16)],
            # 'node18': [('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            # 'node15': [('conv3', 14), ('skip', 14)],
            # 'node16': [('conv3', 15), ('skip', 15)],
            # 'node17': [('conv3', 16), ('skip', 16)],
            # 'node18': [('conv3', 17), ('skip', 17)],
            # 'node19': [('conv3', 18), ('skip', 18)],
            # 'node20': [('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            #'node14': [('conv3', 13), ('skip', 13)],
            # 'node15': [('conv3', 14), ('skip', 14)],
            # 'node16': [('conv3', 15), ('skip', 15)],
            # 'node17': [('conv3', 16), ('skip', 16)],
            # 'node18': [('conv3', 17), ('skip', 17)],
            },
)

geno_image_normal_short2 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            #'node14': [('conv3', 13), ('skip', 13)],
            # 'node15': [('conv3', 14), ('skip', 14)],
            # 'node16': [('conv3', 15), ('skip', 15)],
            # 'node17': [('conv3', 16), ('skip', 16)],
            # 'node18': [('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            # 'node15': [('conv3', 14), ('skip', 14)],
            # 'node16': [('conv3', 15), ('skip', 15)],
            # 'node17': [('conv3', 16), ('skip', 16)],
            # 'node18': [('conv3', 17), ('skip', 17)],
            # 'node19': [('conv3', 18), ('skip', 18)],
            # 'node20': [('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            #'node14': [('conv3', 13), ('skip', 13)],
            # 'node15': [('conv3', 14), ('skip', 14)],
            # 'node16': [('conv3', 15), ('skip', 15)],
            # 'node17': [('conv3', 16), ('skip', 16)],
            # 'node18': [('conv3', 17), ('skip', 17)],
            },
)

geno_image_normal_short3 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            # 'node8': [('conv3', 7), ('skip', 7)],
            # 'node9': [('conv3', 8), ('skip', 8)],
            # 'node10': [('conv3', 9), ('skip', 9)],
            # 'node11': [('conv3', 10), ('skip', 10)],
            # 'node12': [('conv3', 11), ('skip', 11)],
            # 'node13': [('conv3', 12), ('skip', 12)],
            # 'node14': [('conv3', 13), ('skip', 13)],
            # 'node15': [('conv3', 14), ('skip', 14)],
            # 'node16': [('conv3', 15), ('skip', 15)],
            # 'node17': [('conv3', 16), ('skip', 16)],
            # 'node18': [('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            # 'node9': [('conv3', 8), ('skip', 8)],
            # 'node10': [('conv3', 9), ('skip', 9)],
            # 'node11': [('conv3', 10), ('skip', 10)],
            # 'node12': [('conv3', 11), ('skip', 11)],
            # 'node13': [('conv3', 12), ('skip', 12)],
            # 'node14': [('conv3', 13), ('skip', 13)],
            # 'node15': [('conv3', 14), ('skip', 14)],
            # 'node16': [('conv3', 15), ('skip', 15)],
            # 'node17': [('conv3', 16), ('skip', 16)],
            # 'node18': [('conv3', 17), ('skip', 17)],
            # 'node19': [('conv3', 18), ('skip', 18)],
            # 'node20': [('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            # 'node9': [('conv3', 8), ('skip', 8)],
            # 'node10': [('conv3', 9), ('skip', 9)],
            # 'node11': [('conv3', 10), ('skip', 10)],
            # 'node12': [('conv3', 11), ('skip', 11)],
            # 'node13': [('conv3', 12), ('skip', 12)],
            # 'node14': [('conv3', 13), ('skip', 13)],
            # 'node15': [('conv3', 14), ('skip', 14)],
            # 'node16': [('conv3', 15), ('skip', 15)],
            # 'node17': [('conv3', 16), ('skip', 16)],
            # 'node18': [('conv3', 17), ('skip', 17)],
            },
)

geno_image_search1 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('conv3', 2), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6),('conv3', 3), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('conv3', 9), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('conv3', 11), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('conv3', 12), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('conv3', 2), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6),('conv3', 3), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('conv3', 9), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('conv3', 11), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('conv3', 12), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('conv3', 14), ('skip', 17)],
            'node19': [('conv3', 18),  ('conv3', 16), ('skip', 18)],
            'node20': [('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('conv3', 2), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6),('conv3', 3), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('conv3', 9), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('conv3', 11), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('conv3', 12), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
)

geno_image_search2 = Genotype(
    stage1={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 0), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 4), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7)],
            'node9': [('skip', 7), ('conv3', 8)],
            'node10': [('conv3', 7), ('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 8), ('skip', 8), ('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 13), ('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('skip', 16), ('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('skip', 2), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('skip', 4), ('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9)],
            'node11': [('skip', 9), ('conv3', 10), ('skip', 10)],
            'node12': [('skip', 10), ('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 11), ('skip', 13), ('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            'node19': [('conv3', 18), ('conv3', 16), ('skip', 17), ('skip', 18)],
            'node20': [('conv3', 19), ('conv3', 16), ('skip', 19)],
            },
    stage3={'node1': [('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3)],
            'node5': [('skip', 3), ('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 6), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('skip', 8), ('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 8), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 10), ('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14)],
            'node16': [('skip', 14), ('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 15), ('conv3', 17), ('skip', 17)],
            },
)

geno_image_search3 = Genotype(
    stage1={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('skip', 0), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7)],
            'node9': [('skip', 7), ('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 7), ('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 8)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('skip', 16), ('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('skip', 2), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('skip', 4), ('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9)],
            'node11': [('skip', 9), ('conv3', 10), ('skip', 10)],
            'node12': [('skip', 10), ('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 11), ('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            'node19': [('conv3', 18), ('conv3', 16), ('skip', 18)],
            'node20': [('conv3', 16), ('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3)],
            'node5': [('skip', 3), ('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 6), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('skip', 8), ('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 8), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 10), ('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14)],
            'node16': [('skip', 14), ('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
)

geno_image_search4 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('skip', 1), ('conv3', 2), ('skip', 2)],
            'node4': [('skip', 0), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7)],
            'node9': [('skip', 7), ('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 7), ('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 8)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('skip', 16), ('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('skip', 2), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('skip', 4), ('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9)],
            'node11': [('skip', 9), ('conv3', 10), ('skip', 10)],
            'node12': [('skip', 10), ('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 11), ('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            'node19': [('conv3', 18), ('conv3', 16), ('skip', 18)],
            'node20': [('conv3', 16), ('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3)],
            'node5': [('skip', 3), ('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 6), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('skip', 8), ('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 8), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 10), ('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14)],
            'node16': [('skip', 14), ('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
)


geno_image_search4 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 0), ('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 1), ('skip', 2), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 3), ('skip', 4), ('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7)],
            'node9': [('skip', 7), ('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 7), ('conv3', 9), ('skip', 9)],
            'node11': [('skip', 9), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 10), ('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 16), ('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 0), ('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 1), ('skip', 2), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 3), ('skip', 4), ('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7)],
            'node9': [('skip', 7), ('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 7), ('conv3', 9), ('skip', 9)],
            'node11': [('skip', 9), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 10), ('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 15), ('conv3', 17), ('skip', 17)],
            'node19': [('conv3', 18), ('conv3', 18)],
            'node20': [('conv3', 19), ('skip', 19)]
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 0), ('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 1), ('skip', 2), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 3), ('skip', 4), ('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7)],
            'node9': [('skip', 7), ('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 7), ('conv3', 9), ('skip', 9)],
            'node11': [('skip', 9), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 10), ('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 16), ('conv3', 17), ('skip', 17)],
            },
)

geno_image_search5 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 0), ('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 1), ('skip', 2), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 3), ('skip', 4), ('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7)],
            'node9': [('skip', 7), ('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 7), ('conv3', 9), ('skip', 9)],
            'node11': [('skip', 9), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 10), ('conv3', 13), ('skip', 13)],
            #'node15': [('conv3', 14), ('skip', 14)],
            #'node16': [('conv3', 15), ('skip', 15)],
            #'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            #'node18': [('conv3', 16), ('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 0), ('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 1), ('skip', 2), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 3), ('skip', 4), ('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7)],
            'node9': [('skip', 7), ('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 7), ('conv3', 9), ('skip', 9)],
            'node11': [('skip', 9), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 10), ('conv3', 13), ('skip', 13)],
            #'node15': [('conv3', 14), ('skip', 14)],
            #'node16': [('conv3', 15), ('skip', 15)],
            #'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            #'node18': [('conv3', 15), ('conv3', 17), ('skip', 17)],
            #'node19': [('conv3', 18), ('conv3', 18)],
            #'node20': [('conv3', 19), ('skip', 19)]
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 0), ('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 1), ('skip', 2), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 3), ('skip', 4), ('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7)],
            'node9': [('skip', 7), ('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 7), ('conv3', 9), ('skip', 9)],
            'node11': [('skip', 9), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 10), ('conv3', 13), ('skip', 13)],
            #'node15': [('conv3', 14), ('skip', 14)],
            #'node16': [('conv3', 15), ('skip', 15)],
            #'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            #'node18': [('conv3', 16), ('conv3', 17), ('skip', 17)],
            },
)

taas_no_sample_l4 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node3': [('conv3', 1), ('skip', 1)],
            'node4': [('conv3', 0), ('skip', 0)],
            'node5': [('conv3', 1), ('skip', 1), ('conv3', 3)],
            'node6': [('conv3', 3), ('skip', 3)],
            'node7': [('conv3', 4), ('skip', 4)],
            'node9': [('conv3', 5), ('skip', 5)],
            'node10': [('conv3', 6), ('skip', 6)],
            'node11': [('conv3', 7), ('skip', 7), ('conv3', 9), ('skip', 9), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 9), ('skip', 9), ('conv3', 10)],
            'node14': [('conv3', 10), ('skip', 10), ('conv3', 11), ('skip', 11)],
            'node15': [('conv3', 11), ('skip', 11)],
            'node16': [('conv3', 12), ('skip', 12)],
            'node18': [('conv3', 14), ('skip', 14), ('conv3', 15), ('skip', 15), ('conv3', 16), ('skip', 16)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node3': [('conv3', 1), ('skip', 1)],
            'node4': [('conv3', 0), ('skip', 0)],
            'node5': [('conv3', 1), ('skip', 1), ('conv3', 3), ('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 3), ('skip', 3)],
            'node7': [('conv3', 4), ('skip', 4)],
            'node9': [('conv3', 5), ('skip', 5)],
            'node10': [('conv3', 6), ('skip', 6)],
            'node11': [('conv3', 7), ('skip', 7), ('conv3', 9), ('skip', 9)],
            'node12': [('conv3', 9), ('skip', 9), ('conv3', 10)],
            'node14': [('conv3', 10), ('skip', 10), ('conv3', 11), ('skip', 11)],
            'node16': [('conv3', 12), ('skip', 12)],
            'node18': [('conv3', 14), ('skip', 14), ('conv3', 14), ('skip', 14), ('conv3', 16), ('skip', 16)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node3': [('conv3', 0), ('conv3', 1), ('skip', 1)],
            'node4': [('conv3', 0), ('skip', 0), ('conv3', 1), ('skip', 1)],
            'node5': [('conv3', 1), ('skip', 1), ('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 3), ('skip', 3)],
            'node7': [('conv3', 4), ('skip', 4), ('conv3', 5), ('skip', 5)],
            'node9': [('conv3', 5), ('skip', 5)],
            'node10': [('conv3', 6), ('skip', 6)],
            'node11': [('conv3', 7), ('skip', 7)],
            'node12': [('conv3', 9), ('skip', 9)],
            'node14': [('conv3', 10), ('skip', 10), ('conv3', 11), ('skip', 11)],
            'node15': [('conv3', 11), ('skip', 11)],
            'node16': [('conv3', 12), ('skip', 12)],
            'node18': [('conv3', 14), ('skip', 14), ('conv3', 15), ('skip', 15), ('conv3', 16), ('skip', 16)],
            },
)

taas_sample_l4 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node4': [('conv3', 1), ('skip', 1), ('conv3', 2), ('skip', 2)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node9': [('conv3', 5), ('skip', 5)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 9),('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 13), ('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 14), ('skip', 14), ('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 0), ('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node9': [('conv3', 5), ('conv3', 5), ('conv3', 6), ('skip', 6)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 9), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 10), ('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 14), ('conv3', 17), ('skip', 17)],
            'node19': [('conv3', 15), ('conv3', 18), ('skip', 18)],
            'node20': [('conv3', 17), ('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node5': [('conv3', 1), ('conv3', 2), ('skip', 2)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 6), ('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 8), ('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 7), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 10), ('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 12), ('skip', 12), ('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
)

taas_sample_l4_v2 = Genotype(
    stage1={'node1': [('conv3', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node4': [('conv3', 0), ('conv3', 1), ('skip', 1), ('conv3', 2)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node9': [('conv3', 5), ('skip', 5)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 9), ('skip', 9), ('conv3', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 13), ('conv3', 14), ('skip', 15)],
            'node17': [('conv3', 14), ('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 14), ('skip', 14), ('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 0), ('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node9': [('conv3', 5), ('conv3', 5), ('conv3', 6), ('skip', 6)],
            'node10': [('skip', 9)],
            'node11': [('skip', 9), ('conv3', 10), ('skip', 10)],
            'node12': [('skip', 10), ('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 14), ('conv3', 17), ('skip', 17)],
            'node19': [('conv3', 15), ('conv3', 18), ('skip', 18)],
            'node20': [('conv3', 16), ('conv3', 17), ('skip', 17), ('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node5': [('conv3', 1), ('conv3', 2), ('skip', 2)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 6), ('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('skip', 8), ('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 7), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 10), ('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14)],
            'node16': [('conv3', 12), ('skip', 12), ('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
)


taas_no_sample_l4_v2 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 0), ('skip', 0)],
            'node3': [('conv3', 0), ('conv3', 1), ('skip', 1)],
            'node4': [('conv3', 1), ('skip', 1)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 2), ('skip', 2), ('conv3', 3)],
            'node7': [('conv3', 4), ('skip', 4)],
            'node8': [('conv3', 5), ('skip', 5), ('conv3', 6), ('skip', 6)],
            'node9': [('conv3', 5), ('skip', 5), ('conv3', 6), ('skip', 6)],
            'node11': [('conv3', 7), ('skip', 7), ('skip', 8)],
            'node12': [('conv3', 8), ('skip', 8), ('conv3', 9), ('skip', 9)],
            'node15': [('conv3', 11), ('skip', 11), ('conv3', 12), ('skip', 12)],
            'node18': [('conv3', 15), ('skip', 15)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node3': [('conv3', 0), ('conv3', 1), ('skip', 1)],
            'node4': [('conv3', 1), ('skip', 1)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 3), ('skip', 3)],
            'node7': [('conv3', 4), ('skip', 4)],
            'node8': [('conv3', 5), ('skip', 5), ('conv3', 6), ('skip', 6)],
            'node9': [('conv3', 5), ('skip', 5), ('conv3', 6), ('skip', 6)],
            'node10': [('conv3', 6), ('skip', 6), ('conv3', 7)],
            'node11': [('conv3', 7), ('skip', 7), ('conv3', 8), ('skip', 8)],
            'node12': [('conv3', 8), ('skip', 8), ('conv3', 9), ('skip', 9)],
            'node14': [('conv3', 10), ('conv3', 11), ('skip', 11)],
            'node15': [('conv3', 11), ('skip', 11), ('conv3', 12), ('skip', 12)],
            'node18': [('conv3', 14), ('skip', 14), ('conv3', 15), ('skip', 15)],
            'node20': [('conv3', 18), ('skip', 18)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node3': [('conv3', 0), ('conv3', 1), ('skip', 1)],
            'node4': [('conv3', 1), ('skip', 1)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 3), ('skip', 3), ('conv3', 4)],
            'node7': [('conv3', 4), ('skip', 4), ('conv3', 5), ('skip', 5)],
            'node8': [('conv3', 5), ('skip', 5), ('conv3', 6), ('skip', 6)],
            'node9': [('conv3', 5), ('skip', 5), ('conv3', 6), ('skip', 6)],
            'node11': [('conv3', 7), ('skip', 7), ('conv3', 8), ('skip', 8)],
            'node12': [('conv3', 8), ('skip', 8), ('conv3', 9), ('skip', 9)],
            'node15': [('conv3', 11), ('skip', 11), ('conv3', 12), ('skip', 12)],
            'node18': [('conv3', 15), ('skip', 15)],
            },
)


taas_no_sample_l8_keep1 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node4': [('conv3', 1), ('skip', 1)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node10': [('conv3', 5), ('skip', 5)],
            'node18': [('conv3', 10), ('skip', 10)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node3': [('conv3', 1), ('skip', 1)],
            'node8': [('conv3', 3), ('skip', 3)],
            'node14': [('conv3', 8), ('skip', 8)],
            'node20': [('conv3', 14), ('skip', 14)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node3': [('conv3', 1), ('skip', 1)],
            'node8': [('conv3', 3), ('skip', 3)],
            'node11': [('conv3', 8), ('skip', 8)],
            'node18': [('conv3', 11), ('skip', 11)],
            },
)

taas_no_sample_l8_keep2 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node3': [('conv3', 0), ('skip', 0), ('conv3', 1), ('skip', 1)],
            'node4': [('conv3', 0), ('skip', 0), ('conv3', 1), ('skip', 1)],
            'node5': [('conv3', 3), ('skip', 3), ('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 4), ('skip', 4), ('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 4), ('skip', 4), ('conv3', 6), ('skip', 6)],
            'node10': [('conv3', 3), ('skip', 3), ('conv3', 5), ('skip', 5)],
            'node11': [('conv3', 6), ('skip', 6), ('conv3', 7), ('skip', 7)],
            'node12': [('conv3', 7), ('skip', 7), ('conv3', 11), ('skip', 11)],
            'node18': [('conv3', 10), ('skip', 10), ('conv3', 12), ('skip', 12)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node3': [('conv3', 1), ('skip', 1)],
            'node4': [('conv3', 1), ('skip', 1), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 3), ('skip', 3), ('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 4), ('skip', 4), ('conv3', 5), ('skip', 5)],
            'node8': [('conv3', 3), ('skip', 3), ('conv3', 6), ('skip', 6)],
            'node12': [('conv3', 4), ('skip', 4), ('conv3', 8), ('skip', 8)],
            'node14': [('conv3', 6), ('skip', 6), ('conv3', 8), ('skip', 8)],
            'node20': [('conv3', 12), ('skip', 12), ('conv3', 14), ('skip', 14)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 0), ('skip', 0), ('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 1), ('skip', 1), ('conv3', 2), ('skip', 1)],
            'node6': [('conv3', 2), ('skip', 2), ('conv3', 3), ('skip', 3)],
            'node7': [('conv3', 3), ('skip', 3), ('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 3), ('skip', 3), ('conv3', 6), ('skip', 6)],
            'node11': [('conv3', 6), ('skip', 6), ('conv3', 8), ('skip', 8)],
            'node15': [('conv3', 7), ('skip', 7), ('conv3', 8), ('skip', 8)],
            'node16': [('conv3', 8), ('skip', 8), ('conv3', 15), ('skip', 15)],
            'node18': [('conv3', 11), ('skip', 11), ('conv3', 16), ('skip', 16)],
            },
)

if_l4_keep1 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node7': [('conv3', 3), ('skip', 3)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            'node19': [('conv3', 18), ('skip', 18)],
            'node20': [('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
)


taas_no_sample_l4_keep1 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node3': [('conv3', 1), ('skip', 1)],
            'node6': [('conv3', 3), ('skip', 3)],
            'node10': [('conv3', 6), ('skip', 6)],
            'node14': [('conv3', 10), ('skip', 10)],
            'node18': [('conv3', 14), ('skip', 14)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node4': [('conv3', 1), ('skip', 1)],
            'node8': [('conv3', 4), ('skip', 4)],
            'node12': [('conv3', 8), ('skip', 8)],
            'node16': [('conv3', 12), ('skip', 12)],
            'node20': [('conv3', 16), ('skip', 16)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node4': [('conv3', 1), ('skip', 1)],
            'node7': [('conv3', 4), ('skip', 4)],
            'node11': [('conv3', 7), ('skip', 7)],
            'node15': [('conv3', 11), ('skip', 11)],
            'node18': [('conv3', 15), ('skip', 15)],
            },
)

if_l6_keep1 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node7': [('conv3', 4), ('skip', 4)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node16': [('conv3', 13), ('skip', 13)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node7': [('conv3', 3), ('skip', 3)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node11': [('conv3', 8), ('skip', 8)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node18': [('conv3', 16), ('skip', 16)],
            'node19': [('conv3', 18), ('skip', 18)],
            'node20': [('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node7': [('conv3', 3), ('skip', 3)],
            'node8': [('conv3', 7), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node14': [('conv3', 13), ('skip', 13)],
            'node17': [('conv3', 14), ('skip', 14)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
)


if_l8_keep1 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node9': [('conv3', 6), ('skip', 6)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node17': [('conv3', 12), ('skip', 12)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node7': [('conv3', 3), ('skip', 3)],
            'node10': [('conv3', 7), ('skip', 7)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node17': [('conv3', 13), ('skip', 13)],
            'node18': [('conv3', 17), ('skip', 17)],
            'node19': [('conv3', 18), ('skip', 18)],
            'node20': [('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 6), ('skip', 6)],
            'node9': [('conv3', 7), ('skip', 7)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node14': [('conv3', 10), ('skip', 10)],
            'node15': [('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
)

if_l8 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 1), ('skip', 1), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 1), ('skip', 1), ('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 3), ('skip', 3), ('conv3', 5), ('skip', 5)],
            'node9': [('conv3', 6), ('skip', 6)],
            'node10': [('conv3', 4), ('skip', 4), ('conv3', 9), ('skip', 9)],
            'node11': [('conv3', 6), ('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 12), ('skip', 12)],
            'node15': [('conv3', 9), ('conv3', 11), ('skip', 11)],
            'node17': [('conv3', 11), ('conv3', 12), ('skip', 12), ('conv3', 15), ('skip', 15)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node7': [('conv3', 3), ('skip', 3)],
            'node8': [('conv3', 3), ('conv3', 7), ('skip', 7)],
            'node10': [('conv3', 7), ('skip', 7)],
            'node11': [('conv3', 10), ('skip', 10)],
            'node12': [('conv3', 4), ('skip', 4), ('conv3', 11), ('skip', 11)],
            'node13': [('conv3', 8), ('skip', 8), ('conv3', 12), ('skip', 12)],
            'node17': [('conv3', 13), ('skip', 13)],
            'node18': [('conv3', 12), ('conv3', 17), ('skip', 17)],
            'node19': [('conv3', 13), ('conv3', 18), ('skip', 18)],
            'node20': [('conv3', 13), ('skip', 13), ('conv3', 19), ('skip', 19)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 2), ('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 5), ('skip', 5)],
            'node7': [('conv3', 3), ('skip', 3), ('conv3', 6), ('skip', 6)],
            'node9': [('conv3', 7), ('skip', 7)],
            'node10': [('conv3', 9), ('skip', 9)],
            'node14': [('conv3', 7), ('skip', 7), ('conv3', 10), ('skip', 10)],
            'node15': [('conv3', 9), ('skip', 9), ('conv3', 14), ('skip', 14)],
            'node16': [('conv3', 10), ('conv3', 15), ('skip', 15)],
            'node17': [('conv3', 10), ('skip', 10), ('conv3', 16), ('skip', 16)],
            'node18': [('conv3', 17), ('skip', 17)],
            },
)

geno_cosine_similarity = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            'node6': [('conv3', 3), ('skip', 3), ('conv3', 4), ('skip', 4), ('conv3', 5), ('skip', 5)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            },
)

geno_cosine_similarity2 = Genotype(
    stage1={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            },
    stage2={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('none', 2), ],
            'node4': [('avg_pool_3x3', 2)],
            'node5': [('max_pool_3x3', 2)],
            'node6': [('skip', 2)],
            'node7': [('conv3', 2)],
            'node8': [('skip', 3), ('skip', 4), ('skip', 5), ('skip', 6), ('skip', 7)],
            'node9': [('conv3', 8), ('skip', 8)],
            #'node10': [('conv3', 9), ('skip', 9)],
            #'node11': [('conv3', 10), ('skip', 10)],
            #'node12': [('conv3', 11), ('skip', 11)],
            #'node13': [('conv3', 12), ('skip', 12)],
            },
    stage3={'node1': [('conv3', 0), ('skip', 0)],
            'node2': [('conv3', 1), ('skip', 1)],
            'node3': [('conv3', 2), ('skip', 2)],
            'node4': [('conv3', 3), ('skip', 3)],
            'node5': [('conv3', 4), ('skip', 4)],
            },
)

