import random
import itertools as it
import operator as op

random.seed(0.8784878514265989)

MAX_SAMPLES_PER_CLASS = 1000

with open('UCF101.dat') as fd:
    lines = fd.readlines()
lines = map(lambda line: line.split(), lines)
lines = map(lambda line: [int(line[0])] + map(float, line[1:]), lines)

get_label = lambda line: line[0]
groups = []
data = sorted(lines, key=get_label)
for _, g in it.groupby(data, get_label):
    groups.append(list(g))

def limit_group(group):
    random.shuffle(group)
    return group[:MAX_SAMPLES_PER_CLASS]

groups = map(limit_group, groups)

lines = reduce(op.concat, groups)

with open('UCF101_limited.dat', 'w') as fd:
    for line in lines:
        fd.write('{}'.format(line[0] + 1))
        for feat in line[1:]:
            fd.write(' {}'.format(feat))
        fd.write('\n')
